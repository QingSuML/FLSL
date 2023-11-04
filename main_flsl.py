import argparse
import os
import sys
import time
import datetime
import math
import random
import json
from pathlib import Path
from functools import partial

import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
import torch.backends.cudnn as cudnn

from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import FLSLhead


torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    parser = argparse.ArgumentParser('FLSL', add_help=False)


    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/deit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small with patch size of 16.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_small).""") 
    parser.add_argument('--out_dim', default=8192, type=int, help="""Number of centroids,i.e., 
                        dimensionality of the FLSL head output. 
                        For complex and large datasets larger values work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the FLSL head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small 
        and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with 
        cosine schedule.We recommend setting a higher value with small batches: for example 
        use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="""Whether to use batch normalizations in projection head (Default: False)""")
    parser.add_argument('--teacher_centering', default=False, type=utils.bool_flag,
        help="""Whether to use teacher centering rather than the KL regularizor 
        (Default: False)""")
    parser.add_argument('--volume_maximization', default=True, type=utils.bool_flag,
        help="""Whether to use the KL volume maximization regularizor in the objective.
        (Default: True)""")
    parser.add_argument('--post_layernorm', type=utils.bool_flag, default=True, 
        help="""Apply layer normalization to the output from the last ViT layer""")
    parser.add_argument('--layernorm_on_mode', type=utils.bool_flag, default=False, 
        help="""Applying layer normalization to modes of randomly sampled tokens 
        other than all tokens,""")

    # Temperature parameters
    parser.add_argument('--student_temp', default=0.1, type=float,
        help="""Student temperature: 0.1 works well in most cases. Try decreasing it together
        with teacher temperature if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value 
        (after linear warmup) of the teacher temperature. For most experiments, anything 
        above 0.07 is unstable. We recommend starting with the default value of 0.04 and 
        increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help="""Number of warmup epochs for the teacher temperature (Default: 30).""")
    
    # Meanshift layer parameters
    parser.add_argument('--meanshift_sa_temp', type=float, default=0.1/math.sqrt(384), 
        help="""Temperature for the non-parametric self-attention MS layer. 
        Default value is 0.1/sqrt(d=384) ~ 0.005. For higher semantic level,
        smaller values are recommended.""")
    parser.add_argument('--meanshift_ca_temp', type=float, default=0.2/math.sqrt(384), 
        help="""Temperature for the non-parametric cross-attention MS layer. 
        Default value is 0.2/sqrt(d=384) ~ 0.01. Using smaller value than 
        meanshift_sa_temp leads to collapse""")
    parser.add_argument('--meanshift_sa_temp_start', default=None, help="""Initial value of the
        temperature of self-attention MS layer.""")
    parser.add_argument('--meanshift_ca_temp_start', default=None, help="""Initial value of the
        temperature of cross-attention MS layer.""")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether 
        or not to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training 
        with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number o f epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=5e-4, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=0.1, type=float,
        help="""Percentage of epochs for the linear learning-rate warm up""")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw 
        with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop and random pooling parameters
    parser.add_argument('--global_crops_scale_t', type=float, nargs='+', default=(0.8, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping for teacher model.""")
    parser.add_argument('--global_crops_scale_s', type=float, nargs='+', default=(0.5, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping for teacher model. When disabling multi-crop 
        (--local_crops_number 0), we recommand using a wider range of scale 
        ("--global_crops_scale 0.15 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=2, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.1, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--random_sampling', type=utils.bool_flag, default=False)
    parser.add_argument('--sampling_window_size_g', type=int, default=3, 
        help="""Window size of random pooling for global crops""")
    parser.add_argument('--sampling_window_size_l', type=int, default=2, 
        help="""Window size of random pooling for local crops""")
    parser.add_argument('--global_resize', type=int, default=224)
    parser.add_argument('--local_resize', type=int, default=96)
    parser.add_argument('--sin_cos_pos_emb', type=utils.bool_flag, default=True, 
        help="""Use sinusoidal PE other than a leanable PE""")

    #Loss weights
    parser.add_argument('--loss_coefficients', type=float, nargs='+', default=(0.03, 1.0, 5.0),
        help="""Coefficients for intra-veiw clustering, inter-view clustering, and volume 
        maximization regularizer, respectively. We recommend smaller intra-view coefficient 
        and larger VMR coefficientsif the loss is unstable.""")
    parser.add_argument('--start_intraview', type=float, default=None, help="""Initial value for 
        the coefficient of intra-view clustering. We recommend smaller values, e.g. 0.0, -0.03, 
        for the model to learn higher semantic level.""")
    
    #Multistaged semantic constraint
    parser.add_argument('--model_previous_stage', default=None, type=str, 
                        help="""The directory of the model checkpoint from previous training stage.""")
    parser.add_argument('--attention_level', type=int, default=11, help="""The level of features used
                        to create attention map for semantic constraints (Default: 11)""")
    parser.add_argument('--prev_ca_temp', type=float, default= 0.2/math.sqrt(384), 
        help="""Temperature of cross-attention on fixed features from previous stage.""")
    parser.add_argument('--mixture_bias', type=float, default=1.0, help="""mixture bias for 
        cross-attention map calculated with the fixed feature.""")
    parser.add_argument('--map_sharpness', type=float, default = 1e4, help="""Sharpness in 
        sigmoid to push the values of attention map to 0 and 1.""")

    # Training miscellanies
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help="""Please specify path to the ImageNet training data.""")
    parser.add_argument('--output_dir', default=".", type=str, help="""Path to save logs and checkpoints.""")
    parser.add_argument('--saveckp_freq', default=20, type=int, help="""Save checkpoint every x epochs.""")
    parser.add_argument('--seed', default=0, type=int, help="""Random seed.""")
    parser.add_argument('--num_workers', default=12, type=int, 
                        help="""Number of data loading workers per GPU.""")
    parser.add_argument("--dist_url", default="env://", type=str, 
                        help="""url used to set up distributed training; 
                        see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--wandb', type=utils.bool_flag, default=True, help="""Use WandB for loss curve 
                        visualization and monitoring.""")
    parser.add_argument('--project_name', type=str, default='flsl')
    
    return parser


def train_flsl(args):
    
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ============
    transform = DataAugmentationFLSL(
        global_crops_scale_teacher=args.global_crops_scale_t,
        global_crops_scale_student=args.global_crops_scale_s,
        local_crops_scale=args.local_crops_scale,
        local_crops_number=args.local_crops_number,
        global_resize=args.global_resize,
        local_resize=args.local_resize,
        sampling_window_size_g=args.sampling_window_size_g,
        sampling_window_size_l=args.sampling_window_size_l,
        patch_size=args.patch_size,
        random_sampling=args.random_sampling,
        )
    
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
        #make the ViT model exclusive of the cls token
        sincos_pos = None
        if args.sin_cos_pos_emb:
            sincos_pos = utils.get_2d_sincos_pos_embed(embed_dim, 
                        args.global_resize // args.patch_size,
                        cls_token=False)
        vits.make_vit_noclass(student, sincos_pos, hook={'norm': 'norm'}, 
                                 post_layernorm=args.post_layernorm)
        vits.make_vit_noclass(teacher, sincos_pos, hook={'norm': 'norm'}, 
                                 post_layernorm=args.post_layernorm)
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.ModelWrapper(student, 
                                FLSLhead(embed_dim, 
                                        args.out_dim,
                                        use_bn=args.use_bn_in_head,
                                        norm_last_layer=args.norm_last_layer,),
                                mode_norm=args.layernorm_on_mode)
    teacher = utils.ModelWrapper(teacher,
                                FLSLhead(embed_dim, 
                                         args.out_dim, 
                                         use_bn=args.use_bn_in_head,
                                         norm_last_layer=args.norm_last_layer,),
                                mode_norm=args.layernorm_on_mode)
                                
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # stop gradient for teacher
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")
    
    # ============ Multi-staged training for enriched semantics ============
    teacher_prev = None
    if args.model_previous_stage:
        # building teacher network trained from previous stage... 
        if args.arch in vits.__dict__.keys():
            teacher_prev = vits.__dict__[args.arch](patch_size=args.patch_size)
        elif args.arch in torchvision_models.__dict__.keys():
            teacher_prev = torchvision_models.__dict__[args.arch]()
            
        utils.load_pretrained_weights(teacher_prev, args.model_previous_stage, "teacher")
        vits.make_vit_noclass(teacher_prev, post_layernorm=False)
        # set the attention level to be used... 
        teacher_prev.blocks = teacher_prev.blocks[:args.attention_level]
        teacher_prev.cuda()
        teacher_prev.eval()
        print(f"Teacher model from previous stage is built: it is {args.arch} network.")
        
    # ============ preparing loss ... ============
    VMR = utils.NonEmptyPriorKL(args.out_dim) if args.volume_maximization else None
    teacher_centering = utils.Centering(args.out_dim) if args.teacher_centering else None
    flsl_loss = FLSLLoss(args.student_temp,
                         utils.temp_scheduler(args.warmup_teacher_temp, args.teacher_temp, 
                                              args.warmup_teacher_temp_epochs, args.epochs),
                         VMR,
                         teacher_centering,
                         coefficients=args.loss_coefficients).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # suitable for ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # suitable for convnet and large batches
    # for mixed precision training
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # ============ init schedulers ... ============
    nipe = len(data_loader) 
    lr_schedule = utils.cosine_scheduler(
        args.lr * args.batch_size_per_gpu * utils.get_world_size()/ 256.,  #linear scaling rule
        args.min_lr,
        args.epochs, nipe, warmup_epochs=int(args.warmup_epochs * args.epochs),
    )
    wd_schedule = utils.cosine_scheduler(args.weight_decay, args.weight_decay_end,
                                         args.epochs, nipe,)
    
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1.,
                                               args.epochs, nipe)
    
    #first-level clustering schedule
    start_intraview = args.start_intraview or args.loss_coefficients[0]
    ups_schedule = utils.cosine_scheduler(start_intraview, args.loss_coefficients[0],
                                             args.epochs, nipe)
    #meanshift temperature schedule
    ms_sa_temp_start = args.meanshift_sa_temp_start or args.meanshift_sa_temp
    ms_sa_temp_schedule = utils.cosine_scheduler(ms_sa_temp_start, args.meanshift_sa_temp,
                                                 args.epochs, nipe)
    ms_ca_temp_start = args.meanshift_ca_temp_start or args.meanshift_ca_temp
    ms_ca_temp_schedule = utils.cosine_scheduler(ms_ca_temp_start, args.meanshift_ca_temp,
                                                 args.epochs, nipe)
    #multi-stage attention mixture schedule
    if args.model_previous_stage:
        mixture_schedule = utils.cosine_scheduler(0., 1-args.mixture_bias,
                                                  args.epochs, nipe)
    else:
        mixture_schedule = [1.] * (args.epochs * nipe)
    
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        flsl_loss=flsl_loss,
    )
    start_epoch = to_restore["epoch"]

    # ============ start training ... ============
    start_time = time.time()
    print("Starting FLSL training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of FLSL ... ============
        train_stats = train_one_epoch(nipe, epoch,
                                      student, 
                                      teacher, teacher_without_ddp, 
                                      teacher_prev, 
                                      flsl_loss,
                                      data_loader, optimizer, fp16_scaler,
                                      lr_schedule, wd_schedule, 
                                      momentum_schedule, ups_schedule,
                                      ms_sa_temp_schedule, ms_ca_temp_schedule,
                                      mixture_schedule,
                                      args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'flsl_loss': flsl_loss.state_dict(),
        }
        
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    print('Training time {}'.format(total_time_str))


def train_one_epoch(nipe, epoch,
                    student, 
                    teacher, teacher_without_ddp, 
                    flsl_loss, 
                    data_loader, optimizer, fp16_scaler,
                    lr_schedule, wd_schedule, 
                    momentum_schedule, ups_schedule, 
                    ms_sa_temp_schedule=None, ms_ca_temp_schedule=None,
                    teacher_prev=None, 
                    mixture_schedule=None,
                    args=None):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        
        # update weight decay, learning rate, intraview clustering coefficient
        # according to their schedule
        it = nipe * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        flsl_loss.coefficients[0] = ups_schedule[it]
                
        # move images to gpu
        augs = [im.cuda(non_blocking=True) for im in images['augs']]
        masks = [mask.cuda(non_blocking=True) for mask in images['masks']]
        unisize_idx = torch.cumsum(
                        torch.unique_consecutive( 
                        torch.tensor([x.shape[-1] for x in augs]), 
                        return_counts=True,)[1], 0)
            
        # teacher and student forward passes + compute flsl loss
        if teacher_prev is None:
            loss, loss_specs = train_one_batch(it, epoch, student, teacher, flsl_loss, augs, masks, unisize_idx,
                                                sa_temp_schedule=ms_sa_temp_schedule, ca_temp_schedule=ms_ca_temp_schedule, 
                                                layernorm_on_mode=args.layernorm_on_mode, autocast=fp16_scaler is not None)
        else:
            crops = [im.cuda(non_blocking=True) for im in images['crops']]
            loss, loss_specs = train_one_batch_multistage(it, epoch, student, teacher, flsl_loss, 
                                                          teacher_prev, mixture_schedule, 
                                                          crops, augs, masks, unisize_idx,
                                                          sa_temp_schedule=ms_sa_temp_schedule, 
                                                          ca_temp_schedule=ms_ca_temp_schedule, 
                                                          layernorm_on_mode=args.layernorm_on_mode, 
                                                          autocast=fp16_scaler is not None)
                
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss_specs[0]}, {loss_specs[1]}, {loss_specs[2]}, {loss_specs[3]} \
                  stopping training", force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        
        # compute gradient
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_meanshift=loss_specs[0])
        metric_logger.update(loss_kmeans=loss_specs[1])
        metric_logger.update(loss_vmr=loss_specs[2])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(ups=ups_schedule[it])

        if args.wandb and utils.is_main_process():
            wandb.log({k: meter.avg for k, meter in metric_logger.meters.items()})
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_batch(it, epoch,
                    student, teacher, flsl_loss,
                    augs, masks, unisize_idx,
                    sa_temp_schedule=None, ca_temp_schedule=None, 
                    layernorm_on_mode=False,
                    autocast=True):
    
    with torch.cuda.amp.autocast(autocast):
        out_s, out_mode_s, out_pred_s, out_pred_t = [torch.empty(0).cuda()] * 4
        
        enc_t = teacher.forward_backbone(augs[0])
        bt, N, C = enc_t.shape
                
        start_idx = 1
        for end_idx in unisize_idx:
            
            b = end_idx - start_idx
            mask = torch.cat(masks[start_idx: end_idx])

            enc_s = student.forward_backbone(torch.cat(augs[start_idx : end_idx]))
            
            _s, mode_s = utils.flsl_meanshift(enc_s, enc_s, mask, temp=sa_temp_schedule[it])
            _, mode_t = utils.flsl_meanshift(enc_s, enc_t.unsqueeze(0).expand(b,-1,-1,-1).reshape(b*bt,N,-1), 
                                              mask, temp=ca_temp_schedule[it])
            if layernorm_on_mode:
                pred_s = student.forward_head(student.forward_norm(mode_s))
                pred_t = teacher.forward_head(teacher.forward_norm(mode_t))
            else:   
                pred_s = student.forward_head(mode_s)
                pred_t = teacher.forward_head(mode_t)
            
            out_s = torch.cat((out_s, _s.view(-1,C)))
            out_mode_s = torch.cat((out_mode_s, mode_s.view(-1,C)))
            out_pred_s = torch.cat((out_pred_s, pred_s.view(-1,pred_s.size(-1))))
            out_pred_t = torch.cat((out_pred_t, pred_t.view(-1,pred_t.size(-1))))
            
            start_idx = end_idx
        
        return flsl_loss(out_s, out_mode_s, out_pred_s, out_pred_t, epoch)


def train_one_batch_multistage(it, epoch,
                               student, teacher, flsl_loss,
                               teacher_prev, mixture_schedule,
                               crops, augs, masks, unisize_idx,
                               sa_temp_schedule=None, ca_temp_schedule=None, 
                               layernorm_on_mode=False,
                               autocast=False):
            
    with torch.cuda.amp.autocast(autocast):
        
        out_s, out_mode_s, out_pred_s, out_pred_t = [torch.empty(0).cuda()] * 4
        
        enc_t = teacher(augs[0], module='backbone')
        feature_prev_k = teacher_prev(crops[0])
        bt, N, C = enc_t.shape
        
        start_idx = 1
        for end_idx in unisize_idx:
            
            b = end_idx - start_idx
            mask = torch.cat(masks[start_idx: end_idx])

            feature_prev_q = teacher_prev(torch.cat(crops[start_idx : end_idx]))
            attn_prev = utils.attention_map(feature_prev_q, 
                                            feature_prev_k.unsqueeze(0).expand(b,-1,-1,-1).reshape(b*bt,N,-1), 
                                            mask, temp=args.prev_ca_temp, sharpness=args.map_sharpness).detach()
            
            enc_s = student(torch.cat(augs[start_idx : end_idx]), module='backbone')
            
            _s, mode_s = utils.flsl_meanshift(enc_s, enc_s, mask, temp=sa_temp_schedule[it])
            _, mode_t = utils.flsl_meanshift(enc_s, enc_t.unsqueeze(0).expand(b,-1,-1,-1).reshape(b*bt,N,-1), 
                                             mask, tgt_attn=attn_prev,
                                             temp=ca_temp_schedule[it], attn_mixture=mixture_schedule[it])
            if layernorm_on_mode:
                pred_s = student.forward_head(student.forward_norm(mode_s))
                pred_t = teacher.forward_head(teacher.forward_norm(mode_t))
            else:   
                pred_s = student.forward_head(mode_s)
                pred_t = teacher.forward_head(mode_t)
            
            out_s = torch.cat((out_s, _s.view(-1,C)))
            out_mode_s = torch.cat((out_mode_s, mode_s.view(-1,C)))
            out_pred_s = torch.cat((out_pred_s, pred_s.view(-1,pred_s.size(-1))))
            out_pred_t = torch.cat((out_pred_t, pred_t.view(-1,pred_t.size(-1))))
            
            start_idx = end_idx
            
        return flsl_loss(out_s, out_mode_s, out_pred_s, out_pred_t, epoch)


class DataAugmentationFLSL(object):
    def __init__(self, 
                 global_crops_scale_teacher=(.8, 1.), 
                 global_crops_scale_student=(.5, 1.),
                 local_crops_scale=(.05, .4), 
                 local_crops_number=2,
                 global_resize=224,
                 local_resize=96,
                 sampling_window_size_g=3,
                 sampling_window_size_l=2,
                 patch_size=16,
                 random_sampling=False):
        
        self.global_resize=global_resize
        self.local_resize=local_resize
        self.local_crops_number = local_crops_number
        self.sampling_window_size_g=sampling_window_size_g
        self.sampling_window_size_l=sampling_window_size_l
        self.random_sampling = random_sampling
        self.global_feature_size = global_resize // patch_size
        self.local_feature_size = local_resize //patch_size
        
        # color jitter, grayscale, random flip
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        # normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # global crop
        self.global_crop_t = transforms.RandomResizedCrop(self.global_resize, 
                                scale=global_crops_scale_teacher, 
                                interpolation=transforms.InterpolationMode.BICUBIC)
        self.global_crop_s = transforms.RandomResizedCrop(self.global_resize, 
                                scale=global_crops_scale_student, 
                                interpolation=transforms.InterpolationMode.BICUBIC)
        self.local_crop = transforms.RandomResizedCrop(self.local_resize, 
                                scale=local_crops_scale, 
                                interpolation=transforms.InterpolationMode.BICUBIC)
        
        #transform
        self.transformation = transforms.Compose([flip_and_color_jitter,
                                                  utils.RandomSelect(
                                                      utils.GaussianBlur(p=0.5),
                                                      transforms.Compose([
                                                          utils.GaussianBlur(.1),
                                                          utils.Solarization(.2),
                                                      ]),
                                                      p=0.5,
                                                  ),
                                                  self.normalize,
                                                 ])
    
    def _sampling_mask(self, feature_size, window_size=2):
        
        grid_size = feature_size // window_size
        sample_feature_size = grid_size * window_size
        window_length = window_size ** 2
        
        selections = torch.randint(0, window_length, (grid_size ** 2,)) if self.random_sampling \
            else torch.zeros(grid_size ** 2, dtype=torch.int64)
        mask = F.one_hot(selections, num_classes=window_length).type(torch.float32)\
                .reshape(-1, grid_size, grid_size, window_size, window_size)\
                .transpose(-2,-3).reshape(sample_feature_size, sample_feature_size)
        if feature_size > sample_feature_size:
            mask_s = mask
            mask = torch.zeros(feature_size, feature_size)
            p = random.random()
            if p < .25:
                mask[:sample_feature_size,:sample_feature_size] = mask_s
            elif p >= .25 and p < .5:
                mask[:sample_feature_size,-sample_feature_size:] = mask_s
            elif p >=.5 and p < .75:
                mask[-sample_feature_size:,:sample_feature_size] = mask_s
            elif p >= .75:
                mask[-sample_feature_size:,-sample_feature_size:] = mask_s
        return (mask>0).flatten(0) #[N]
    
    
    def __call__(self, image):
        sample = defaultdict(list)
        sample['crops'].append(self.global_crop_t(image)) 
        sample['crops'].append(self.global_crop_s(sample['crops'][0]))
        sample['masks'].append(torch.empty(0))
        sample['masks'].append(self._sampling_mask(self.global_feature_size, 
                                                  window_size=self.sampling_window_size_g))
        for i in range(self.local_crops_number):
            sample['crops'].append(self.local_crop(sample['crops'][0]))
            sample['masks'].append(self._sampling_mask(self.local_feature_size,
                                                      window_size=self.sampling_window_size_l))
        for i, crop in enumerate(sample['crops']):
            sample['augs'].append(self.transformation(crop))
            sample['crops'][i] = self.normalize(crop)
        return sample


class FLSLLoss(nn.Module):
    def __init__(self, 
                 temp_s,
                 temp_t_schedule, 
                 volume_maximization=None, 
                 centering=None, 
                 coefficients=[.3, 1., 5.]):
        
        super().__init__()
        self.temp_s = temp_s
        self.temp_t_schedule =  temp_t_schedule
        self.volume_maximization = volume_maximization
        self.centering = centering
        self.coefficients = coefficients    

    def forward(self, x, mode, student_pred, teacher_pred, epoch):
        #first-level clustering loss
        x_n = nn.functional.normalize(x, dim=-1, p=2)
        mode_n = nn.functional.normalize(mode, dim=-1, p=2)
        ms_loss = 2. - 2. * torch.sum(x_n * mode_n, dim=-1).mean()
        
        if self.centering:
            teacher_pred = self.centering(teacher_pred)

        #second-level clustering loss
        s_out = F.softmax(student_pred / self.temp_s, dim=-1)
        t_out = F.softmax(teacher_pred.detach() / self.temp_t_schedule[epoch], dim=-1)
        kmeans_loss = - torch.sum(t_out * s_out.log(), dim=-1).mean()

        #second-level VMR
        vmr = self.volume_maximization(s_out) if self.volume_maximization else 0.
        
        ups, eta, gam = self.coefficients
        total_loss = ups * ms_loss + eta * kmeans_loss + gam * vmr
        
        return total_loss, (ms_loss, kmeans_loss, vmr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FLSL', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.wandb:
        import wandb
        wandb.init(project=args.project_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_flsl(args)
