import os
import sys
import json
import argparse
from PIL import Image

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader


import utils
import vision_transformer as vits


def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation with bbox-aligned k-NN on ImageNet', add_help=False)
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--pin_memory', default=True, type=utils.bool_flag,)
    parser.add_argument('--non_blocking', default=True, type=utils.bool_flag,)
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--checkpoint_key', default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--dist_url', default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument('--local_rank', default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--ann_path', default='/path/to/imagenet/annotations/', type=str)
    parser.add_argument('--target_size', type=int, default=256, help="Target size of the input images.")
    parser.add_argument('--centercrop_size', type=int, default=224, 
                        help="The size of the center crop of an image resized to the target size.")
    parser.add_argument('--grid_size', default=3, type=int, help="Dimension size of pooling grid.")

    return parser
    
    
def extract_feature_pipeline(args):
    # ============ preparing data ... ============
    # standard image preprocessing
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.target_size, interpolation=3),
        pth_transforms.CenterCrop(args.centercrop_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    dataset_train = BboxAlignedDataset(args.ann_path, "cls_to_idx.json", transform=transform,
                                       ann_file_bbox='imagenet1k_train_bboxaligned.json')
    dataset_val = BboxAlignedDataset(args.ann_path, "cls_to_idx.json", transform=transform,
                                     ann_file_bbox='imagenet1k_val_bboxaligned.json')
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        vits.make_vit_noclass(model)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, use_cuda=args.use_cuda, 
                                      set_non_blocking=args.non_blocking) #L,np,C
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, use_cuda=args.use_cuda, 
                                     set_non_blocking=args.non_blocking)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=2, p=2)
        test_features = nn.functional.normalize(test_features, dim=2, p=2)

    train_labels = torch.tensor([s['class_idx'] for s in dataset_train.samples]).unsqueeze(1)\
                   .expand(-1, train_features.size(1)).long() #L,np
    test_labels = torch.tensor([s['class_idx'] for s in dataset_val.samples]).unsqueeze(1)\
                   .expand(-1, test_features.size(1)).long()
    
    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
        
    return train_features, test_features, train_labels, test_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False, set_non_blocking=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for image, index, patch_ids, _, _ in metric_logger.log_every(data_loader, 10):
        image = image.cuda(non_blocking=set_non_blocking) #set non_blocking to True with pinned memory
        index = index.cuda(non_blocking=set_non_blocking)
        patch_ids = torch.cat([x.unsqueeze(dim=1) for x in patch_ids], dim=1).cuda(non_blocking=set_non_blocking)
        feats = (model(image)[torch.arange(image.shape[0])[..., None], patch_ids]).clone()
        patch_num = feats.shape[1]

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), patch_num, feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=set_non_blocking)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.shape[0],
            feats.shape[1],
            feats.shape[2],
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()
        output_all = torch.cat(output_l)

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, output_all)
            else:
                features.index_copy_(0, index_all.cpu(), output_all.cpu())

    return features
    

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes=1000):
    #train_features: L, np, C
    #test_features: L', np, C
    #train_labels: L, np
    #test_labels: L', np
    
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.reshape(-1, train_features.size(-1)).transpose(-1,-2) #C,L*np
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images)] #B,np,C
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)] #B,np
        batch_size = targets.shape[0]
        num_patches = features.shape[1]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.matmul(features, train_features) # B,np,L*np
        distances, indices = similarity.topk(k, largest=True, sorted=True) # B,np,k, B,np,k
        candidates = train_labels.reshape(1, 1, -1).expand(batch_size, num_patches, -1) # B,np,L*np
        retrieved_neighbors = torch.gather(candidates, 2, indices).transpose(1,2) # B,k,np

        retrieval_one_hot.resize_(batch_size * k, num_patches, num_classes).zero_() #B*k,np,L
        retrieval_one_hot.scatter_(2, retrieved_neighbors.reshape(-1, num_patches, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_patches, num_classes),
                distances_transform.transpose(1,2).view(batch_size, -1, num_patches, 1),
            ),
            1,
        ) #B,np,L
        _, predictions = probs.sort(-1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.unsqueeze(-1)) #B,np,L True/False
        correct = correct.any(1) #B,L
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5
    
    
class BboxAlignedDataset(VisionDataset):
    def __init__(self, root, class_to_idx, ann_file = None,
                 loader=default_loader, transform = None,
                 target_transform = None,
                 ann_file_bbox=None
                ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.class_to_idx = json.load(open(os.path.join(self.root, class_to_idx), 'r'))
        self.classes = list(self.class_to_idx)
        
        """
        samples is a list of dictionaries, each has keys:
        'path': directory to the selected image
        'bbox': the largest bounding box of that image
        'patch_ids': patch indices of flattened dense features
        'class_idx': class label of the largest bounding box
        """
        assert not (ann_file and ann_file_bbox)
        if ann_file:
            self.samples = make_bbox_aligned_dataset(os.path.join(self.root, ann_file), 
                                                     self.class_to_idx)
        elif ann_file_bbox:
            self.samples = json.load(open(os.path.join(self.root, ann_file_bbox), 'r'))
        self.loader = loader
        self.targets = [s['class_idx'] for s in self.samples]
    
    def __getitem__(self, index):
        
        sample = self.samples[index]
        
        target = sample['class_idx']
        image = self.loader(sample['path'])
        patch_ids = sample['patch_ids']
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, index, patch_ids, target, sample['bbox']

    def __len__(self) -> int:
        return len(self.samples)
    
    
def make_bbox_aligned_dataset(ann_file_dir, class_to_idx):
    """
    ann, a list of dictionary with keys:
        'id', imagename {class}_{image number}
        'filename', image path
        'width', image width
        'height', image height
        'bbox', list of bounding boxes in the image in VOC format
        'bbox_cls' list of classes corresponding to 'bbox'
    """
    
    ann_list = json.load(open(ann_file_dir, 'r'))
    instances = []
    for ann in ann_list:
        sample=dict()
        sample['path'] = ann['filename']
        bbox_no, bbox_cls, patch_ids = partition_max_bbox(ann['width'], ann['height'],
                                                          ann['bbox'], 
                                                          ann['bbox_cls'])
        sample['bbox_no'] = bbox_no.item()
        sample['bbox'] = ann['bbox'][bbox_no]
        sample['patch_ids'] = patch_ids.tolist()
        sample['class_idx'] = class_to_idx[bbox_cls]
        instances.append(sample)
        
    return instances


def partition_max_bbox(width, height, bboxes, classes, resize=256, crop_size=224, patch_size=16, grid_num=3):
    scale_x, scale_y = resize/width, resize/height
    shift_x = shift_y = int((crop_size - resize) / 2)
    bbox = torch.tensor(bboxes, dtype=torch.float)
    
    #transform the bounding boxes to target size
    bbox[:,0::2] = scale_x * bbox[:,0::2] + shift_x
    bbox[:,1::2] = scale_y * bbox[:,1::2] + shift_y
    bbox[bbox<0] = 0
    bbox[bbox>crop_size] = crop_size - 1e-3
    bbox /= patch_size
    
    max_no = torch.prod(bbox[:, 2:] - bbox[:,:2], dim=1).argmax()
    max_bbox = bbox[max_no]
    
    #bounding box-aligned patch postion
    grid_res = 3
    gridw, gridh = (max_bbox[2:] - max_bbox[:2])/grid_res
    mesh_size = int(crop_size / patch_size)

    xmin, ymin = max_bbox[:2]
    grid_ctrs = torch.tensor([[[xmin + (i + 0.5) * gridw, ymin + (j + 0.5) * gridh] 
             for i in range(grid_num)] for j in range(grid_num)])
    patch_id = torch.floor(grid_ctrs).reshape(-1, 2)
    patch_id = patch_id[:,1] * mesh_size + patch_id[:,0]
   
    return max_no, classes[max_no], patch_id.type(torch.long)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with bbox-aligned k-NN on ImageNet', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    utils.init_distributed_mode(args)
    
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            top1, top5 = knn_classifier(train_features, train_labels,
                test_features, test_labels, k, args.temperature)
            print(f"{k}-NN classifier result: Top1: {top1}, Top5: {top5}")