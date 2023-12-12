# FLSL: Feature-Level Self-supervised Learning

PyTorch implementation and pretrained models for FLSL. For details, see **FLSL: Feature-Level Self-supervised Learning**.  
[[`NeurIPS2023`](https://openreview.net/pdf?id=8pOBo5NgTQ)] [[`video`](https://www.youtube.com/watch?v=h12345?)]

<div align="center">
  <img width="100%" alt="FLSL Framework" src=".github/flsl_framework_large.gif">
</div>

## Training

### Documentation
Please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset. This codebase has been developed with python version 3.9, PyTorch version 1.12.0, CUDA 11.6 and torchvision 0.13.0. For a glimpse at the full documentation of DINO training please run:
```
python main_flsl.py --help
```
### Vanilla FLSL training
Run FLSL with ViT-small network on a single node with 8 GPUs for 300 epochs with the following command. Training time is 3.5 day
<summary>
Full command.
</summary>

To pretrain on ImageNet-1K, run:  

```bash
torchrun --standalone --nproc_per_node=gpu \
    ./main_flsl.py \
    --arch vit_small \
    --patch_size 16 \
    --out_dim 4096 \
    --output_dir ./output/ \
    --data_path /directory/to/imagenet-1k/train/ \
    --local_crops_number 2 \
    --local_crops_scale 0.05 0.4 \
    --global_crops_scale_t 0.8 1.0 \
    --global_crops_scale_s 0.5 1.0 \
    --random_pooling_window 2 \
    --norm_last_layer True \
    --batch_size_per_gpu 64 \
    --epochs 300 \
    --warmup_teacher_temp_epochs 30 \
    --warmup_teacher_temp 0.04 \
    --teacher_temp 0.07 \
    --teacher_centering False \
    --local_crops_size 96
```

Change the vit_small to vit_base for FLSL with ViT-base model.

## Pretrained weights on ImageNet
You can download the weights of the pretrained models on ImageNet.

| Dataset  | arch | checkpoint |
| ------------- | ------------- | ------------- |
| IN-1K | ViT-S/16  | <a href="https://drive.google.com/file/d/1vhzEHHNviL7X13OmGn_rEfb9-JXrNk9C/view?usp=sharing">download</a>|


## Evaluating object detection and instance segmentation on the COCO dataset

Step 1. Prepare COCO dataset  

The dataset can be downloaded at [`https://cocodataset.org/#download`](https://cocodataset.org/#download)  

Step 2. Install mmdetection  

```
git clone https://github.com/open-mmlab/mmdetection.git
```

Step 3. Fine-tune on the COCO dataset  

```
tools/dist_train.sh configs/selfpatch/mask_rcnn_vit_small_12_p16_1x_coco.py [number of gpu]\  
--work-dir /path/to/saving_dir\
--seed 0 --deterministic\
--options model.pretrained=/path/to/model_dir\  
```

## Evaluating semantic segmentation on the ADE20K dataset

Step 1. Prepare ADE20K dataset

The dataset can be downloaded at 
`http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl`

or following instruction of `https://github.com/CSAILVision/ADE20K`

Step 2. Install mmsegmentation

```
git clone https://github.com/open-mmlab/mmsegmentation.git
```

Step 3. Convert your model

```
python tools/model_converters/vit2mmseg.py /path/to/model_dir /path/to/saving_dir
```

Step 4. Fine-tune on the ADE20K dataset

```
tools/dist_train.sh configs/selfpatch/semfpn_vit-s16_512x512_40k_ade20k.py [number of gpu]\
--work-dir /path/to/saving_dir\
--seed 0 --deterministic\
--options model.pretrained=/path/to/model_dir
```
The optimization hyperarameters are adopted from <a href=https://github.com/facebookresearch/xcit>XCiT</a>.

## Evaluating video object segmentation on the DAVIS 2017 dataset
Step 1. Prepare DAVIS 2017 data

```
cd $HOME
git clone https://github.com/davisvideochallenge/davis-2017
cd davis-2017
./data/get_davis.sh
```

Step 2. Run Video object segmentation

```
python eval_video_segmentation.py\
--data_path /path/to/davis-2017/DAVIS/\
--output_dir /path/to/saving_dir\  --pretrained_weights /path/to/model_dir\
--arch vit_small\
--patch_size 16
```

## Citation
If you find this repository useful, please consider giving a star :star: and citation:
```
@inproceedings{
su2023flsl,
title={{FLSL}: Feature-level Self-supervised Learning},
author={Qing Su and Anton Netchaev and Hai Li and Shihao Ji},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=8pOBo5NgTQ}
}
```
