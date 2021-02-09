# RSPNet
Official Pytorch implementation for AAAI2021 paper "[RSPNet: Relative Speed Perception for Unsupervised Video Representation Learning][RSPNET]"

![](https://github.com/PeihaoChen/RSPNet/blob/main/overview.png)

## Getting Started
### Install Dependencies

All dependencies can be installed using pip:

```sh
python -m pip install -r requirements.txt
```

Our experiments run on Python 3.7 and PyTorch 1.6. Other versions should work but are not tested.

### Transcode Videos (Optional)

This step is optional but will increase the data loading speed dramatically.

We decode the videos on the fly while training so we don't need to split frames. This makes disk IO a lot faster but increases CPU usage. This transcode step aims at reducing CPU consumed by decoding by 1) lower video resolution. 2) add more key frames.

To perform transcode, you need to have `ffmpeg` installed, then run:

```sh
python utils/transcode_dataset.py PATH/TO/ORIGIN_VIDEOS PATH/TO/TRANSCODED_VIDEOS
```

Be warned, this will use all your CPU and will take several hours (on our Intel E5-2630 *2 workstation) to complete.

### Prepare Datasets

Your are expected to prepare date for pre-training ([Kinetics-400](https://deepmind.com/research/open-source/kinetics) dataset) and fine-tuning ([UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads) and [Something-something-v2](https://20bn.com/datasets/something-something) datasets).
To let the scripts find datasets on your system, the recommended way is to create symbolic links in `./data` directory to the actual path. We found this solution flexible.

The expected directory hierarchy is as follow:

```
├── data
│   ├── hmdb51
│   │   ├── metafile
│   │   │   ├── brush_hair_test_split1.txt
│   │   │   └── ...
│   │   └── videos
│   │       ├── brush_hair
│   │       │   └── *.avi
│   │       └── ...
│   ├── UCF101
│   │   ├── ucfTrainTestlist
│   │   │   ├── classInd.txt
│   │   │   ├── testlist01.txt
│   │   │   ├── trainlist01.txt
│   │   │   └── ...
│   │   └── UCF-101
│   │       ├── ApplyEyeMakeup
│   │       │   └── *.avi
│   │       └── ...
│   ├── kinetics400
│   │   ├── train_video
│   │   │   ├── answering_questions
│   │   │   │   └── *.mp4
│   │   │   └── ...
│   │   └── val_video
│   │       └── (same as train_video)
│   ├── kinetics100
│   │   └── (same as kinetics400)
│   └── smth-smth-v2
│       ├── 20bn-something-something-v2
│       │   └── *.mp4
│       └── annotations
│           ├── something-something-v2-labels.json
│           ├── something-something-v2-test.json
│           ├── something-something-v2-train.json
│           └── something-something-v2-validation.json
└── ...
```

Alternatively, you can change the path in `config/dataset` to match your system.

### Build Kinetics-100 dataset (Optional)

Some of our ablation study experiments use the Kinetics-100 dataset for pre-training. This dataset is built by extract 100 classes from Kinetics-400, which has the smallest file size on the train set.

If you have Kinetics-400 available, you can build Kinetics-100 by:

```sh
python -m utils.build_kinetics_subset
```

This script will create symbolic links instead of copy data. It is expected to complete in a minute.

We have included a pre-built one at `data/kinetics100_links` and created the symbolic link `data/kinetics100` that related to it. You need to have `data/kinetics400` available at runtime.

## Pre-training on Pretext Tasks

Now you have set up the environment. Run the following command to pre-train your models on pretext tasks.

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Architecture: C3D
python pretrain.py -e exps/pretext-c3d -c config/pretrain/c3d.jsonnet
# Architecture: ResNet-18
python pretrain.py -e exps/pretext-resnet18 -c config/pretrain/resnet18.jsonnet
# Architecture: S3D-G
python pretrain.py -e exps/pretext-s3dg -c config/pretrain/s3dg.jsonnet
# Architecture: R(2+1)D
python pretrain.py -e exps/pretext-r2plus1d -c config/pretrain/r2plus1d.jsonnet
```
You can use kinetics100 dataset for training by [editing](config/pretrain/moco-train-base.jsonnet) `config/pretrain/moco-train-base.jsonnet` (line 13)
<!-- 
```json
dataset: kinetics400, // or kinetics100
``` -->

## Action Recognition

After pre-trained on pretext tasks, these models are fine-tuned to perform action recognition task on UCF101, HMDB51 and Something-something-v2 datasets.

```sh
export CUDA_VISIBLE_DEVICES=0,1
# Dataset: UCF101
#     Architecture: C3D ACC@1=76.71%
python finetune.py -c config/finetune/ucf101_c3d.jsonnet \
                   --mc exps/pretext-c3d/model_best.pth.tar \
                   -e exps/ucf101-c3d
#     Architecture: ResNet-18 ACC@1=74.33%
python finetune.py -c config/finetune/ucf101_resnet18.jsonnet \
                   --mc exps/pretext-resnet18/model_best.pth.tar \
                   -e exps/ucf101-resnet18
#     Architecture: S3D-G ACC@1=89.9%
python finetune.py -c config/finetune/ucf101_s3dg.jsonnet \
                   --mc exps/pretext-s3dg/model_best.pth.tar \
                   -e exps/ucf101-s3dg
#     Architecture: R(2+1)D ACC@1=81.1%
python finetune.py -c config/finetune/ucf101_r2plus1d.jsonnet \
                   --mc exps/pretext-r2plus1d/model_best.pth.tar \
                   -e exps/ucf101-r2plus1d

# Dataset: HMDB51
#     Architecture: C3D ACC@1=44.58%
python finetune.py -c config/finetune/hmdb51_c3d.jsonnet \
                   --mc exps/pretext-c3d/model_best.pth.tar \
                   -e exps/hmdb51-c3d
#     Architecture: ResNet-18 ACC@1=41.83%
python finetune.py -c config/finetune/hmdb51_resnet18.jsonnet \
                   --mc exps/pretext-resnet18/model_best.pth.tar \
                   -e exps/hmdb51-resnet18
#     Architecture: S3D-G ACC@1=59.6%
python finetune.py -c config/finetune/hmdb51_s3dg.jsonnet \
                   --mc exps/pretext-s3dg/model_best.pth.tar \
                   -e exps/hmdb51-s3dg
#     Architecture: R(2+1)D ACC@1=44.6%
python finetune.py -c config/finetune/hmdb51_r2plus1d.jsonnet \
                   --mc exps/pretext-r2plus1d/model_best.pth.tar \
                   -e exps/hmdb51-r2plus1d

# Dataset: Something-something-v2
#     Architecture: C3D ACC@1=47.76%
python finetune.py -c config/finetune/smth_smth_c3d.jsonnet \
                   --mc exps/pretext-c3d/model_best.pth.tar \
                   -e exps/smthv2-c3d
#     Architecture: ResNet-18 ACC@1=44.02%
python finetune.py -c config/finetune/smth_smth_resnet18.jsonnet \
                   --mc exps/pretext-resnet18/model_best.pth.tar \
                   -e exps/smthv2-resnet18
#     Architecture: S3D-G ACC@1=55.03%
python finetune.py -c config/finetune/smth_smth_s3dg.jsonnet \
                   --mc exps/pretext-s3dg/model_best.pth.tar \
                   -e exps/smthv2-s3dg
```

## Results and Pre-trained Models

| Architecture | Pre-trained dataset | Pre-training epoch | Pre-trained model | Acc. on UCF101 | Acc. on HMDB51 |
|:------------:|:-------------------:|:------------------:|:-----------------:|:--------------:|:--------------:|
|     S3D-G    |     Kinetics-400    |        1000        |   [Download link](https://github.com/PeihaoChen/RSPNet/releases/download/pretrained_model/model_best_s3dg_1000epoch.pth.tar)   |      93.7      |      64.7      |
|     S3D-G    |     Kinetics-400    |         200        |   [Download link](https://github.com/PeihaoChen/RSPNet/releases/download/pretrained_model/model_best_s3dg_200epoch.pth.tar)   |      89.9      |      59.6      |
|    R(2+1)D   |     Kinetics-400    |         200        |   [Download link](https://github.com/PeihaoChen/RSPNet/releases/download/pretrained_model/model_best_r21d_200epoch.pth.tar)   |      81.1      |      44.6      |
|   ResNet-18  |     Kinetics-400    |         200        |   [Download link](https://github.com/PeihaoChen/RSPNet/releases/download/pretrained_model/model_best_resnet18_200epoch.pth.tar)   |      74.3      |      41.8      |
|      C3D     |     Kinetics-400    |         200        |   [Download link](https://github.com/PeihaoChen/RSPNet/releases/download/pretrained_model/model_best_c3d_200epoch.pth.tar)   |      76.7      |      44.6      |


## Troubleshoot

* DECORDError cannot find video stream with wanted index: -1

  Some video from Kinetics dataset does not contain a valid video stream for some unknown reason. To filter them out, run `python utils/verify_video.py PATH/TO/VIDEOS`, then copy the output to the `blacklist` config in `config/dataset/kinetics{400,100}.libsonnet`. You need to have `ffmpeg` installed.


## Citation


Please cite the following paper if you feel RSPNet useful to your research
```
@InProceedings{chen2020RSPNet,
author = {Peihao Chen, Deng Huang, Dongliang He, Xiang Long, Runhao Zeng, Shilei Wen, Mingkui Tan, and Chuang Gan},
title = {RSPNet: Relative Speed Perception for Unsupervised Video Representation Learning},
booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
year = {2021}
}
```

[RSPNET]:https://arxiv.org/abs/2011.07949

## Contact
For any question, please file an issue or contact
```
Peihao Chen: phchencs@gmail.com
Deng Huang: im.huangdeng@gmail.com
```
