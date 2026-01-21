# CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models

This is the official PyTorch implementation of LiteLVLM (ICML Under Review).

## Outline

1. [LiteLVLM](#LiteLVLM)
2. [Installation](#Installation)
3. [Preparation](#Preparation)
3. [Model Zoo](#Model-Zoo)
4. [Evaluation](#Evaluation)
5. [Acknowledgement](#Acknowledgement)

## LiteLVLM


## Installation

1. Clone this repository.
```bash
https://github.com/2026ICML4414/CLIP-Tricks-You.git
cd CLIP-Tricks-You
```

2. Setup a conda environment and install packages.
```bash
conda create -n LiteLVLM python=3.10 -y
conda activate LiteLVLM
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

3. Install mmcv
```bash
git clone https://github.com/open-mmlab/mmcv
cd mmcv
git checkout v1.4.7
MMCV_WITH_OPS=1 pip install -e .
```

## Datasets

Please see [`docs/datasets.md`](docs/datasets.md) for dataset preparation guidelines.

## Model Zoo

See [`MODEL_ZOO.md`](./MODEL_ZOO.md) for our pretrained models.

## Training

Our MS-DePro is built with [Detectron2](https://github.com/facebookresearch/detectron2). See [Getting Started with Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) to learn about basic usage. We provide an example below for training our object detector on MSDA and MSDG settings.

<details>

<summary>
1. Prepare the pretrained RegionCLIP model and set up the dataset.
</summary>
  
- Check [`RegionCLIP`](https://github.com/microsoft/RegionCLIP/blob/main/docs/MODEL_ZOO.md) to 
  - download the pretrained RegionCLIP checkpoint `regionclip_pretrained-cc_rn50.pth` to the folder `./pretrained`, 
  - (optional) download the trained RPN checkpoint `rpn_coco_{48,65,80}.pth` to the folder `./pretrained`.
- Check [`datasets/README.md`](datasets/README.md) to set up dataset.

</details>

<details>

<summary>
2. After preparation, run the following script to train an object detector.
</summary>

```
#!/bin/bash

CONFIG=$1
NUM_GPUS=$2
CUDA_DEVICES=$3
NUM_THREADS=$4
NUM_SRCS=$5
IMG_BATCH=$6
OUTPUT_DIR=$7

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$NUM_THREADS \
    python tools/train_net.py \
    --num-gpus $NUM_GPUS \
    --config $CONFIG \
    MODEL.BACKBONE_WEIGHTS pretrained/regionclip_pretrained-cc_rn50.pth \
    MODEL.RESNETS.OUT_FEATURES "(('res2'), ('res4'))" \
    DATASETS.NUM_SOURCES $NUM_SRCS \
    SOLVER.IMG_PER_BATCH_LABEL $IMG_BATCH SOLVER.IMG_PER_BATCH_UNLABEL $IMG_BATCH \
    OUTPUT_DIR $OUTPUT_DIR
```

For example, to run the `Cross-time` experiment using 4 GPUs, execute the following command:
```
sh dist_train.sh configs/MSDA/cross_time.sh 4 0,1,2,3 8 2 8 output/cross_time
```

</details>

## Inference

We provide an example below for evaluating our object detector on MSDA and MSDG settings.

<details>

<summary>
1. Prepare the trained detector and set up the dataset.
</summary>
  
- Check [`MODEL_ZOO.md`](MODEL_ZOO.md) to 
  - download the trained detector checkpoints to the folder `./output/`.
- Check [`datasets/README.md`](datasets/README.md) to set up dataset.

</details>

<details>

<summary>
2. After preparation, run the following script to evaluate an object detector.
</summary>
  
```
#!/bin/bash

CONFIG=$1
NUM_GPUS=$2
CUDA_DEVICES=$3
NUM_THREADS=$4
NUM_SRCS=$5
WEIGHTS=$6
OUTPUT_DIR=$7

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES OMP_NUM_THREADS=$NUM_THREADS \
    python tools/train_net.py \
    --eval-only \
    --num-gpus $NUM_GPUS \
    --config $CONFIG \
    MODEL.BACKBONE_WEIGHTS pretrained/regionclip_pretrained-cc_rn50.pth \
    MODEL.WEIGHTS $WEIGHTS \
    MODEL.RESNETS.OUT_FEATURES "(('res2'), ('res4'))" \
    DATASETS.NUM_SOURCES $NUM_SRCS \
    OUTPUT_DIR $OUTPUT_DIR
```

For example, to evaluate the `Cross-time` experiment using a single GPU, execute the following command:
```
sh slurm_test.sh configs/MSDA/cross_time.yaml 1 0 1 2 output/cross_time.pth eval/cross_time
```

</details>

## Acknowledgement
We thank to [GLaMM](https://github.com/mbzuai-oryx/groundingLMM) and [VideoGLaMM](https://github.com/mbzuai-oryx/VideoGLaMM) for releasing their code as open source.
