# Towards Few-shot Object Detection through Dual Calibration

The official PyTorch implementation for our work: [Towards Few-shot Object Detection through Dual Calibration](##)

### Abstract
> Object detection is crucial in traffic scenes for accurately identifying multiple objects within complex environments. Traditional systems rely on deep learning models trained on large-scale datasets, but this approach can be expensive and impractical. Few-shot object detection (FSOD) offers a potential solution by addressing limited data availability. However, object detectors trained with FSOD frameworks often generalize poorly on classes with limited samples. Although most existing methods alleviate this problem by calibrating either the feature maps or prediction heads of the object detector, none of them, like this work, have proposed a unified, dual calibration strategy that operates in both the latent feature space and the prediction probability space of the object detector. Specifically, we propose to improve representation precision by reducing the variances of feature vectors using highly adaptive centroids learned from ensembles of training features in the latent space. These centroids are employed to calibrate the features and reveal the underlying structure of the latent feature space. Moreover, we further exploit the association between the query and support features to calibrate inaccurate predictions resulting from overfitting or underfitting when fine-tuned with few training samples and low training iterations. Through visualization, we demonstrate that our method produces more discriminative high-level features, ultimately improving the precision of an object detector's predictions. To validate the effectiveness of our approaches, we conduct comprehensive experiments on well-known benchmarks, including PASCAL VOC and MS-COCO, showing considerable performance gains compared to existing works.

![Overview](assets/overview.png)

## Updates
- (Aug 2024) `README.md` updated!
- (Aug 2023) Initial code released!

## Installation
**Requirements**
- Python 3.8
- [PyTorch](https://pytorch.org/) 1.10.0
- [detectron2](https://github.com/facebookresearch/detectron2/tree/v0.6) 0.6
- CUDA 11.1

## Data Preparation
The pretrained models used in this work can be downloaded from [PyTorch ResNet-101](https://download.pytorch.org/models/resnet101-cd907fc2.pth) [[link](https://download.pytorch.org/models/resnet101-cd907fc2.pth)]. The Detectron2 [script](https://raw.githubusercontent.com/facebookresearch/detectron2/main/tools/convert-torchvision-to-d2.py) is available for converting the PyTorch checkpoint into a Detectron2 backbone that is compatible. Please download the pretrained model to the directory `pretrain/`, and run the conversion script.

We evaluate our models on two datasets:
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- [MS\-COCO](https://cocodataset.org/#home)

We follow the data preparation procedures specified by [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md).

The expected directory structure:
```
assets/
configs/
dataset/
    coco/
    cocosplit/
    VOC2007/
    VOC2012/
    vocsplit/
pretrain/
    R-101.pkl
src/
tools/
main.py
README.md
hashmap.sh
run_coco.sh
run_voc.sh
```

## Training and Evaluation
```bash
# PASCAL VOC experiments
sh run_voc.sh EXP_ID
# MS-COCO experiments
sh run_coco.sh EXP_ID
```

## Acknowledgement
This repository is developed based on [Detectron2](https://github.com/facebookresearch/detectron2).

