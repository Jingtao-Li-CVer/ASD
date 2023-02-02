## Anomaly Segmentation for High-Resolution Remote Sensing Images Based on Pixel Descriptors (AAAI2023)

<p align="center">
  <img src=./figs/ASD.jpg width="600"> 
</p>

This is a PyTorch implementation of the [AAAI2023 paper](http://arxiv.org/abs/2301.13422) (to be published):
```
@Article{Li2022ASD,
  author = {Jingtao Li and Xinyu Wang and Hengwei Zhao and Shaoyu Wang and Yanfei Zhong},
  title = {Anomaly Segmentation for High-Resolution Remote Sensing Images Based on Pixel Descriptors},
  journal = {arXiv preprint arXiv:2301.13422},
  year = {2023},
}
```

### Introduction

A novel anomaly segmentation model based on pixel descriptors (ASD) is implemented to segment anomaly patterns of the earth deviating from normal patterns, which plays an important role in various Earth vision applications. The ASD model incorporates the data argument for generating virtual abnormal samples, which can force the pixel descriptors to be compact for normal data and meanwhile to be diverse to avoid the model collapse problems when only positive samples participated in the training. In addition, the ASD introduced a multi-level and multi-scale feature extraction strategy for learning the low-level and semantic information to make the pixel descriptors feature-rich. The three conditions (compact, diverse, and feature-rich) direct the design of architecture and optimization.



### Preparation

1. Install required packages according to the requirements.txt.
2. Download the datasets (i.e. Agriculture-vison, DeepGlobe, Landslide4sense and FAS) with the following link.
    (https://pan.baidu.com/s/1lY5RfPOq_KIxvWJ4F8c0GA   password:171j)


### Model Training and Testing (without visualization)

1. Each normal class is trained separately.
2. The first 10 epochs are trained only to initialize the hypersphere center without testing the model.
3. Starting the training and testing process using the following command.
```
python run.py 'config_fie_path'
```
For example, to train the ASD model when treating the drydown in Agriculture-vison dataset as the normal class
```
python run.py ./configs/asd_drydown_config.yaml
```

### Result visualization

1. Write the trained parameter path in the config file.
```
ckpt_dir: 'ckpt_path'
```
2. To visualize the results, set the parameter (i.e. visualization) to be True in _test function.
```
self._test(epoch, visualization=True)
```

