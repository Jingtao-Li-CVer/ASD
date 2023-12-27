## Anomaly Segmentation for High-Resolution Remote Sensing Images Based on Pixel Descriptors (AAAI2023)

<p align="center">
  <img src=./figs/ASD.jpg width="650"> 
</p>

This is a PyTorch implementation of the [AAAI2023 paper](https://ojs.aaai.org/index.php/AAAI/article/view/25563/25335):
```
@Article{Li2022ASD,
  author = {Jingtao Li and Xinyu Wang and Hengwei Zhao and Shaoyu Wang and Yanfei Zhong},
  title = {Anomaly Segmentation for High-Resolution Remote Sensing Images Based on Pixel Descriptors},
  journal = {arXiv preprint arXiv:2301.13422},
  year = {2023},
}
```

### Outline
1. Anomaly segmentation for HRS images is a new task in remote sensing community, which is of great significance for environmental monitoring application.
2. Proposed model ASD sets the first baseline, which aims to learn better normal descriptors.
3. FAS dataset is made and publicly available.


### Introduction

A novel anomaly segmentation model based on pixel descriptors (ASD) is implemented to segment anomaly patterns of the earth deviating from normal patterns, which plays an important role in various Earth vision applications. The ASD model incorporates the data argument for generating virtual abnormal samples, which can force the pixel descriptors to be compact for normal data and meanwhile to be diverse to avoid the model collapse problems when only positive samples participated in the training. In addition, the ASD introduced a multi-level and multi-scale feature extraction strategy for learning the low-level and semantic information to make the pixel descriptors feature-rich. The three conditions (compact, diverse, and feature-rich) direct the design of architecture and optimization.

<p align="center">
  <img src=./figs/sample.jpg width="600"> 
</p>

### Preparation

1. Install required packages according to the requirements.txt.
2. Download the datasets (i.e. Agriculture-vison, DeepGlobe, Landslide4sense and FAS) with the following link.
    (https://pan.baidu.com/s/1lY5RfPOq_KIxvWJ4F8c0GA   password:171j) 
   
   Notice: The Agriculture-vison dataset can also be downloaded from this [link](https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021).


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

### Trained checkpoints

1. Agriculture-vison dataset

|Normal class | BaiduDrive | Normal class | BaiduDrive | Normal class | BaiduDrive|
| --- | --- |  --- |  --- |  --- |  --- |
| Drydown |  [Link](https://pan.baidu.com/s/1zRkr8WpXNQPBqYA8-GBLtQ?pwd=CVer) | Double plant |  [Link](https://pan.baidu.com/s/1nogMZYCt-0XYgioB4-GRQQ?pwd=CVer) | Endrow | [Link](https://pan.baidu.com/s/1Imszb20McNHazjSZh6K9_w?pwd=CVer) |
| Weed cluster |  [Link](https://pan.baidu.com/s/1mSSJeUGXO-iylSz7RorGfA?pwd=CVer) | ND |  [Link](https://pan.baidu.com/s/1IurLCEIlHs8vdDOLqIdsWA?pwd=CVer) | Water | [Link](https://pan.baidu.com/s/1Igk2unnSJLJQu06-LgBlww?pwd=CVer) |

2. DeepGlobe dataset

|Normal class | BaiduDrive | Normal class | BaiduDrive | Normal class | BaiduDrive|
| --- | --- |  --- |  --- |  --- |  --- |
| Urban land |  [Link](https://pan.baidu.com/s/1YYCI2S0zha05sm5xbUfA7Q?pwd=CVer) | Agriculture |  [Link](https://pan.baidu.com/s/1pi7bfNWPRGWCHTrqnqhQSQ?pwd=CVer) | Range land |  [Link](https://pan.baidu.com/s/15MiDsFtby5acbQAXF9LaOg?pwd=CVer) |
| Forest land |  [Link](https://pan.baidu.com/s/1uZka5h77jwp48oe5mriwdA?pwd=CVer) | Water |  [Link](https://pan.baidu.com/s/1Hb9n1fHtQnTShPm9PDD5tQ?pwd=CVer) | Barren land |  [Link](https://pan.baidu.com/s/1DEUSPpjVB2ChUu5iC68ijg?pwd=CVer) |

3. FAS and Landslide4sense datasets

|Dataset | BaiduDrive | Dataset | BaiduDrive |
| --- | --- |  --- |  --- |
| FAS |  [Link](https://pan.baidu.com/s/1OsUFQoyeG2Ks-uzUMH7jMQ?pwd=CVer) | Landslide4sense |  [Link](https://pan.baidu.com/s/1EJbPM1T9PPUOPsgqJjB0gQ?pwd=CVer) |

