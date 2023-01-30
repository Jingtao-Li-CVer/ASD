from typing import List
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray
import seaborn as sns
from tqdm import tqdm
import logging 
import cv2


def save_heatmap(data, working_dir, save_width, save_height, dpi=300):
    save_path = os.path.join(working_dir, "heatmap.png")
    plt.figure(figsize=(save_width, save_height))
    sns.heatmap(data, cmap="jet", cbar=False) #热图
    plt.show()
    plt.axis('off')
    plt.savefig(save_path, dpi=dpi, bbox_inches = 'tight') 
    plt.close()

def savefig(
    epoch: int,
    imgs: List[NDArray],
    masks: List[NDArray],
    amaps: List[NDArray],
    working_dir: str,
    mean: NDArray, 
    std: NDArray, 
    maxValue: float
) -> None:

    save_dir = os.path.join(working_dir, "epochs-" + str(epoch))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (img, mask, amap) in enumerate(tqdm(zip(imgs, masks, amaps))):

        # How to get two subplots to share the same y-axis with a single colorbar
        # https://stackoverflow.com/a/38940369
        grid = ImageGrid(
            fig=plt.figure(figsize=(12, 4)),
            rect=111,
            nrows_ncols=(1, 3),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.15,
        )

        img = denormalize(img, mean, std, maxValue)

        grid[0].imshow(img,)
        grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[0].set_title("Input Image", fontsize=14)

        grid[1].imshow((mask), alpha=1.0, cmap="gray")
        grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[1].set_title("Annotation", fontsize=14)

        if amap.shape[0] == 1:
            im = grid[2].imshow(np.squeeze(amap, 0), alpha=1.0, cmap="jet")
        else:
            im = grid[2].imshow(amap, alpha=1.0, cmap="jet")
        grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[2].cax.colorbar(im)
        grid[2].cax.toggle_label(True)
        grid[2].set_title("Anomaly Map", fontsize=14) 

        plt.savefig(os.path.join(save_dir, str(i) + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

def denormalize(img: NDArray, mean: NDArray, std: NDArray, maxValue: float) -> NDArray:
    img = (img * std + mean) * maxValue
    return img.astype(np.uint8)

def label2rgb(label):
    """"""
    label2color_dict = {
        0: [255, 0, 0],
        1: [0, 0, 255],
        2: [0, 0, 0],
    }

    # visualize
    visual_anno = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for class_index in label2color_dict:
        visual_anno[label == class_index, :]=label2color_dict[class_index]
    return visual_anno

def savefig_argriculture_vision(
    epoch: int,
    imgs: List[NDArray],
    masks: List[NDArray],
    amaps: List[NDArray],
    working_dir: str,
    mean: NDArray, 
    std: NDArray, 
    maxValue: float
) -> None:

    save_dir = os.path.join(working_dir, "epochs-" + str(epoch))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (img, mask, amap) in enumerate(tqdm(zip(imgs, masks, amaps))):

        # How to get two subplots to share the same y-axis with a single colorbar
        # https://stackoverflow.com/a/38940369
        grid = ImageGrid(
            fig=plt.figure(figsize=(16, 4)),
            rect=111,
            nrows_ncols=(1, 4),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.15,
        )

        img = denormalize(img, mean, std, maxValue)

        grid[0].imshow(img[:,:,0:3])
        grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[0].set_title("RGB", fontsize=14)

        grid[1].imshow(img[:,:,3])
        grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[1].set_title("NIR", fontsize=14) 

        # grid[2].imshow(img)
        # grid[2].imshow(label2rgb(mask), alpha=0.3, cmap="Reds")
        grid[2].imshow((mask), alpha=1.0, cmap="gray")
        grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[2].set_title("Annotation", fontsize=14)

        # grid[3].imshow(img)
        im = grid[3].imshow(amap, alpha=1.0, cmap="jet")
        grid[3].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[3].cax.colorbar(im)
        grid[3].cax.toggle_label(True)
        grid[3].set_title("Anomaly Map", fontsize=14)

        plt.savefig(os.path.join(save_dir, str(i) + ".png"), bbox_inches="tight", dpi=300)
        plt.close()

def TwoPercentLinear(image, max_out=255, min_out=0):
    b, g, r = cv2.split(image)
    def gray_process(gray, maxout = max_out, minout = min_out):
        high_value = np.percentile(gray, 98)
        low_value = np.percentile(gray, 2)
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value) 
        processed_gray = ((truncated_gray - low_value)/(high_value - low_value)) * (maxout - minout)
        return processed_gray
    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = cv2.merge((b_p, g_p, r_p))
    return np.uint8(result)

def savefig_landslide_detection(
    epoch: int,
    imgs: List[NDArray],
    # reconsts: List[NDArray],
    masks: List[NDArray],
    amaps: List[NDArray],
    working_dir: str,
    mean: NDArray, 
    std: NDArray, 
    maxValue: float
) -> None:

    save_dir = os.path.join(working_dir, "epochs-" + str(epoch))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, (img, mask, amap) in enumerate(tqdm(zip(imgs, masks, amaps))):

        # How to get two subplots to share the same y-axis with a single colorbar
        # https://stackoverflow.com/a/38940369
        grid = ImageGrid(
            fig=plt.figure(figsize=(20, 4)),
            rect=111,
            nrows_ncols=(1, 5),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.15,
        )

        img = denormalize(img, mean, std, maxValue)
        # reconst = denormalize(reconst, mean, std, maxValue)

        rgb_image = img[:,:,[3,2,1]] 
        rgb_image = TwoPercentLinear(rgb_image)
        grid[0].imshow(rgb_image, )
        # logging.info(rgb_image) 
        grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[0].set_title("RGB", fontsize=14)

        grid[1].imshow(img[:,:,12])
        grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[1].set_title("Slope", fontsize=14) 

        # grid[2].imshow(img)
        # grid[2].imshow(label2rgb(mask), alpha=0.3, cmap="Reds")
        grid[2].imshow(img[:,:,13])
        grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[2].set_title("DEM", fontsize=14)

        grid[3].imshow((mask), alpha=1.0, cmap="gray")
        grid[3].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[3].set_title("Anotation", fontsize=14)

        # grid[3].imshow(img)
        im = grid[4].imshow(amap, alpha=1.0, cmap="jet")
        grid[4].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[4].cax.colorbar(im)
        grid[4].cax.toggle_label(True)
        grid[4].set_title("Anomaly Map", fontsize=14)

        plt.savefig(os.path.join(save_dir, str(i) + ".png"), bbox_inches="tight", dpi=300)
        plt.close()