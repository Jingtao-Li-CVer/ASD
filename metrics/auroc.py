from gc import get_threshold
import logging
import matplotlib.pyplot as plt
from numpy import ndarray as NDArray
from sklearn.metrics import roc_auc_score, roc_curve
import os
import random
import numpy as np
from scipy import integrate
from tqdm import tqdm
import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import sys
sys.path.append("/home/ljt21/ad/RSAD/metrics/")
from iou_metric import SegEvaluator


def compute_auroc(epoch: int, ep_amaps, ep_gt, working_dir: str, image_level=False, save_image=False, compute_iou=True) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)

    Args:
        epoch (int): Current epoch
        ep_amaps (NDArray): Anomaly maps in a current epoch
        ep_gt (NDArray): Ground truth masks in a current epoch

    Returns:
        float: AUROC score
    """
    save_dir = os.path.join(working_dir, "epochs-" + str(epoch))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    y_score, y_true = [], []
    for i, (amap, gt) in enumerate(tqdm(zip(ep_amaps, ep_gt))):
        anomaly_scores = amap[np.where(gt == 0)]
        normal_scores = amap[np.where(gt == 1)]
        y_score += anomaly_scores.tolist()
        y_true += np.zeros(len(anomaly_scores)).tolist()
        y_score += normal_scores.tolist()
        y_true += np.ones(len(normal_scores)).tolist() 
        
    scoreDF = roc_auc_score(y_true, y_score)
    logging.info("scoreDF: " + str(scoreDF))


    if compute_iou:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
        threshold = thresholds[maxindex]
        evaluator = SegEvaluator(2)
        evaluator.reset()
        for i, (amap, gt) in enumerate(tqdm(zip(ep_amaps, ep_gt))):
            amap = np.where(amap > threshold, 1, 0)
            amap = amap.astype(np.int8)
            evaluator.add_batch(gt, amap)
        Iou = evaluator.mean_iou()
        logging.info("Iou: " + str(Iou))


    if save_image:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        plt.plot(fpr, tpr, marker="o", color="k", label=f"AUROC Score: {round(scoreDF, 3)}")
        plt.xlabel("FPR: FP / (TN + FP)", fontsize=14)
        plt.ylabel("TPR: TP / (TP + FN)", fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,"roc_curve.png"))
        plt.close()

    return scoreDF
