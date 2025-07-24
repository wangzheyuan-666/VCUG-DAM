import torch
import numpy as np
import torch.distributed as dist

import os
import pickle as pkl
from pathlib import Path
import tempfile
import shutil
from util import mean_iou

"""
ImageNet classifcation accuracy
"""


def accuracy(output, target, topk=(1,)):
    """
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k /= batch_size
            res.append(correct_k)
        return res


"""
Segmentation mean IoU
based on collect_results_cpu
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/apis/test.py#L160-L200
"""


def compute_metrics(
    seg_pred,
    seg_gt,
    n_cls,
    ignore_index=None,
    ret_cat_iou=False,
    tmp_dir=None,
    distributed=False,
):
    ret_metrics_mean = torch.zeros(3, dtype=float, device=seg_pred.device)
    list_seg_pred = []
    list_seg_gt = []
    keys = sorted(seg_pred.keys())
    for k in keys:
        list_seg_pred.append(np.asarray(seg_pred[k]))
        list_seg_gt.append(np.asarray(seg_gt[k]))
    ret_metrics = mean_iou(
        results=list_seg_pred,
        gt_seg_maps=list_seg_gt,
        num_classes=n_cls,
        ignore_index=ignore_index,
    )
    ret_metrics = [ret_metrics["aAcc"], ret_metrics["Acc"], ret_metrics["IoU"]]
    ret_metrics_mean = torch.tensor(
        [
            np.round(np.nanmean(ret_metric.astype(np.float64)) * 100, 2)
            for ret_metric in ret_metrics
        ],
        dtype=float,
        device=seg_pred.device,
    )
    cat_iou = ret_metrics[2]
    # broadcast metrics from 0 to all nodes
    if distributed:
        dist.broadcast(ret_metrics_mean, 0)
    pix_acc, mean_acc, miou = ret_metrics_mean
    ret = dict(pixel_accuracy=pix_acc, mean_accuracy=mean_acc, mean_iou=miou)
    return ret
