# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import numpy as np

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from util.seg_metrics import compute_metrics
from util.mean_iou import ConfusionMatrix

import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from util.loss import DiceLoss, get_compactness_cost, FocalLoss,FocalLoss_Binary

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score, roc_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import auc
import matplotlib as mpl

# from util
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, masks, grade_labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        # print(grade_labels)
        target_bladder = grade_labels[:,0].to(device, non_blocking=True)
        target_urethra = grade_labels[:,1].to(device, non_blocking=True)
        target_l_vur = grade_labels[:,2].to(device, non_blocking=True)
        target_r_vur = grade_labels[:,3].to(device, non_blocking=True)
 
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            
            loss_seg = criterion(outputs[0], masks)

            output_bladder, output_urethra, output_l_vur, output_r_vur = torch.split(outputs[1], [2,2,6,6], dim=-1)    
                  
            loss_cls = criterion(output_bladder, target_bladder) + \
                   criterion(output_urethra, target_urethra) + \
                   criterion(output_l_vur, target_l_vur) + \
                   criterion(output_r_vur, target_r_vur)
            loss = loss_cls + loss_seg
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    bladder_metrics = AverageMeter(name='bladder', num_classes=2)
    urethra_metrics = AverageMeter(name='urethra', num_classes=2)
    l_vur_metrics = AverageMeter(name='l_vur', num_classes=6)
    r_vur_metrics = AverageMeter(name='r_vur', num_classes=6)

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        targets = batch[1]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_bladder = batch[-1][:,0].to(device, non_blocking=True)
        target_urethra = batch[-1][:,1].to(device, non_blocking=True)
        target_l_vur = batch[-1][:,2].to(device, non_blocking=True)
        target_r_vur = batch[-1][:,3].to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss_seg = criterion(outputs[0], targets)
            output_bladder, output_urethra, output_l_vur, output_r_vur = torch.split(outputs[1], [2,2,6,6], dim=-1)
            loss_cls = criterion(output_bladder, target_bladder) + \
                   criterion(output_urethra, target_urethra) + \
                   criterion(output_l_vur, target_l_vur) + \
                   criterion(output_r_vur, target_r_vur)
            loss = loss_cls + loss_seg

        bladder_metrics.update(output_bladder, target_bladder)
        urethra_metrics.update(output_urethra, target_urethra)
        l_vur_metrics.update(output_l_vur, target_l_vur)
        r_vur_metrics.update(output_r_vur, target_r_vur)


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

    bladder_results = bladder_metrics.compute()
    urethra_results = urethra_metrics.compute()
    l_vur_results = l_vur_metrics.compute()
    r_vur_results = r_vur_metrics.compute()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* bladder_Acc@1_auc {:.3f}_{:.3f}  urethra_Acc@1_auc {:.3f}_{:.3f}  l_vur_Acc@1_auc {:.3f}_{:.3f}  r_vur_Acc@1_auc {:.3f}_{:.3f} loss {:.3f}'
          .format(
            bladder_results['Accuracy'],bladder_results['AUC'],
            urethra_results['Accuracy'],urethra_results['AUC'],
            l_vur_results['Accuracy'],l_vur_results['AUC'],
            r_vur_results['Accuracy'],r_vur_results['AUC'],
            metric_logger.loss.global_avg))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, num_classes=2, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.labels = []
        self.preds = []
        self.one_hot_labels = []

    def update(self, pred, label):
        
        self.labels.append(label.cpu())
        self.preds.append(pred.cpu())
        one_hot_label=F.one_hot(label.to(torch.int64), num_classes=self.num_classes)
        self.one_hot_labels.append(one_hot_label)

    def compute(self):
        labels = torch.cat(self.labels, dim = 0).cpu().numpy()
        preds = torch.cat(self.preds, dim = 0).cpu().numpy()
        one_hot_labels = torch.cat(self.one_hot_labels, dim = 0).cpu().numpy()
        result = self.compute_metric(datanpGT=labels, datanpPRED=preds, target_names = [str(i) for i in range(self.num_classes)])
        return result


    def specificity(self, y_true: np.array, y_pred: np.array, classes: set = None):
        if classes is None:
            classes = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))
        specs = []
        for cls in classes:
            y_true_cls = np.array((y_true == cls), np.int32)
            y_pred_cls = np.array((y_pred == cls), np.int32)
            specs.append(recall_score(y_true_cls, y_pred_cls, pos_label=0))
        return specs

    def compute_metric(self, datanpGT, datanpPRED, target_names):
        n_class = len(target_names)
        argmaxPRED = np.argmax(datanpPRED, axis=1)
        F1_metric = np.zeros([n_class, 1])
        tn = np.zeros([n_class, 1])
        fp = np.zeros([n_class, 1])
        fn = np.zeros([n_class, 1])
        tp = np.zeros([n_class, 1])

        Accuracy_score = accuracy_score(datanpGT, argmaxPRED)
        ROC_curve = {}
        mAUC = 0

        for i in range(n_class):
            tmp_label = datanpGT == i
            tmp_pred = argmaxPRED == i
            F1_metric[i] = f1_score(tmp_label, tmp_pred)
            tn[i], fp[i], fn[i], tp[i] = confusion_matrix(tmp_label, tmp_pred).ravel()
            outAUROC = roc_auc_score(tmp_label, datanpPRED[:, i])

            mAUC = mAUC + outAUROC
            [roc_fpr, roc_tpr, roc_thresholds] = roc_curve(tmp_label, datanpPRED[:, i])

            ROC_curve[i] = {'ROC_fpr': roc_fpr,
                            'ROC_tpr': roc_tpr,
                            'ROC_T': roc_thresholds,
                            'AUC': outAUROC}

        mPrecision = sum(tp) / sum(tp + fp)
        mRecall = sum(tp) / sum(tp + fn)

        output = {
            'class_name': target_names,
            'F1': F1_metric,
            'AUC': mAUC / n_class,
            'Accuracy': Accuracy_score,

            'Sensitivity': tp / (tp + fn),
            'Precision': tp / (tp + fp),
            'Specificity': tn / (fp + tn),
            'ROC_curve': ROC_curve,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,

            'micro-Precision': mPrecision,
            'micro-Sensitivity': mRecall,
            'micro-Specificity': sum(tn) / sum(fp + tn),
            'micro-F1': 2*mPrecision * mRecall / (mPrecision + mRecall),
        }
        return output

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
