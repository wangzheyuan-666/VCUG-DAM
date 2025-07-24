"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile

from tqdm import tqdm


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    mean=torch.tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1).to('cuda')
    std=torch.tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1).to('cuda')
    out = x * std + mean
    # out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_image(x, ncol, filename):
    x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)


def z_code(delta_age, delta_pos, pos_dim=36,gap_dim=220, l=-5, u=5, device='cpu', noise=0):    #img_a: image source
    # print(img_a.shape, img_a_age.shape)
    bs = delta_age.shape[0]  #batch size
    
    pos_tensor = torch.ones(bs, pos_dim).to(device) * (delta_pos.unsqueeze(dim=1))

    age_tensor = torch.zeros([bs,gap_dim]).to(device)
    # 生成对应的下标范围
    # print(img_b_age-img_a_age)
    gap = int(gap_dim/(u-l+1))
    start_indices_tensor = torch.clip((delta_age).to(torch.float32), min=l, max=u) * gap + (-1 * gap * l)
    start_indices_tensor = start_indices_tensor.to(torch.int32)

    index_ranges = start_indices_tensor.unsqueeze(1) + torch.arange(gap).to(device)
    # 使用index_ranges来修改相应的元素
    age_tensor[torch.arange(age_tensor.size(0)).unsqueeze(1), index_ranges] += 1.0
    z = torch.cat([pos_tensor, age_tensor], dim=1).to(torch.float32)

    z += noise * torch.randn([bs,gap_dim+pos_dim]).to(device)
    return z


@torch.no_grad()
def debug_image(model, args, img_a, img_b, delta_age, delta_pos, step, epoch=1):
    delta_age_zero = torch.zeros_like(delta_age).to(img_a.device, non_blocking=True)
    delta_pos_zero = torch.zeros_like(delta_pos).to(img_a.device, non_blocking=True) 

    with torch.no_grad():       
        z_a2b = z_code(delta_age, delta_pos, pos_dim=36,gap_dim=220, l=-5, u=5, device=img_a.device, noise=0.0000)
        _, pred_a2b, _, _ = model(img_a, mask_ratio=0.15, s=z_a2b)


        z_s2s = z_code(delta_age_zero, delta_pos_zero, pos_dim=36,gap_dim=220, l=-5, u=5, device=img_a.device, noise=0.0000)
        _, pred_a2a, _, _ = model(img_a, mask_ratio=0.15, s=z_s2s)

        _, pred_b2b, _, _ = model(img_b, mask_ratio=0.15, s=z_s2s)
           
        x_concat = [img_a, unpatchify(pred_a2b), unpatchify(pred_a2a), unpatchify(pred_b2b), img_b]
        x_concat = torch.cat(x_concat, dim=0)
        filename = ospj(args.sample_dir, '%06d_sample.jpg' % (step))
        save_image(x_concat, img_a.size(0), filename)
        del x_concat

def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = 16
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

