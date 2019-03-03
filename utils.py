'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import copy
import numpy as np

import torch.nn as nn
import torch.nn.init as init

datasets = { 'CIFAR10': {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010), 'num_classes': 10, 'image_channels': 3, 'size': 32}, # std is wrong. should be [0.2471, 0.2435, 0.2616]
            'CIFAR100': {'mean': (0.5071, 0.4865, 0.4409), 'std': (0.2673, 0.2564, 0.2762), 'num_classes': 100, 'image_channels': 3, 'size': 32},
            'SVHN': {'mean': (0.4377, 0.4438, 0.4728), 'std': (0.1980, 0.2010, 0.1970), 'num_classes': 10, 'image_channels': 3, 'size': 32},

            'FashionMNIST': {'mean': (0.0,), 'std': (1.0,), 'num_classes': 10, 'image_channels': 1, 'size': 28},
            'MNIST': {'mean': (0.0,), 'std': (1.0,), 'num_classes': 10, 'image_channels': 1, 'size': 28},
}

class MovingMaximum(object):
    def __init__(self):
        self.data = [] # data[i] is the maximum val in data[0:i+1]
        self.max = 0.0

    def push(self, current_data):
        if len(self.data) == 0:
            self.max = current_data
        elif current_data > self.max:
            self.max = current_data
        self.data.append(self.max)

    def get(self):
        return self.data

    def delta(self, start, end):
        try:
            res = self.data[end] - self.data[start]
        except IndexError:
            res = self.data[end]
        return res

class ExponentialMovingAverage(object):
    def __init__(self, decay=0.95):
        self.data = []
        self.decay = decay
        self.avg_val = 0.0

    def push(self, current_data):
        self.avg_val = self.decay * self.avg_val + (1 - self.decay) * current_data
        self.data.append(self.avg_val)

    def get(self):
        return self.data

    def delta(self, start, end):
        try:
            res = self.data[end] - self.data[start]
        except IndexError:
            res = self.data[end]
        return res

class TorchExponentialMovingAverage(object):
    def __init__(self, decay=0.999):
        self.decay = decay
        self.ema = {}
        self.number = {}

    def push(self, current_data):
        assert isinstance(current_data, dict), "current_data should be a dict"
        for key in current_data:
            if key in self.ema:
                # in-place
                self.ema[key] -= (1.0 - self.decay) * (self.ema[key] - current_data[key])
                self.number[key] += 1
            else:
                # self.ema[key] = copy.deepcopy(current_data[key])
                self.ema[key] = current_data[key] * (1.0 - self.decay)
                self.number[key] = 1

    def average(self):
        scaled_ema = {}
        for key in self.ema:
            scaled_ema[key] = self.ema[key] / (1.0 - self.decay ** self.number[key])
        return scaled_ema


# net: pytorch module
# strict: strict matching for set
def set_named_parameters(net, named_params, strict=True):
    assert isinstance(named_params, dict), "named_params should be a dict"
    orig_params_data = {}
    for n, p in net.named_parameters():
        orig_params_data[n] = copy.deepcopy(p.data)
    if strict:
        assert len(named_params) == len(list(net.named_parameters())), "Unmatched number of params!"
    for n, p in net.named_parameters():
        if strict:
            assert n in named_params, "Unknown param name!"
        if n in named_params:
            p.data.copy_(named_params[n])

    return orig_params_data

def next_group(g, maxlim, arch, logger):
    if g < 0 or g >= len(maxlim):
        logger.info('group index %d is out of range.' % g)
        return -1
    for i in range(len(maxlim)):
        idx = (g+i+1)%len(maxlim)
        if maxlim[idx] > arch[idx]:
            return idx
    return -1

def next_arch(mode, maxlim, arch, logger, sub=None, rate=0.333, group=0):
    tmp_arch = [v for v in arch]
    if 'all' == mode:
        tmp_arch = [v+1 for v in tmp_arch]
    elif 'group' == mode and group >= 0 and group < len(arch):
        tmp_arch[group] += 1
    elif 'rate' == mode:
        num = int(round(sum(arch)*rate))
        while num >= len(arch):
            tmp_arch = [v + 1 for v in tmp_arch]
            num -= len(arch)
        if num:
            rperm = np.random.permutation(len(arch))
            for idx in rperm[:num]:
                tmp_arch[idx] += 1
    elif 'sub' == mode and (sub is not None):
        for idx, val in enumerate(tmp_arch):
            tmp_arch[idx] += sub[idx]
    else:
        logger.fatal('Unknown mode')
        exit()

    res = []
    for r, m in zip(tmp_arch, maxlim):
        if r > m:
            res.append(m)
        else:
            res.append(r)
    return res