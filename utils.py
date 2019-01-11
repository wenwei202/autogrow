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

import torch.nn as nn
import torch.nn.init as init

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