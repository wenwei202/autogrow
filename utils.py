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

