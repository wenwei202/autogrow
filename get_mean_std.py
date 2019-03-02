'''Get dataset mean and std with PyTorch.'''
from __future__ import print_function

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from copy import deepcopy
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import models.switch as ms
import models.pswitch as ps

import os
import argparse
import numpy as np
import models
import utils
import time

# from models import *
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
parser.add_argument('--batch_size', default='200', type=int, help='dataset')

args = parser.parse_args()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])
if 'SVHN' == args.dataset:
    trainset = getattr(torchvision.datasets, args.dataset)(root='./data-' + args.dataset, split='train', download=True,
                                                           transform=transform_train)
else:
    trainset = getattr(torchvision.datasets, args.dataset)(root='./data-'+args.dataset, train=True, download=True, transform=transform_train)
print('%d training samples.' % len(trainset))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
h, w = 0, 0
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(device)
    if batch_idx == 0:
        h, w = inputs.size(2), inputs.size(3)
        print(inputs.min(), inputs.max())
        chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
mean = chsum/len(trainset)/h/w
print('mean: %s' % mean.view(-1))

chsum = None
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs = inputs.to(device)
    if batch_idx == 0:
        chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
std = torch.sqrt(chsum/(len(trainset) * h * w - 1))
print('std: %s' % std.view(-1))

print('Done!')