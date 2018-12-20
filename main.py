'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
import models
import utils

# from models import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--optimizer', '--opt', default='sgd', type=str, help='sgd variants (sgd, adam, amsgrad, adagrad, adadelta, rmsprop)')

parser.add_argument('--grow-interval', '--gi', default=3, type=int, help='an interval (in epochs) to grow new structures')
parser.add_argument('--grow-threshold', '--gt', default=0.2, type=float, help='the accuracy threshold to grow or stop')
parser.add_argument('--net', default='1-1-1-1', type=str, help='starting net')
parser.add_argument('--epochs', default=300, type=int, help='the number of epochs')
parser.add_argument('--batch-size', '--bz', default=128, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

def list_to_str(l):
    list(map(str, l))
    s = ''
    for v in l:
        s += str(v) + '-'
    return s[:-1]

def get_module(name, *args, **keywords):
    net = getattr(models, name)(*args, **keywords)
    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    return net

def get_optimizer(net):
    if 'sgd' == args.optimizer:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif 'adam' == args.optimizer:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif 'amsgrad' == args.optimizer:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4, amsgrad=True)
    elif 'adagrad' == args.optimizer:
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif 'adadelta' == args.optimizer:
        optimizer = optim.Adadelta(net.parameters(), weight_decay=5e-4)
    elif 'rmsprop' == args.optimizer:
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, weight_decay=5e-4)
    else:
        logger.fatal('Unknown --optimizer')
        raise ValueError('Unknown --optimizer')
    return optimizer

def params_id_to_name(net):
    themap = {}
    for k, v in net.named_parameters():
        themap[id(v)] = k
    return themap

def params_name_to_id(net):
    themap = {}
    for k, v in net.named_parameters():
        themap[k] = id(v)
    return themap

def save_all(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'name_id_map': params_name_to_id(model),
    }, path)

def load_all(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # insert missing states as empty dict
    old_name_id_map = checkpoint['name_id_map']
    new_id_name_map = params_id_to_name(model)
    new_checkpoint = deepcopy(optimizer.state_dict())
    old_checkpoint = checkpoint['optimizer_state_dict']
    if len(old_checkpoint['param_groups']) != 1 or len(new_checkpoint['param_groups']) != 1:
        logger.fatal('The number of param_groups is not 1.')
        exit()
    for new_id in new_checkpoint['param_groups'][0]['params']:
        name = new_id_name_map[new_id]
        if name in old_name_id_map:
            logger.info('loading param {} state...'.format(name))
            old_id = old_name_id_map[name]
            new_checkpoint['state'][new_id] = old_checkpoint['state'][old_id]
        else:
            if new_id not in new_checkpoint['state']:
                logger.info('initializing param {} state as an empty dict...'.format(name))
                new_checkpoint['state'][new_id] = {}
            else:
                logger.info('skipping param {} state (initial state exists)...'.format(name))
    optimizer.load_state_dict(new_checkpoint)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return epoch

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device = 'cuda' # if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

save_path = os.path.join('./results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    raise OSError('Directory {%s} exists. Use a new one.' % save_path)
logging.basicConfig(filename=os.path.join(save_path, 'log.txt'), level=logging.INFO)
logger = logging.getLogger('main')
logger.addHandler(logging.StreamHandler())
logger.info("Saving to %s", save_path)
logger.info("Running arguments: %s", args)

# Data
logger.info('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
logger.info('==> Building model..')
current_arch = list(map(int, args.net.split('-')))
max_arch = [100, 100, 100, 100]
growing_group = 0
net = get_module('ResNetBasic', current_arch)
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()

criterion = nn.CrossEntropyLoss()
optimizer = get_optimizer(net)

# Training
def train(epoch, net):
    logger.info('\nEpoch: %d (train)' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if 0 == batch_idx % 100 or batch_idx == len(trainloader) - 1:
            logger.info('(%d/%d) ==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (batch_idx+1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss / len(trainloader), 100.*correct/total

def test(epoch, net, save=False):
    logger.info('Epoch: %d (test)' % epoch)
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if 0 == batch_idx % 100 or batch_idx == len(testloader) - 1:
                logger.info('(%d/%d) ==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (batch_idx+1, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    global current_arch
    if acc > best_acc and save:
        logger.info('Saving best %.3f @ %d  ( resnet-%s )...' %(acc, epoch, list_to_str(current_arch)))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(save_path, 'best_ckpt.t7'))
    best_acc = acc if acc > best_acc else best_acc

    return test_loss / len(testloader), acc

# main func
# epoch, train loss, train accu, test loss, test accu
curves = np.zeros((args.epochs, 5))
clr1 = (0.6, 0., 0.)
clr2 = (0.42, 0.56, 0.14)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('epoch')
ax1.set_ylabel('Loss', color=clr1)
ax1.tick_params(axis='y', colors=clr1)
ax2.set_ylabel('Accuracy (%)', color=clr2)
ax2.tick_params(axis='y', colors=clr2)
emv = utils.ExponentialMovingAverage(decay=0.95)
stop_growing = False
growed = False  # if growed in the recent interval
for epoch in range(start_epoch, start_epoch+args.epochs):
    # grow or stop
    if epoch and (epoch % args.grow_interval == 0): # check after every interval
        delta_accu = emv.delta(-1 - args.grow_interval, -1)
        logger.info('******> improved %.3f (ExponentialMovingAverage) in the last %d epochs' % (delta_accu, args.grow_interval))
        if growed and delta_accu < args.grow_threshold: # just growed but no improvement
            logger.info('******> stop growing permanently')
            stop_growing = True # stop growing permanently
        if not stop_growing and delta_accu < args.grow_threshold:
            # save current model
            save_ckpt = os.path.join(save_path, 'resnet-{}_ckpt.t7'.format(list_to_str(current_arch)))
            save_all(epoch-1, net, optimizer, save_ckpt)
            # create a new net and optimizer
            current_arch[growing_group] += 1
            growing_group = (growing_group+1)%len(current_arch)
            logger.info('******> growing to resnet-%s @ epoch %d' % (list_to_str(current_arch), epoch))
            net = get_module("ResNetBasic", num_blocks=current_arch)
            optimizer = get_optimizer(net)
            loaded_epoch = load_all(net, optimizer, save_ckpt)
            # new params initialization
            # TODO
            logger.info('testing new model ...')
            test(loaded_epoch, net)
            growed = True
        else:
            growed = False
    # training and testing
    if 'sgd' == args.optimizer:
        adjust_learning_rate(optimizer, epoch, args)
    curves[epoch, 0] = epoch
    curves[epoch, 1], curves[epoch, 2] = train(epoch, net)
    curves[epoch, 3], curves[epoch, 4] = test(epoch, net, save=True)
    emv.push(curves[epoch, 4])
    # plotting
    ax1.semilogy(curves[:epoch+1, 0], curves[:epoch+1, 1], '--', color=clr1, mfc=clr1, markersize=2)
    ax1.semilogy(curves[:epoch+1, 0], curves[:epoch+1, 3], '-', color=clr1, mfc=clr1, markersize=2)
    ax2.plot(curves[:epoch+1, 0], curves[:epoch+1, 2], '--', color=clr2, mfc=clr2, markersize=2)
    ax2.plot(curves[:epoch+1, 0], curves[:epoch+1, 4], '-', color=clr2, mfc=clr2, markersize=2)
    ax1.legend(('Train loss', 'Val loss'), loc='lower right')
    ax2.legend(('Train accuracy', 'Val accuracy'), loc='upper left')
    plt.savefig(os.path.join(save_path, 'curves.pdf'))

logger.info('Done!')