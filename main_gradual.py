'''Train CIFAR10 with PyTorch.'''
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
import torchvision.datasets as datasets
import models.switch as ms
import models.pswitch as ps

import os
import argparse
import numpy as np
import models
import utils
import time
import GPUtil
import gc
import pickle

# from models import *
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--optimizer', '--opt', default='sgd', type=str, help='sgd variants (sgdc, sgd, adam, amsgrad, adagrad, adadelta, rmsprop)')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--switch-reg', '--sr', default=None, type=float, help='lambda of L1 regularization on pswitch')
parser.add_argument('--epochs', default=1000, type=int, help='the number of epochs')
parser.add_argument('--grow-threshold', '--gt', default=0.05, type=float, help='the accuracy threshold to grow or stop')
parser.add_argument('--ema-params', '--ep', action='store_true', help='validating accuracy by a exponentially moving average of parameters')
parser.add_argument('--growing-mode', default='group', type=str, help='how new structures are added (rate, all, sub, group)')
parser.add_argument('--tail-epochs', '--te', default=90, type=int, help='the number of epochs after growing epochs (--epochs)')
parser.add_argument('--pswitch-thre', '--pt', default=0.005, type=float, help='threshold to zero pswitchs')

parser.add_argument('--batch-size', '--bz', default=256, type=int, help='batch size')
parser.add_argument('--switch-off', '--so', action='store_true', help='switch off at initialization')
parser.add_argument('--grow-interval', '--gi', default=1, type=int, help='an interval (in epochs) to grow new structures')
parser.add_argument('--stop-interval', '--si', default=30, type=int, help='an interval (in epochs) to grow new structures')
parser.add_argument('--net', default='1-1-1-1', type=str, help='starting net')
parser.add_argument('--sub-net', default='1-1-1-1', type=str, help='a sub net to grow')
parser.add_argument('--max-net', default='6-16-72-32', type=str, help='The maximum net')
parser.add_argument('--residual', default='ResNetBasic', type=str,
                    help='the type of residual block (ResNetBasic or ResNetBottleneck)')
parser.add_argument('--initializer', '--init', default='gaussian', type=str, help='initializers of new structures (zero, uniform, gaussian, adam)')

parser.add_argument('--rate', default=0.4, type=float, help='the rate to grow when --growing-mode=rate')
parser.add_argument('--growing-metric', default='max', type=str, help='the metric for growing (max or avg)')
parser.add_argument('--reset-states', '--rs', action='store_true', help='reset optimizer states or not (such as momentum)')
parser.add_argument('--init-meta', default=1.0, type=float, help='a meta parameter for initializer')
parser.add_argument('--evaluate', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')
parser.add_argument('--data', default='./imagenet', type=str, metavar='PATH',
                    help='path to imagenet dataset (default: ./imagenet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')

args = parser.parse_args()
cur_chunk_num = 1

def list_to_str(l):
    list(map(str, l))
    s = ''
    for v in l:
        s += str(v) + '-'
    return s[:-1]

def get_module(name, switch_steps, *_args, **keywords):
    net = getattr(models, name)(*_args, **keywords)
    net = net.to('cuda')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    configure_switch_policy(net, switch_steps)
    return net

def get_optimizer(net):
    if 'sgd' == args.optimizer or 'sgdc' == args.optimizer:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif 'adam' == args.optimizer:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    elif 'amsgrad' == args.optimizer:
        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4, amsgrad=True)
    elif 'adagrad' == args.optimizer:
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=1e-4)
    elif 'adadelta' == args.optimizer:
        optimizer = optim.Adadelta(net.parameters(), weight_decay=1e-4)
    elif 'rmsprop' == args.optimizer:
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, weight_decay=1e-4)
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

def save_all(epoch, train_accu, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'train_accu': train_accu,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'name_id_map': params_name_to_id(model),
    }, path)

def zerout_pswitchs(model, threshold=0.0001, log=False):
    total = 0
    zeroed = 0
    for idx, m in enumerate(model.named_modules()):
        if isinstance(m[1], ps.PSwitch):
            total += 1
            # logger.info('******> switch {} is {}...'.format(m[0], m[1].get_switch()))
            if m[1].switch.data.abs() < threshold:
                if log:
                    logger.info('******> switch {} is zeroed out...'.format(m[0]))
                m[1].switch.data.fill_(0.0)
                zeroed += 1
    if log:
        logger.info('%d/%d zeroed out' % (zeroed, total))

def reg_pswitchs(model):
    reg = 0.0
    for idx, m in enumerate(model.named_modules()):
        if isinstance(m[1], ps.PSwitch):
            reg += m[1].switch.norm(p=1)
    return reg * args.switch_reg

def print_switchs(model):
    for idx, m in enumerate(model.named_modules()):
        if isinstance(m[1], ms.Switch):
            logger.info('******> switch {} is {}...'.format(m[0], m[1].get_switch()))

def print_pswitchs(model):
    for idx, m in enumerate(model.named_modules()):
        if isinstance(m[1], ps.PSwitch):
            logger.info('******> switch {} is {}...'.format(m[0], m[1].get_switch()))

def increase_switchs(model):
    for idx, m in enumerate(model.named_modules()):
        if isinstance(m[1], ms.Switch):
            m[1].increase()

def configure_switch_policy(model, steps, start=0.0, stop=1.0, mode='linear'):
    for idx, m in enumerate(model.named_modules()):
        if isinstance(m[1], ms.Switch):
            logger.info('******> configuring switch {}...'.format(m[0]))
            m[1].set_params(steps, start, stop, mode)


def load_all(model, optimizer, path):
    checkpoint = torch.load(path)
    old_name_id_map = checkpoint['name_id_map']
    new_id_name_map = params_id_to_name(model)
    # load existing params, and initializing missing ones
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    new_params = []
    if args.residual == 'ResNetBasic':
        reinit_pattern = '.*layer.*bn2\.((weight)|(bias))$'
    elif args.residual == 'ResNetBottleneck':
        reinit_pattern = '.*layer.*bn3\.((weight)|(bias))$'
    else:
        logger.fatal('Unknown --residual')
        exit()
    for n, p in model.named_parameters():
        if n not in old_name_id_map and re.match(reinit_pattern, n):
            logger.info('******> reinitializing param {} ...'.format(n))
            new_params.append(p)
            if args.initializer == 'zero':
                logger.info('******> Initializing as zeros...')
                p.data.zero_()
            elif args.initializer == 'uniform':
                logger.info('******> Initializing by uniform noises...')
                p.data.uniform_(0.0, to=args.init_meta)
            elif args.initializer == 'gaussian':
                logger.info('******> Initializing by gaussian noises')
                p.data.normal_(0.0, std=args.init_meta)
            elif args.initializer == 'adam':
                logger.info('******> Initializing by adam optimizer')
            else:
                logger.fatal('Unknown --initializer.')
                exit()
            if args.switch_off:
                switch_name = '.'.join(n.split('.')[:-2]+['switch.switch'])
                if switch_name in model.state_dict():
                    logger.info('******> resetting %s to 0.0 from %.3f' % (switch_name, model.state_dict()[switch_name]))
                    model.state_dict()[switch_name].zero_()

    if len(new_params) and args.initializer == 'adam':
        logger.info('******> Using adam to find a good initialization')
        old_train_accu = checkpoint['train_accu']
        local_optimizer = optim.Adam(new_params, lr=0.001, weight_decay=1e-4)
        max_epoch = 10
        founded = False
        for e in range(max_epoch):
            _, accu = train(e, model, local_optimizer, chunk_num=cur_chunk_num)
            if accu > old_train_accu - 0.5:
                logger.info('******> Found a good initial position with training accuracy %.2f (vs. old %.2f) at epoch %d' % (
                accu, old_train_accu, e))
                founded = True
                break
        if not founded:
            logger.info('******> failed to find a good initial position in %d epochs. Continue...' % max_epoch)


    # load existing states, and insert missing states as empty dict
    if not args.reset_states:
        new_checkpoint = deepcopy(optimizer.state_dict())
        old_checkpoint = checkpoint['optimizer_state_dict']
        if len(old_checkpoint['param_groups']) != 1 or len(new_checkpoint['param_groups']) != 1:
            logger.fatal('The number of param_groups is not 1.')
            exit()
        for new_id in new_checkpoint['param_groups'][0]['params']:
            name = new_id_name_map[new_id]
            if name in old_name_id_map:
                old_id = old_name_id_map[name]
                if old_id in old_checkpoint['state']:
                    logger.info('loading param {} state...'.format(name))
                    new_checkpoint['state'][new_id] = old_checkpoint['state'][old_id]
                else:
                    logger.info('initializing param {} state as an empty dict...'.format(name))
                    new_checkpoint['state'][new_id] = {}
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

def set_learning_rate(optimizer, lr):
    """Sets the learning rate """
    logger.info('\nSetting learning rate to %.6f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def decay_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1

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
train_sampler = None
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

testloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)
# Model
logger.info('==> Building model..')
current_arch = list(map(int, args.net.split('-')))
subnet_arch = list(map(int, args.sub_net.split('-')))
max_arch = list(map(int, args.max_net.split('-')))
if len(current_arch) != len(max_arch):
    logger.fatal('max_arch has different size.')
    exit()
growing_group = -1
for cnt, v in enumerate(current_arch):
    if v < max_arch[cnt]:
        growing_group = cnt
        break
net = get_module(args.residual, args.grow_interval, current_arch)
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

criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = get_optimizer(net)
param_ema = utils.TorchExponentialMovingAverage()
# Training
def train(epoch, net, own_optimizer=None, increase_switch=False, chunk_num=1):
    logger.info('\nTraining epoch %d @ %.1f sec' % (epoch, time.time()))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    print_pswitchs(net)
    if increase_switch:
        increase_switchs(net)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if own_optimizer is not None:
            own_optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        # avoid out of memory by split a batch to chunks
        if targets.size(0) < chunk_num:
            logger.warning('%d samples cannot be chunked to %d. Set the chunk number to %d' % (targets.size(0), chunk_num, targets.size(0)))
            chunk_num = targets.size(0)
        sub_inputs = inputs.chunk(chunk_num)
        sub_targets = targets.chunk(chunk_num)
        for chunk_ins, chunk_tgts in zip(sub_inputs, sub_targets):
            outputs = net(chunk_ins)
            loss = criterion(outputs, chunk_tgts)
            loss.backward()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += chunk_tgts.size(0)
            correct += predicted.eq(chunk_tgts).sum().item()
        for p in net.parameters():
            p.grad.data.div_(len(inputs))
        sreg = 0.0
        if args.switch_reg is not None:
            sreg = reg_pswitchs(net)
            sreg.backward()

        if own_optimizer is not None:
            own_optimizer.step()
        else:
            optimizer.step()

        if args.switch_reg is not None:
            zerout_pswitchs(net, args.pswitch_thre)

        # maintain a moving average
        if args.ema_params:
            params_data_dict = {}
            for n, p in net.named_parameters():
                params_data_dict[n] = p.data
            param_ema.push(params_data_dict)

        if 0 == batch_idx % 100 or batch_idx == len(trainloader) - 1:
            logger.info('(%d/%d) ==> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (batch_idx+1, len(trainloader), train_loss/total, 100.*correct/total, correct, total))
            if args.switch_reg is not None:
                logger.info('        ==> PSwitch L1 Reg.: %.6f ' % sreg)
    return train_loss / total, 100.*correct/total

def test(epoch, net, save=False):
    logger.info('Testing epoch %d @ %.1f sec' % (epoch, time.time()))
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        if args.ema_params:
            logger.info('Using average params for test')
            orig_params = utils.set_named_parameters(net, param_ema.average(), strict=False)
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
                    % (batch_idx+1, len(testloader), test_loss/total, 100.*correct/total, correct, total))

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

    with torch.no_grad():
        if args.ema_params:
            utils.set_named_parameters(net, orig_params, strict=True)

    return test_loss / total, acc

# main func

# resume and evaluate from a checkpoint
if args.evaluate:
    if os.path.isfile(args.evaluate):
        # load existing params, and initializing missing ones
        print("=> loading checkpoint '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        net.load_state_dict(checkpoint['net'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.evaluate, checkpoint['epoch']))
        print_pswitchs(net)
        logger.info('zeroing out small pswitchs...')
        zerout_pswitchs(net, args.pswitch_thre, log=True)
        test(checkpoint['epoch'], net)
        logger.info('evaluation done!')
    else:
        print("=> no checkpoint found at '{}'".format(args.evaluate))
    exit()

if args.growing_metric == 'max':
    ema = utils.MovingMaximum()
elif args.growing_metric == 'avg':
    ema = utils.ExponentialMovingAverage(decay=0.95)
else:
    logger.fatal('Unknown --growing-metric')
    exit()


def can_grow(maxlim, arch):
    for maxv, a in zip(maxlim, arch):
        if maxv > a:
            return True
    return False

num_tail_epochs = args.tail_epochs # if (args.optimizer == 'sgd' or args.optimizer == 'sgdc') else 0
last_epoch = -1
growing_epochs = []
intervals = (args.epochs - 1) // args.grow_interval + 1
# epoch, train loss, train accu, test loss, test accu, timestamps
curves = np.zeros((intervals*args.grow_interval + num_tail_epochs, 6))
for interval in range(0, intervals):
    # training and testing
    for epoch in range(interval*args.grow_interval, (interval+1)*args.grow_interval):
        if 'sgdc' == args.optimizer:
            e = epoch % args.grow_interval
            if e < args.grow_interval // 3:
                set_learning_rate(optimizer, args.lr)
            elif e < args.grow_interval * 2 // 3:
                set_learning_rate(optimizer, args.lr * 0.1)
            else:
                set_learning_rate(optimizer, args.lr * 0.01)
        curves[epoch, 0] = epoch
        curves[epoch, 1], curves[epoch, 2] = train(epoch, net, chunk_num=cur_chunk_num)
        curves[epoch, 3], curves[epoch, 4] = test(epoch, net, save=True)
        curves[epoch, 5] = time.time() / 60.0
        ema.push(curves[epoch, 4])

    # limit max arch
    logger.info('******> improved %.3f (ExponentialMovingAverage) in the last %d epochs' % (
        ema.delta(-1 - args.grow_interval, -1), args.grow_interval))
    delta_accu = ema.delta(-1 - args.stop_interval, -1)
    logger.info(
        '******> improved %.3f (ExponentialMovingAverage) in the last %d epochs' % (delta_accu, args.stop_interval))
    if delta_accu < args.grow_threshold: # no improvement
        if args.growing_mode == 'group':
            max_arch[growing_group] = current_arch[growing_group]
            logger.info('******> stop growing group %d permanently. Limited as %s .' % (growing_group, list_to_str(max_arch)))
        else:
            max_arch[:] = current_arch[:]
            logger.info('******> stop growing all permanently. Limited as %s .' % (list_to_str(max_arch)))

    if can_grow(max_arch, current_arch):
        # save current model
        save_ckpt = os.path.join(save_path, 'resnet-growing_ckpt.t7')
        save_all((interval + 1) * args.grow_interval - 1,
                 curves[(interval + 1) * args.grow_interval - 1, 2],
                 net,
                 optimizer,
                 save_ckpt)
        pickle.dump(ema, open(os.path.join(save_path, 'ema.obj'), 'w'))
        pickle.dump(curves, open(os.path.join(save_path, 'curves.obj'), 'w'))

        for gpu_stat in GPUtil.getGPUs():
            if gpu_stat.memoryFree < 1000:
                logger.info('******> hitting gpu memory limit. Only %d MB / %d MB is free in GPU %d.' % (
                gpu_stat.memoryFree, gpu_stat.memoryTotal, gpu_stat.id))
                cur_chunk_num += 1
                logger.info('******> increased chunk number to %d' % cur_chunk_num)
                gc.collect()
                break

        # create a new net and optimizer
        current_arch = utils.next_arch(args.growing_mode, max_arch, current_arch, logger, sub=subnet_arch,
                                       rate=args.rate, group=growing_group)
        logger.info(
            '******> growing to resnet-%s before epoch %d' % (list_to_str(current_arch), (interval + 1) * args.grow_interval))
        net = get_module(args.residual, args.grow_interval, num_blocks=current_arch)
        optimizer = get_optimizer(net)
        loaded_epoch = load_all(net, optimizer, save_ckpt)
        # logger.info('testing new model ...')
        # test(loaded_epoch, net)
        growing_epochs.append((interval + 1) * args.grow_interval)
        if args.growing_mode == 'group':
            growing_group = utils.next_group(growing_group, max_arch, current_arch, logger)
    else:
        logger.info('******> stop growing all groups')
        last_epoch = (interval + 1) * args.grow_interval - 1
        logger.info('******> reach limitation. Finished in advance @ epoch %d' % last_epoch)
        curves = curves[:last_epoch+1+num_tail_epochs, :]
        break
    last_epoch = (interval + 1) * args.grow_interval - 1

set_learning_rate(optimizer, args.lr)
for epoch in range(last_epoch + 1, last_epoch + 1 + num_tail_epochs):
    if ((epoch == last_epoch + 1 + num_tail_epochs // 3) or (epoch == last_epoch + 1 + num_tail_epochs * 2 // 3)) and (
            args.optimizer == 'sgd' or args.optimizer == 'sgdc'):
        logger.info('======> decaying learning rate')
        decay_learning_rate(optimizer)
    curves[epoch, 0] = epoch
    curves[epoch, 1], curves[epoch, 2] = train(epoch, net, chunk_num=cur_chunk_num)
    curves[epoch, 3], curves[epoch, 4] = test(epoch, net, save=True)
    curves[epoch, 5] = time.time() / 60.0
    ema.push(curves[epoch, 4])

# align time
for e in range(curves.shape[0]):
    curves[curves.shape[0]-1-e, 5] -= curves[0, 5]

# plotting
plot_segs = [0] + growing_epochs
if len(growing_epochs) == 0 or growing_epochs[-1] != curves.shape[0]-1:
    plot_segs = plot_segs + [curves.shape[0]-1]
logger.info('growing epochs {}'.format(list_to_str(growing_epochs)))
logger.info('curves: \n {}'.format(np.array_str(curves)))
np.savetxt(os.path.join(save_path, 'curves.dat'), curves)
clr1 = (0.5, 0., 0.)
clr2 = (0.0, 0.5, 0.)
fig, ax1 = plt.subplots()
fig2, ax3 = plt.subplots()
ax2 = ax1.twinx()
ax4 = ax3.twinx()
ax1.set_xlabel('epoch')
ax1.set_ylabel('Loss', color=clr1)
ax1.tick_params(axis='y', colors=clr1)
ax2.set_ylabel('Accuracy (%)', color=clr2)
ax2.tick_params(axis='y', colors=clr2)

ax3.set_xlabel('time (mins)')
ax3.set_ylabel('Loss', color=clr1)
ax3.tick_params(axis='y', colors=clr1)
ax4.set_ylabel('Accuracy (%)', color=clr2)
ax4.tick_params(axis='y', colors=clr2)

# ax2.set_ylim(80, 100) # no plot if enabled
for idx in range(len(plot_segs)-1):
    start = plot_segs[idx]
    end = plot_segs[idx+1] + 1 if (plot_segs[idx+1] == curves.shape[0] - 1) else plot_segs[idx+1]
    markersize = 12
    coef = 2. if idx % 2 else 1.
    if idx == len(plot_segs)-2:
        ax1.semilogy(curves[start:end, 0], curves[start:end, 1], '--', color=[c*coef for c in clr1], markersize=markersize)
        ax1.semilogy(curves[start:end, 0], curves[start:end, 3], '-', color=[c*coef for c in clr1], markersize=markersize)
        ax2.plot(curves[start:end, 0], curves[start:end, 2], '--', color=[c*coef for c in clr2], markersize=markersize)
        ax2.plot(curves[start:end, 0], curves[start:end, 4], '-', color=[c*coef for c in clr2], markersize=markersize)

        ax3.semilogy(curves[start:end, 5], curves[start:end, 1], '--', color=[c * coef for c in clr1], markersize=markersize)
        ax3.semilogy(curves[start:end, 5], curves[start:end, 3], '-', color=[c * coef for c in clr1], markersize=markersize)
        ax4.plot(curves[start:end, 5], curves[start:end, 2], '--', color=[c * coef for c in clr2], markersize=markersize)
        ax4.plot(curves[start:end, 5], curves[start:end, 4], '-', color=[c * coef for c in clr2], markersize=markersize)
    else:
        ax1.semilogy(curves[start:end, 0], curves[start:end, 1], '--', color=[c*coef for c in clr1], markersize=markersize, label='_nolegend_')
        ax1.semilogy(curves[start:end, 0], curves[start:end, 3], '-', color=[c*coef for c in clr1], markersize=markersize, label='_nolegend_')
        ax2.plot(curves[start:end, 0], curves[start:end, 2], '--', color=[c*coef for c in clr2], markersize=markersize, label='_nolegend_')
        ax2.plot(curves[start:end, 0], curves[start:end, 4], '-', color=[c*coef for c in clr2], markersize=markersize, label='_nolegend_')

        ax3.semilogy(curves[start:end, 5], curves[start:end, 1], '--', color=[c * coef for c in clr1], markersize=markersize, label='_nolegend_')
        ax3.semilogy(curves[start:end, 5], curves[start:end, 3], '-', color=[c * coef for c in clr1], markersize=markersize, label='_nolegend_')
        ax4.plot(curves[start:end, 5], curves[start:end, 2], '--', color=[c * coef for c in clr2], markersize=markersize, label='_nolegend_')
        ax4.plot(curves[start:end, 5], curves[start:end, 4], '-', color=[c * coef for c in clr2], markersize=markersize, label='_nolegend_')

ax2.plot(curves[:, 0], ema.get(), '-', color=[1.0, 0, 1.0])
logger.info('Val accuracy moving average: \n {}'.format(np.array_str(np.array(ema.get()))))
np.savetxt(os.path.join(save_path, 'ema.dat'), np.array(ema.get()))
ax2.set_ylim(bottom=40, top=100)
ax1.legend(('Train loss', 'Val loss'), loc='lower right')
ax2.legend(('Train accuracy', 'Val accuracy', 'Val max'), loc='lower left')
fig.savefig(os.path.join(save_path, 'curves-vs-epochs.pdf'))

ax4.plot(curves[:, 5], ema.get(), '-', color=[1.0, 0, 1.0])
ax4.set_ylim(bottom=20, top=85)
ax3.legend(('Train loss', 'Val loss'), loc='lower right')
ax4.legend(('Train accuracy', 'Val accuracy', 'Val moving avg'), loc='lower left')
fig2.savefig(os.path.join(save_path, 'curves-vs-time.pdf'))


logger.info('Done!')