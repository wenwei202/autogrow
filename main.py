import argparse
import os
import random
import time
import warnings
import pickle
import collections
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# get logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--feature-analyze', dest='feature_analyze', action='store_true',
                    help='analyze features')
parser.add_argument('--maskout', dest='maskout', action='store_true',
                    help='whether use mask for training')
parser.add_argument('--workspace', default='myworkspace', type=str,
                    help='the directory of workspace to save results')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        logger.info("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        logger.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # add hooks
    g_mask_batch = collections.OrderedDict()
    all_masks = None
    mean_mask_dir = args.workspace + "/" + args.arch + "/mean_feature_masks"
    ratio_mask_dir = args.workspace + "/" + args.arch + "/ratio_feature_masks"
    if args.maskout:
        tmp = torch.zeros(args.batch_size)
        tmp = torch.chunk(tmp, torch.cuda.device_count())
        strides = [t.size()[0] for t in tmp]
        def myhook(m, input, name=None, strides=None):
            device_idx = input[0].device.index
            sidx = sum([s for s in strides[0:device_idx]])
            eidx = sum([s for s in strides[0:device_idx+1]])
            input[0].mul_(g_mask_batch[name][sidx:eidx].cuda(input[0].device).float())
        for idx, m in enumerate(model.named_modules()):
            if isinstance(m[1], nn.Conv2d):
                logger.info('\t{} registering hook...'.format(m[0]))
                m[1].register_forward_pre_hook(hook=partial(myhook, name=m[0], strides=strides))
        all_masks = get_all_masks(mean_mask_dir, num_classes=1000)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
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

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.feature_analyze:
        if args.gpu is None:
            logger.error('Only single-gpu mode is supported in feature analysis')
            exit()
        mean_dir = args.workspace + "/"+ args.arch+"/mean_features"
        ratio_dir = args.workspace + "/"+ args.arch+"/ratio_features"
        fig_mean_dir = args.workspace + "/"+ args.arch+"/fig_mean_features"
        fig_ratio_dir = args.workspace + "/" + args.arch + "/fig_ratio_features"
        fig_mean_mask_dir = args.workspace + "/" + args.arch + "/fig_mean_masks"
        fig_ratio_mask_dir = args.workspace + "/" + args.arch + "/fig_ratio_masks"
        feature_analyze_all_classes(val_loader, model, criterion, directory=mean_dir)
        get_contri_ratios(from_dir=mean_dir, to_dir=ratio_dir)
        seed = datetime.now().microsecond
        plot_features(feature_dir=mean_dir, fig_dir=fig_mean_dir, seed=seed)
        plot_features(feature_dir=ratio_dir, fig_dir=fig_ratio_dir, seed=seed)
        get_masks_by_norm(from_dir=mean_dir, to_dir=mean_mask_dir)
        get_masks_by_norm(from_dir=ratio_dir, to_dir=ratio_mask_dir)
        plot_features(feature_dir=mean_mask_dir, fig_dir=fig_mean_mask_dir, seed=seed)
        plot_features(feature_dir=ratio_mask_dir, fig_dir=fig_ratio_mask_dir, seed=seed)
        logger.info("Done!")
        return

    if args.evaluate:
        validate(val_loader, model, criterion, mask_batch_ptr=g_mask_batch, all_masks=all_masks)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, mask_batch_ptr=g_mask_batch, all_masks=all_masks)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, mask_batch_ptr=None, all_masks=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # fetch masks
            if (mask_batch_ptr is not None) and (all_masks is not None):
                assert (args.maskout)
                for layer, _ in all_masks.iteritems():
                    maskb = [all_masks[layer][t] for t in target]
                    mask_batch_ptr[layer] = torch.stack(maskb, dim=0)

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


# get masks for all classes
# saving format: all_masks[layer_name][class_index]
def get_all_masks(mask_dir, num_classes=1000):
    logger.info("collecting feature masks from {}...".format(mask_dir))
    if not os.path.exists(mask_dir):
        logger.error("\t{} does not exist!".format(mask_dir))
        exit()
    all_masks = collections.OrderedDict()
    allfiles = sorted_filenames(mask_dir)
    assert (len(allfiles) == num_classes)
    for f in allfiles:
        logger.info("\tretrieving from {}".format(f))
        cur_label = int(f.split("_")[-1])
        mask = load_obj(mask_dir + "/" + f)
        for key, value in mask.iteritems():
            if key in all_masks:
                assert (all_masks[key][cur_label] is None)
            else:
                all_masks[key] = [None] * num_classes
            if num_classes > 100 or (args.gpu is None):
                # Move to cpu to avoid OOM in GPUs when we have lots of classes
                # In multi-gpu mode, we may copy from a CPU to GPUs
                all_masks[key][cur_label] = value.cpu()
            else:
                all_masks[key][cur_label] = value
    return all_masks


# analyze statistics of features for all classes
# for efficient execution:
# ensure loader loads samples in the order of labels
# that is, make sure shuffling is disable in the loader
def feature_analyze_all_classes(loader, model, criterion, directory = 'mean_features', num_classes=1000):
    # add hook to store input features of each conv2d
    logger.info("Analyzing features of all classes...")
    if os.path.exists(directory):
        logger.warning("\t{} exists! Skipped.".format(directory))
        return
    os.makedirs(directory)
    conv2d_inputs = collections.OrderedDict()
    def myhook(m, input, output, name=None):
        conv2d_inputs[name] = torch.sum(torch.abs(input[0]), dim=0)
    handles = []
    layer_names = []
    for idx, m in enumerate(model.named_modules()):
        if isinstance(m[1], nn.Conv2d):
            logger.info('\t{} registering hook...'.format(m[0]))
            h = m[1].register_forward_hook(hook=partial(myhook, name=m[0]))
            handles.append(h)
            layer_names.append(m[0])

    batch_time = AverageMeter()
    total_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start = time.time()
        end = time.time()
        counts = np.zeros(num_classes)
        for i, (input, target) in enumerate(loader):
            # concatenate data matching the label]
            labels = torch.unique(target, sorted=True)
            inputs_matched = []
            targets_matched = []
            for label in labels:
                indices_matched = (target == label).nonzero().reshape(-1)
                inputs_matched.append(input[indices_matched])
                targets_matched.append(target[indices_matched])
            assert (sum([e.size()[0] for e in inputs_matched]) == input.size()[0])
            assert (sum([e.size()[0] for e in targets_matched]) == input.size()[0])

            if args.gpu is not None:
                for idx, val in enumerate(inputs_matched):
                    inputs_matched[idx] = inputs_matched[idx].cuda(args.gpu, non_blocking=True)

            for idx, val in enumerate(targets_matched):
                targets_matched[idx] = targets_matched[idx].cuda(args.gpu, non_blocking=True)

            # compute output
            for input_one_class, target_one_class in zip(inputs_matched, targets_matched):
                counts[target_one_class[0]] += target_one_class.size()[0]
                assert (input_one_class.size()[0])
                file_name = "{path}/class_{num:03d}".format(path=directory, num=target_one_class[0])
                try:
                    history_conv2d_inputs = load_obj(file_name)
                except IOError:
                    history_conv2d_inputs = None
                # run forward and save current features by hooks
                output = model(input_one_class)
                # accumulate features
                if history_conv2d_inputs is not None:
                    for key, value in conv2d_inputs.iteritems():
                        assert (key in history_conv2d_inputs)
                        conv2d_inputs[key] += history_conv2d_inputs[key]
                # save new features
                save_obj(conv2d_inputs, file_name)

                loss = criterion(output, target_one_class)
                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target_one_class, topk=(1, 5))
                losses.update(loss.item(), input_one_class.size(0))
                top1.update(prec1[0], input_one_class.size(0))
                top5.update(prec5[0], input_one_class.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('\tTest: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        logger.info ("\tTotally {} images were processed".format(sum(counts)))
        print counts
        # averging
        allfiles = sorted_filenames(directory)
        for f in allfiles:
            cur_label = int(f.split("_")[-1])
            features = load_obj(directory + "/" + f)
            for key, value in features.iteritems():
                features[key] = features[key] / counts[cur_label]
            save_obj(features, directory + "/" + f)

        total_time.update(time.time() - start)
        logger.info('\t * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {total_time.avg:.3f}s'
              .format(top1=top1, top5=top5, total_time=total_time))

    # remove hooks
    for h in handles:
        h.remove()

    return top1.avg

def sorted_filenames(dir):
    allfiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    allfiles = sorted(allfiles)
    allfiles = [os.path.splitext(f)[0] for f in allfiles]
    return allfiles

def get_contri_ratios(from_dir='mean_features', to_dir="ratio_features"):
    logger.info("Getting the contribution ratios...")
    if not os.path.exists(from_dir):
        logger.error("\t{} does not exist!".format(from_dir))
        exit()
    if os.path.exists(to_dir):
        logger.warning("\t{} exists! Skipped.".format(to_dir))
        return
    else:
        os.makedirs(to_dir)

    allfiles = sorted_filenames(from_dir)
    vals_sum = None
    logger.info("\tGetting the sum...")
    for f in allfiles:
        label = int(f.split("_")[-1])
        if vals_sum is None:
            vals_sum = load_obj(from_dir+"/"+f)
        else:
            features = load_obj(from_dir+"/"+f)
            for key, value in features.iteritems():
                assert (key in vals_sum)
                vals_sum[key] += features[key]
    logger.info("\tCalculating the ratios...")
    for f in allfiles:
        features = load_obj(from_dir+"/"+f)
        for key, value in features.iteritems():
            features[key] = features[key]/(vals_sum[key]+1.0e-8)
        save_obj(features, to_dir+"/"+f)

def plot_features(feature_dir="ratio_features", fig_dir="fig_features", seed=123):
    logger.info("plotting features...")

    if not os.path.exists(feature_dir):
        logger.error("\t{} does not exist!".format(feature_dir))
        exit()
    if os.path.exists(fig_dir):
        logger.warning("\t{} exists! Overwriting...".format(fig_dir))
    else:
        os.makedirs(fig_dir)

    allfiles = sorted_filenames(feature_dir)
    np.random.seed(seed)
    indices = np.random.randint(len(allfiles), size=3)
    for idx in indices:
        f = allfiles[idx]
        logger.info("\tplotting {}".format(f))
        features = load_obj(feature_dir+"/"+f)
        layer_count = 0
        for key, value in features.iteritems():
            fig = plt.figure()
            feature = features[key]
            vmin = feature.min().cpu().numpy()
            vmax = feature.max().cpu().numpy()
            num_channels = feature.size()[0]
            side_size = int(np.ceil(np.sqrt(num_channels)))
            plt.subplot(side_size+1, 1, 1)
            plt.hist(feature.cpu().numpy().flatten(), bins=200)
            plt.xlim(left=0)
            plt.xlim(right=vmax)
            plt.tick_params(labelbottom=False, labeltop=True)
            for c in range(num_channels):
                plt.subplot(side_size+1, side_size, c + 1 + side_size)
                plt.imshow(feature[c],  interpolation='none', cmap=plt.get_cmap('Greys'), vmin=vmin, vmax=vmax)
                plt.tick_params(which='both', labelbottom=False, labelleft=False, bottom=False, top=False, left=False,
                                right=False)
            fig.suptitle(feature_dir+"/"+f+"_{}".format(layer_count))
            fig.savefig(fig_dir+"/"+'{}_{}.pdf'.format(layer_count, f))
            layer_count += 1

def get_top_by_norm(input, ratio=0.8, p=2):
    finput = input.reshape((-1,))
    s = finput.size()[0]
    target_sum = input.sum()

def get_masks_by_norm(from_dir='mean_features', to_dir="feature_masks", ratio=0.8, p=2):
    logger.info("Getting masks...")
    if not os.path.exists(from_dir):
        logger.error("\t{} does not exist!".format(from_dir))
        exit()
    if os.path.exists(to_dir):
        logger.warning("\t{} exists! Skipped.".format(to_dir))
        return
    else:
        os.makedirs(to_dir)

    allfiles = sorted_filenames(from_dir)
    for f in allfiles:
        logger.info("\tProcessing {}".format(f))
        features = load_obj(from_dir+"/"+f)
        masks = collections.OrderedDict()
        for key, value in features.iteritems():
            vlen = torch.norm(value, p=p)
            target_val = (ratio * vlen)**p
            sorted_value, _ = torch.sort(value.reshape(-1).pow(p),descending=True)
            left_idx = 0
            right_idx = sorted_value.size()[0]-1
            while right_idx > left_idx:
                mid_idx = (right_idx + left_idx) / 2
                mid_sum = torch.sum(sorted_value[0:mid_idx+1])
                if abs(mid_sum - target_val)<1.0e-6:
                    left_idx = mid_idx
                    right_idx = mid_idx
                elif mid_sum > target_val:
                    right_idx = mid_idx - 1
                else:
                    left_idx = mid_idx + 1
            threhold = sorted_value[left_idx] ** (1.0/p)
            masks[key] = value.ge(threhold)
        save_obj(masks, to_dir+"/"+f)

# analyze statistics of features for a target class
def feature_analyze_per_class(loader, label, model, criterion):
    batch_time = AverageMeter()
    total_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start = time.time()
        end = time.time()
        input_one_class = None
        target_one_class = None
        for i, (input, target) in enumerate(loader):
            # concate data matching the label
            indices_matched = (target==label).nonzero().reshape(-1)
            input_matched = input[indices_matched]
            target_matched = target[indices_matched]
            if input_one_class is None:
                assert (target_one_class is None)
                input_one_class = input_matched
                target_one_class = target_matched
            else:
                input_one_class = torch.cat((input_one_class, input_matched))
                target_one_class = torch.cat((target_one_class, target_matched))
            assert (input_one_class.size()[0] == target_one_class.size()[0])

            if (input_one_class.size()[0] < args.batch_size) and (
                    i != len(loader)-1):
                continue

            if args.gpu is not None:
                input_one_class = input_one_class.cuda(args.gpu, non_blocking=True)
            target_one_class = target_one_class.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input_one_class)
            loss = criterion(output, target_one_class)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target_one_class, topk=(1, 5))
            losses.update(loss.item(), input_one_class.size(0))
            top1.update(prec1[0], input_one_class.size(0))
            top5.update(prec5[0], input_one_class.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

            # clear the buffer
            input_one_class = None
            target_one_class = None
        total_time.update(time.time() - start)
        logger.info(' * Class {label} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {total_time.avg:.3f}s'
              .format(label=label, top1=top1, top5=top5, total_time=total_time))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
