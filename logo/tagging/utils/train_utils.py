import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim import lr_scheduler
from bisect import bisect_right
import numpy as np

from ..utils.accuracy import get_accuracy_calculator
from ..utils.averagemeter import AverageMeter
from ..lib import layers


def get_criterion(multi_label=False, multi_label_negative_sample_weights_file = None):
    if multi_label:
        if multi_label_negative_sample_weights_file == None:
            print("Use BCEWithLogitsLoss")
            criterion = nn.BCEWithLogitsLoss().cuda()
        else:
            print("Use SigmoidCrossEntropyLossWithBalancing")
            with open(multi_label_negative_sample_weights_file, "r") as f:
                weights = [float(line) for line in f]
                criterion = layers.SigmoidCrossEntropyLossWithBalancing(np.array(weights)).cuda()
    else:
        print("Use CrossEntropyLoss")
        criterion = nn.CrossEntropyLoss().cuda()

    return criterion


def get_init_lr(args):
    if args.start_epoch == 0:
        return args.lr
    if args.lr_policy.lower() == 'step':
        lr = args.lr * args.gamma ** (args.start_epoch // args.step_size)
    elif args.lr_policy.lower() == 'multistep':
        milestones = [int(m) for m in args.milestones.split(',')]
        lr = args.lr * args.gamma ** bisect_right(milestones, args.start_epoch)
    elif args.lr_policy.lower() == 'exponential':
        lr = args.lr * args.gamma ** args.start_epoch
    elif args.lr_policy.lower() == 'plateau':
        assert args.start_epoch == 0, 'cannot resume training for plateau'
        lr = args.lr
    else:
        raise ValueError('Unknown lr policy: {}'.format(args.lr_policy))
    return lr


def set_default_hyper_parameter(args):
    args.epochs = 120
    args.batch_size = 256
    args.lr = 0.1
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.lr_policy = 'STEP'
    args.step_size = 30
    args.gamma = 0.1


def get_optimizer(model, args):
    # use default parameter for reproducible network
    if not args.force:
        print('Use default hyper parameter')
        set_default_hyper_parameter(args)

    init_lr = get_init_lr(args)
    print('initial learning rate: %f' % init_lr)

    if args.start_epoch > 0:
        groups = [dict(params=list(model.parameters()), initial_lr=init_lr)]
    else:
        groups = model.parameters()

    optimizer = torch.optim.SGD(groups, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.fixpartialfeature:
        group_decay = []
        group_no_decay = []
        for name, param in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                if 'bn' in name or 'bias' in name:
                    group_no_decay.append(param)
                else:
                    group_decay.append(param)
            else:
                param.requires_grad = False

        groups = [{'params': group_decay, 'lr': args.lr, 'weight_decay': args.weight_decay},
                  {'params': group_no_decay, 'lr': args.lr, 'weight_decay': 0}]

        optimizer = torch.optim.SGD(groups, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        return optimizer

    if args.finetune:
        group_pretrained = []
        group_new = []
        for name, param in model.named_parameters():
            if 'fc' in name:
                group_new.append(param)
            else:
                group_pretrained.append(param)
        assert len(list(model.parameters())) == len(group_pretrained) + len(group_new)
        groups = [dict(params=group_pretrained, lr=args.lr*0.01, initial_lr=init_lr*0.01),
                    dict(params=group_new,  lr=args.lr, initial_lr=init_lr)]
        optimizer = torch.optim.SGD(groups, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        return optimizer

    return optimizer

def get_scheduler(optimizer, args):
    if args.lr_policy.lower() == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma,
                                        last_epoch=args.start_epoch-1)
    elif args.lr_policy.lower() == 'multistep':
        milestones = [int(m) for m in args.milestones.split(',')]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma,
                                             last_epoch=args.start_epoch - 1)
    elif args.lr_policy.lower() == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma,
                                               last_epoch=args.start_epoch - 1)
    elif args.lr_policy.lower() == 'plateau':
        assert args.start_epoch == 0, 'cannot resume training for plateau'
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose=True)
    elif args.lr_policy.lower() == 'constant':
        scheduler = None
    else:
        raise ValueError('Unknown lr policy: {}'.format(args.lr_policy))

    return scheduler

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def train(args, train_loader, model, criterion, optimizer, epoch, logger, accuracy):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    orig_losses = AverageMeter()
    ccs_losses = AverageMeter()

    ccs_loss_layer = layers.CCSLoss()
    ccs_loss_param = 1.0

    # switch to train mode
    model.train()

    if args.BatchNormEvalMode:
        for module in model.module.children():
            module.apply(set_bn_eval)
    elif args.fixpartialfeature:
        # set the fixed feature layers bn to evaluation mode
        for module_name, module in model.module.named_children():
            if 'layer4' in module_name or 'fc' in module_name:
                continue
            module.apply(set_bn_eval)

    end = time.time()
    tic = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        # compute output
        all_outputs = model(input)
        output, feature = all_outputs[0], all_outputs[1]
        orig_loss = criterion(output, target)
        if ccs_loss_param > 0:
            # NOTE: use detach() to not calculate grad w.r.t. weight in ccs_loss
            weight = model.module.fc.weight
            ccs_loss = ccs_loss_layer(feature, weight, target)
            orig_losses.update(orig_loss.item(), input.size(0))
            ccs_losses.update(ccs_loss.item(), input.size(0))

            loss = orig_loss + ccs_loss_param*ccs_loss
        else:
            loss = orig_loss

        # measure accuracy and record loss
        accuracy.calc(output, target)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            speed = args.print_freq * args.batch_size / float(args.world_size) / (time.time() - tic)
            info_str = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Speed: {speed:.2f} samples/sec\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        epoch, i, len(train_loader), speed=speed, batch_time=batch_time,
                        data_time=data_time, loss=losses)
            if ccs_loss_param > 0:
                loss_str = 'Original Loss {orig_loss.val:.4f} ({orig_loss.avg:.4f})\t' \
                           'CCS Loss {ccs_loss.val:.4f} ({ccs_loss.avg:.4f})\t'.format(
                            orig_loss=orig_losses, ccs_loss=ccs_losses)
                info_str += loss_str
            info_str += accuracy.result_str()
            logger.info(info_str)
            tic = time.time()
