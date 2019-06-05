import argparse
import os
import shutil
import six
import time

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
from torchvision.models.resnet import model_urls

import numpy as np

from qd_classifier.lib import layers
from qd_classifier.lib.dataset import TSVDataset, TSVDatasetPlusYaml
from qd_classifier.utils.parser import get_arg_parser
from qd_classifier.utils.data import get_data_loader
from qd_classifier.utils.train_utils import get_criterion, get_optimizer, get_scheduler, get_accuracy_calculator, train
from qd_classifier.utils.test import validate
from qd_classifier.utils.save_model import save_checkpoint, load_model_state_dict
from qd_classifier.utils.logger import Logger, DistributedLogger

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main(args):
    if isinstance(args, list) or isinstance(args, six.string_types):
        parser = get_arg_parser(model_names)
        args = parser.parse_args(args)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print("WORLD_SIZE = {}".format(os.environ["WORLD_SIZE"] if "WORLD_SIZE" in os.environ else -1))
    args.distributed = num_gpus > 1

    if args.distributed:
        print("Init distributed training on local_rank {}".format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        dist.barrier()

    if not args.distributed:
        logger = Logger(args.output_dir, args.prefix)
    else:
        try:
            logger = DistributedLogger(args.output_dir, args.prefix, args.local_rank)
        except:
            logger.info('Cannot create logger, rank:', args.local_rank)

    logger.info('distributed? {}'.format(args.distributed))

    if args.local_rank == 0:
        logger.info('called with arguments: {}'.format(args))

    # Data loading code
    train_loader, val_loader, train_sampler, train_dataset = get_data_loader(args, logger)

    # create model
    if args.pretrained:
        logger.info("=> using pre-trained model '{}'".format(args.arch))
        model_urls[args.arch] = model_urls[args.arch].replace('https://', 'http://')
        model = models.__dict__[args.arch](pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, train_dataset.label_dim())

        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        torch.nn.init.xavier_uniform_(model.fc.weight)
    else:
        if args.input_size == 112:
            model = layers.ResNetInput112(args.arch, train_dataset.label_dim())
        else:
            logger.info("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch](num_classes=train_dataset.label_dim())

    if args.ccs_loss_param > 0:
        model = layers.ResNetFeatureExtract(model)

    # get loss function (criterion), optimizer, and scheduler (for learning rate)
    class_weights = None
    if args.balance_class:
        assert not args.balance_sampler
        class_counts = train_dataset.label_counts
        num_pos_classes = np.count_nonzero(class_counts)
        assert num_pos_classes > 0
        class_weights = np.zeros(train_dataset.label_dim())
        for idx, c in enumerate(class_counts):
            if c > 0:
                class_weights[idx] = float(len(train_dataset)) / (num_pos_classes * c)
        logger.info("use balanced class weights")
        class_weights = torch.from_numpy(class_weights).float().cuda()

    criterion = get_criterion(train_dataset.is_multi_label(), args.neg_weight_file, class_weights=class_weights)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    accuracy = get_accuracy_calculator(multi_label=train_dataset.is_multi_label())

    best_prec1 = 0
    best_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            load_model_state_dict(model, checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_prec1 = checkpoint['best_prec1']
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    if args.custom_pretrained:
        assert(not args.resume and os.path.isfile(args.custom_pretrained))
        logger.info("=> loading pretrained model '{}'".format(args.custom_pretrained))
        checkpoint = torch.load(args.custom_pretrained)
        load_model_state_dict(model, checkpoint['state_dict'], skip_unmatched_layers=args.skip_unmatched_layers)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    cudnn.benchmark = True
    if args.evaluate:
        prec1 = validate(val_loader, model, criterion, logger)
        print("top1 accuracy: {}".format(prec1))
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if scheduler != None:
            scheduler.step()

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch, logger, accuracy)

        if args.local_rank == 0:
            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, logger)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                best_epoch = epoch
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'num_classes': train_dataset.label_dim(),
                'multi_label': train_dataset.is_multi_label(),
                'labelmap': train_dataset.get_labelmap(),
                'best_prec1': best_prec1,
            }, is_best, args.prefix, epoch, args.output_dir)
            info_str = 'Epoch: [{0}]\t' \
                       'Best Epoch {1:d}\t' \
                       'Best Prec1 {2:.3f}'.format(epoch, best_epoch, best_prec1)
            logger.info(info_str)


if __name__ == '__main__':
    torch.manual_seed(2018)
    parser = get_arg_parser(model_names)
    main(parser.parse_args())
