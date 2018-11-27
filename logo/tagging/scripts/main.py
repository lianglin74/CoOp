import argparse
import os
import shutil
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

from lib.dataset import TSVDataset, TSVDatasetPlusYaml
from utils.parser import parse_args
from utils.data import get_data_loader
from utils.train_utils import get_criterion, get_optimizer, get_scheduler, train
from utils.test import validate
from utils.save_model import save_checkpoint
from utils.logger import Logger, DistributedLogger

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main(args):
    best_prec1 = 0

    args.distributed = args.world_size > 1

    if not args.distributed:
        logger = Logger(args.output_dir, args.prefix)
    else:
        try:
            logger = DistributedLogger(args.output_dir, args.prefix, args.rank)
        except:
            logger.info('Cannot create logger, rank:', args.rank)

    logger.info('distributed? {}'.format(args.distributed))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    if args.rank == 0:
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
        logger.info("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=train_dataset.label_dim())

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # get loss function (criterion), optimizer, and scheduler (for learning rate)
    criterion = get_criterion(train_dataset.is_multi_label(), args.neg_weight_file)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

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


    if args.evaluate:
        validate(val_loader, model, criterion, logger)
        return

    for epoch in range(args.start_epoch, args.epochs):
        epoch_tic = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)

        scheduler.step()

        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch, logger)

        if args.rank == 0:
            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, logger)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                'num_classes': train_dataset.label_dim(),
                'multi_label': train_dataset.is_multi_label(),
                'labelmap': train_dataset.get_labelmap(),
            }, is_best, args.prefix, epoch+1, args.output_dir)
            info_str = 'Epoch: [{0}]\t' \
                        'Time {time:.3f}\t'.format(epoch, time=time.time()-epoch_tic)
            logger.info(info_str)


if __name__ == '__main__':
    torch.manual_seed(2018)
    args = parse_args(model_names)
    main(args)
