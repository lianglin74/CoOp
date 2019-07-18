import argparse
from datetime import datetime
import glob
import logging
import numpy as np
import os
import os.path as op
from pprint import pformat
import shutil
import six
import sys
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
import torchvision.models as models
from torchvision.models.resnet import model_urls

from qd_classifier.lib import layers
from qd_classifier.utils.parser import get_arg_parser
from qd_classifier.utils.data import get_train_data_loader
from qd_classifier.utils.comm import get_dist_url, synchronize, init_random_seed, save_parameters
from qd_classifier.utils.train_utils import get_criterion, get_optimizer, get_scheduler, get_accuracy_calculator, train_epoch
from qd_classifier.utils.test import validate
from qd_classifier.utils.save_model import load_model_state_dict

from qd.qd_common import get_mpi_local_rank, get_mpi_local_size, get_mpi_rank, get_mpi_size, ensure_directory
from qd.qd_common import zip_qd

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class ClassifierPipeline(object):
    def __init__(self, config):
        self.config = config

        self.model_prefix = "model_epoch"
        self.full_expid = '.'.join([self.config.data, self.config.arch, self.config.expid])
        self.output_folder = op.join(self.config.output_dir, self.full_expid)

        self.mpi_rank = get_mpi_rank()
        self.mpi_size= get_mpi_size()
        self.mpi_local_rank = get_mpi_local_rank()
        self.mpi_local_size = get_mpi_local_size()

        if 'WORLD_SIZE' in os.environ:
            assert int(os.environ['WORLD_SIZE']) == self.mpi_size
        if 'RANK' in os.environ:
            assert int(os.environ['RANK']) == self.mpi_rank

        if self.mpi_size > 1:
            self.distributed = True
        else:
            self.distributed = False
        self.is_master = self.mpi_rank == 0

        assert (self.config.effective_batch_size % self.mpi_size) == 0, (self.config.effective_batch_size, self.mpi_size)
        self.config.batch_size = self.config.effective_batch_size // self.mpi_size
        self._initialized = False

    def ensure_train(self):
        self._ensure_initialized()

        last_model_file = self._get_checkpoint_file()
        if op.isfile(last_model_file):
            logging.info('last model file = {}'.format(last_model_file))
            logging.info('skip to train')
            return

        ensure_directory(op.join(self.output_folder, 'snapshot'))

        if self.mpi_rank == 0:
            all_params = vars(self.config)
            all_params.update({"FULL_EXPID": self.full_expid})
            save_parameters(all_params, self.output_folder)

        logging.info(pformat(vars(self.config)))
        logging.info('torch version = {}'.format(torch.__version__))

        trained_model = self.train()
        synchronize()

        # save the source code after training
        if self.mpi_rank == 0 and not self.config.debug:
            zip_qd(op.join(self.output_folder, 'source_code'))

        return trained_model

    def train(self):
        train_dataset, train_loader, train_sampler = get_train_data_loader(self.config, self.distributed)
        num_classes = train_dataset.get_num_labels()

        model = self._get_model(num_classes)
        model = self._data_parallel_wrap(model)
        self.init_model(model)

        optimizer = get_optimizer(model, self.config)

        assert self.config.start_epoch == 0
        last_checkpoint = None
        if self.config.restore_latest_snapshot:
            last_checkpoint = self._get_latest_checkpoint()
            if last_checkpoint and self.config.resume:
                logging.info("overriding resume: {} => {}".format(self.config.resume, last_checkpoint))
            elif self.config.resume:
                assert op.isfile(self.config.resume), "file not exist: {}".format(self.config.resume)
                last_checkpoint = self.config.resume

        if last_checkpoint:
            logging.info("=> loading checkpoint '{}'".format(last_checkpoint))
            checkpoint = torch.load(last_checkpoint)
            self.config.start_epoch = checkpoint['epoch']
            load_model_state_dict(model, checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            logging.info("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))

        criterion = self._get_criterion(train_dataset)
        accuracy = get_accuracy_calculator(multi_label=train_dataset.is_multi_label())
        # create schedule after resume to properly set start_epoch for learning rate
        scheduler = get_scheduler(optimizer, self.config)

        logging.info('start to train')
        for epoch in range(self.config.start_epoch, self.config.epochs):
            if self.distributed:
                train_sampler.set_epoch(epoch)

            if scheduler != None:
                scheduler.step()

            # train for one epoch
            model = train_epoch(self.config, train_loader, model, criterion, optimizer, epoch, accuracy)

            if self.is_master:
                torch.save({
                    'epoch': epoch + 1,
                    'arch': self.config.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'num_classes': train_dataset.get_num_labels(),
                    'multi_label': train_dataset.is_multi_label(),
                    'labelmap': train_dataset.get_labelmap(),
                }, self._get_checkpoint_file(epoch = epoch+1))

        return model

    def _get_model(self, num_classes):
        # create model
        if self.config.pretrained:
            logging.info("=> using pre-trained model '{}'".format(self.config.arch))
            model_urls[self.config.arch] = model_urls[self.config.arch].replace('https://', 'http://')
            model = models.__dict__[self.config.arch](pretrained=True)
            if model.fc.weight.shape[0] != num_classes:
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            # for m in model.modules():
            #     if isinstance(m, nn.Linear):
            #         nn.init.normal_(m.weight, 0, 0.01)
            #         nn.init.constant_(m.bias, 0)
            torch.nn.init.xavier_uniform_(model.fc.weight)
        else:
            if self.config.input_size == 112:
                model = layers.ResNetInput112(self.config.arch, num_classes)
            else:
                logging.info("=> creating model '{}'".format(self.config.arch))
                model = models.__dict__[self.config.arch](num_classes=num_classes)

        if self.config.ccs_loss_param > 0:
            model = layers.ResNetFeatureExtract(model)

        return model

    def _data_parallel_wrap(self, model):
        if self.distributed:
            model.cuda()
            if self.mpi_local_size > 1:
                model = torch.nn.parallel.DistributedDataParallel(model,
                        device_ids=[self.mpi_local_rank])
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            if self.config.arch.startswith('alexnet') or self.config.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()
        return model

    def init_model(self, model):
        if self.config.init_from:
            assert(not self.config.resume and op.isfile(self.config.init_from))
            logging.info("=> loading pretrained model '{}'".format(self.config.init_from))
            checkpoint = torch.load(self.config.init_from)
            load_model_state_dict(model, checkpoint['state_dict'], skip_unmatched_layers=self.config.skip_unmatched_layers)

    def _get_criterion(self, train_dataset):
        class_weights = None
        if self.config.balance_class:
            assert not self.config.balance_sampler
            class_counts = train_dataset.label_counts
            num_pos_classes = np.count_nonzero(class_counts)
            assert num_pos_classes > 0
            class_weights = np.zeros(train_dataset.label_dim())
            for idx, c in enumerate(class_counts):
                if c > 0:
                    class_weights[idx] = float(len(train_dataset)) / (num_pos_classes * c)
            logging.info("use balanced class weights")
            class_weights = torch.from_numpy(class_weights).float().cuda()

        criterion = get_criterion(train_dataset.is_multi_label(), self.config.neg_weight_file, class_weights=class_weights)
        return criterion

    def _ensure_initialized(self):
        if self._initialized:
            return

        torch.backends.cudnn.benchmark = True

        self._setup_logging()
        if self.distributed:
            dist_url = get_dist_url(self.config.init_method_type, self.config.dist_url_tcp_port)
            init_param = {'backend': self.config.dist_backend,
                    'init_method': dist_url,
                    'rank': self.mpi_rank,
                    'world_size': self.mpi_size}
            # always set the device at the very beginning
            torch.cuda.set_device(self.mpi_local_rank)
            logging.info('init param: \n{}'.format(str(init_param)))
            if not dist.is_initialized():
                dist.init_process_group(**init_param)
            # we need to synchronise before exit here so that all workers can
            # finish init_process_group(). If not, worker A might exit the
            # whole program first, but worker B still needs to talk with A. In
            # that case, worker B will never return and will hang there
            synchronize()
        init_random_seed(self.config.random_seed)
        self._initialized = True

    def _setup_logging(self):
        # all ranker outputs the log to a file
        # only rank 0 print the log to console
        log_file = op.join(self.output_folder,
            'log_{}_rank{}.txt'.format(
                datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                self.mpi_rank))
        ensure_directory(op.dirname(log_file))
        file_handle = logging.FileHandler(log_file)
        logger_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s')
        file_handle.setFormatter(fmt=logger_fmt)

        root = logging.getLogger()
        root.handlers = []
        root.setLevel(logging.INFO)
        root.addHandler(file_handle)

        if self.mpi_rank == 0:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logger_fmt)
            root.addHandler(ch)

    def _get_checkpoint_file(self, epoch=None, iteration=None):
        assert(iteration is None)
        if epoch is None:
            epoch = self.config.epochs
        return op.join(self.output_folder, 'snapshot',
                '{}_{:04d}.pth.tar'.format(self.model_prefix, epoch))

    def _get_latest_checkpoint(self):
        all_snapshot = glob.glob(op.join(self.output_folder, 'snapshot',
            '{}_*.pth.tar'.format(self.model_prefix)))
        if len(all_snapshot) == 0:
            return
        snapshot_epochs = [(s, int(op.basename(s)[len(self.model_prefix)+1: len(self.model_prefix)+5])) for s in all_snapshot]
        s, _ = max(snapshot_epochs, key=lambda x: x[1])
        return s


def main(args):
    if isinstance(args, list) or isinstance(args, six.string_types):
        parser = get_arg_parser(model_names)
        args = parser.parse_args(args)


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
    accuracy = get_accuracy_calculator(multi_label=train_dataset.is_multi_label())

    # best_prec1 = 0
    # best_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            load_model_state_dict(model, checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            # best_prec1 = checkpoint['best_prec1']
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # create schedule after resume to properly set start_epoch for learning rate
    scheduler = get_scheduler(optimizer, args)

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
        train_epoch(args, train_loader, model, criterion, optimizer, epoch, accuracy)

        if args.local_rank == 0:
            # evaluate on validation set
            # prec1 = validate(val_loader, model, criterion, logger)

            # remember best prec@1 and save checkpoint
            # is_best = prec1 > best_prec1
            # if is_best:
            #     best_epoch = epoch
            # best_prec1 = max(prec1, best_prec1)
            is_best = False
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'num_classes': train_dataset.label_dim(),
                'multi_label': train_dataset.is_multi_label(),
                'labelmap': train_dataset.get_labelmap(),
                # 'best_prec1': best_prec1,
            }, epoch, op.join(args.output_dir, "snapshot"), is_best)
            # info_str = 'Epoch: [{0}]\t' \
            #            'Best Epoch {1:d}\t' \
            #            'Best Prec1 {2:.3f}'.format(epoch, best_epoch, best_prec1)
            # logger.info(info_str)


if __name__ == '__main__':
    parser = get_arg_parser(model_names)
    # main(parser.parse_args())
    c = ClassifierPipeline(parser.parse_args())
    c.ensure_train()
