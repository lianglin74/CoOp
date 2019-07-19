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
# from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from torchvision.models.resnet import model_urls

from qd_classifier.utils.config import create_config
from qd_classifier.lib import layers
from qd_classifier.utils.data import get_train_data_loader, get_testdata_loader
from qd_classifier.utils.comm import get_dist_url, synchronize, init_random_seed, save_parameters, get_latest_parameter_file
from qd_classifier.utils.train_utils import get_criterion, get_optimizer, get_scheduler, get_accuracy_calculator, train_epoch
from qd_classifier.utils.test import validate
from qd_classifier.utils.save_model import load_model_state_dict, load_from_checkpoint
from qd_classifier.scripts.pred import _predict, _evaluate

from qd.qd_common import get_mpi_local_rank, get_mpi_local_size, get_mpi_rank, get_mpi_size, ensure_directory, ensure_remove_dir
from qd.qd_common import zip_qd, worth_create
from qd.tsv_io import tsv_writer

class ClassifierPipeline(object):
    def __init__(self, config):
        self.model_prefix = "model_epoch"
        self.output_folder = op.join(config.output_dir, config.FULL_EXPID)

        self.config = config

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
        train_dataset, train_loader, train_sampler = get_train_data_loader(self.config, self.mpi_size)
        num_classes = train_dataset.get_num_labels()

        model = self._get_model(num_classes, self.config.pretrained)
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
                    'epoch': epoch,
                    'arch': self.config.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'num_classes': train_dataset.get_num_labels(),
                    'multi_label': train_dataset.is_multi_label(),
                    'labelmap': train_dataset.get_labelmap(),
                }, self._get_checkpoint_file(epoch = epoch))

        return model

    def ensure_predict(self, model_file=None):
        if not self.config.test_data:
            raise ValueError("no test data")

        self._ensure_initialized()
        if not model_file:
            model_file = self._get_checkpoint_file()
        predict_file = self._get_predict_file(model_file, self.config.test_data)
        if self.is_master:
            if not model_file or not op.isfile(model_file):
                logging.info('ignore predict since {} does not exist'.format(
                    model_file))
            elif not worth_create(model_file, predict_file) and not self.config.force_predict:
                logging.info('ignore to do prediction {}'.format(predict_file))
            else:
                model, labelmap = load_from_checkpoint(model_file)
                test_dataloader = get_testdata_loader(self.config, self.mpi_size)
                # NOTE: not support distributed now
                if not self.distributed:
                    model = self._data_parallel_wrap(model)
                _predict(model, predict_file, test_dataloader, labelmap, evaluate=True)

        synchronize()
        return predict_file

    def ensure_evaluate(self, predict_file=None):
        if not self.is_master:
            logging.info('skip evaluation because the rank {} != 0'.format(self.mpi_rank))
            return
        self._ensure_initialized()
        if not predict_file:
            predict_file = self.ensure_predict()
        evaluate_file = self._get_evaluate_file(predict_file)

        eval_dict = _evaluate(predict_file, predict_file + ".det.tsv", evaluate_file)
        return eval_dict

    def monitor_train(self):
        self._ensure_initialized()
        all_step_eval = []
        for step in range(self.config.epochs + 1):
            model_file = self._get_checkpoint_file(epoch=step)
            if op.isfile(model_file):
                predict_file = self.ensure_predict(model_file=model_file)
                eval_dict = self.ensure_evaluate(predict_file=predict_file)
                all_step_eval.append([step, eval_dict])

        if self.is_master:
            tensorboard_folder = op.join(self.output_folder, 'tensorboard_data')
            ensure_remove_dir(tensorboard_folder)
            # writer = SummaryWriter(log_dir=tensorboard_folder)
            tag_prefix = self.config.test_data
            # for step, eval_dict in all_step_eval:
            #     for k in eval_dict:
            #         writer.add_scalar(tag='{}_{}'.format(tag_prefix, k),
            #             scalar_value=eval_dict[k],
            #             global_step=step)
            # writer.close()
            header = ["step"] + [k for k in eval_dict]
            summary = []
            for step, eval_dict in all_step_eval:
                summary.append([step] + [eval_dict[k] for k in header[1:]])
            tsv_writer([header] + summary, op.join(tensorboard_folder, tag_prefix+".eval"))

        synchronize()

    def _get_model(self, num_classes, pretrained=False):
        # create model
        if pretrained:
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

    def _get_predict_file(self, model_file, test_data):
        cc = [model_file, test_data]
        cc.append('predict')
        cc.append('tsv')
        return '.'.join(cc)

    def _get_evaluate_file(self, predict_file):
        cc = [predict_file, "report"]
        return '.'.join(cc)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Classifier training/evaluation pipeline")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        default=None,
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = create_config(args)
    pip = ClassifierPipeline(cfg)
    # pip.ensure_train()
    pip.monitor_train()

if __name__ == '__main__':
    main()
