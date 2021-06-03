from qd.qd_common import execute_func
import qd.data_layer.samplers as samplers
from qd.torch_common import evaluate_topk
from qd.tsv_io import reorder_tsv_keys
from qd.process_tsv import delete_tsv_files
from qd.process_tsv import concat_tsv_files
from qd.qd_common import create_vis_net_file
from qd.torch_common import recursive_to_device
from qd.qd_common import save_parameters
from qd.opt.trainer import do_train_dict
import sys
from tqdm import tqdm
from datetime import datetime
import simplejson as json
from qd.qd_common import ensure_directory
from qd.qd_common import write_to_yaml_file
from qd.qd_common import img_from_base64, load_from_yaml_file
from qd.qd_common import worth_create
from qd.qd_common import read_to_buffer
from qd.qd_common import write_to_file
from qd.tsv_io import load_list_file
from qd.qd_common import get_mpi_rank, get_mpi_size
from qd.qd_common import get_mpi_local_rank, get_mpi_local_size
from qd.qd_common import parse_general_args
from qd.qd_common import plot_to_file
from qd.qd_common import ensure_remove_dir
from qd.process_image import is_pil_image
from collections import OrderedDict
from qd.process_tsv import load_key_rects
from qd.process_tsv import hash_sha1
from qd.tsv_io import tsv_writer, tsv_reader
from qd.tsv_io import TSVFile, CompositeTSVFile
from qd.tsv_io import TSVDataset
from shutil import copyfile
from deprecated import deprecated
import os
import os.path as op
import copy
from pprint import pformat
import logging
import torch
from torch.utils.data import Dataset
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import transforms
import numpy as np
import torchvision
try:
    from itertools import izip as zip
except:
    # python 3
    pass
import time
import re
import glob
from torchvision.transforms import functional as F
import cv2
import math
from qd.qd_common import is_hvd_initialized
import PIL
from qd.torch_common import all_gather_grad_curr
from qd.layers.loss import BCEWithLogitsNegLoss
from qd.data_layer.dataset import TSVSplitProperty
from qd.torch_common import synchronize
from qd.torch_common import ensure_init_process_group
from qd.torch_common import get_master_node_ip
from qd.torch_common import get_aml_mpi_host_names
from qd.torch_common import get_philly_mpi_hosts
from qd.torch_common import torch_save, torch_load
from qd.torch_common import freeze_parameters
from qd.torch_common import init_random_seed
from qd.torch_common import attach_module_name_
from qd.layers.batch_norm import LayerBatchNorm
from qd.layers.loss import MultiHotCrossEntropyLoss
from qd.layers.loss import multi_hot_cross_entropy
from qd.torch_common import concat_all_gather
from qd.data_layer.transform import ImageTransform2Dict
from qd.data_layer.samplers import AttachIterationNumberBatchSampler
from qd.data_layer.transform import ImageCutout
from qd.data_layer.transform import TwoCropsTransform, IoURandomResizedCrop
from qd.data_layer.transform import TwoCropsTransformX
from qd.torch_common import replace_module
from qd.torch_common import InputAsDict
import sys
from datetime import datetime
from pprint import pformat
from qd.qd_common import ensure_directory
from qd.data_layer.dataset import IODataset, DatasetPlusTransform
import json
from qd.qd_common import get_mpi_rank, get_mpi_size
from qd.qd_common import get_mpi_local_rank, get_mpi_local_size
import logging
import os.path as op
import torch
from torch import nn
from qd.qd_pytorch import ModelPipeline, torch_load
from qd.layers.loss import ModelLoss
from qd.qd_common import get_mpi_rank
from qd.tsv_io import tsv_writer
from qd.qd_pytorch import synchronize
import shutil
from qd.qd_common import DummyCfg
import time
from qd.qd_pytorch import replace_module
from qd.layers.group_batch_norm import GroupBatchNorm, get_normalize_groups
from qd.qd_common import json_dump
from qd.qd_common import qd_tqdm as tqdm
from qd.opt.checkpoint import Checkpointer
from qd.qd_common import get_all_path, dict_get_path_value, dict_update_path_value


class Config(object):
    def __init__(self, default, overwrite):
        self.default = default
        self.overwrite = overwrite

    def get(self, k):
        from qd.qd_common import dict_has_path, dict_get_path_value
        if dict_has_path(self.overwrite, k):
            return dict_get_path_value(self.overwrite, k)
        if dict_has_path(self.default, k):
            return dict_get_path_value(self.default, k)

    def __getattr__(self, k):
        return self.get(k)

    def get_dict(self):
        import copy
        default = copy.deepcopy(self.default)
        for p in get_all_path(self.overwrite, with_list=False):
            v = dict_get_path_value(self.overwrite, p)
            dict_update_path_value(default, p, v)
        return default

def get_model_sub_name(i):
    return 'model_iter_{:07d}'.format(i)

class UniPipeline(object):
    def __init__(self, **kwargs):
        self._default = {
            'snapshot_steps': 5000,
            'test_batch_size': 1,
            'effective_batch_size': 8,
            'find_unused_parameters': True,
            'data': 'Unknown',
            'net': 'Unknown',
            'expid': 'Unknown',
            'device': 'cuda',

            'dist_backend': 'nccl',
            'init_method_type': 'tcp',
            'log_step': 100,
            'evaluate_method': 'map',
            'test_split': 'test',
            'num_workers': 8,
            'ovthresh': [-1],
            'step_lr': 30,
            'base_lr': 0.1,
            'max_iter': 10,
            # the default value was 5e-4, which is the default for yolo. We
            # add the default as 5e-4 in yolo_by_mask, and set it 1e-4 for
            # classification.
            'random_seed': 88,
            'apply_nms_gt': True,
            'cudnn_benchmark': False,
            'test_mergebn': False,
            'bgr2rgb': False, # this should be True, but set it False for back compatibility
            'coco_eval_max_det': 100,

            # init
            'dist_url_tcp_port': 12345,

            # data layer
            'train_crop_size': 224,
            'test_crop_size': 224,
            'train_shuffle': True,

            # optimizer
            'momentum': 0.9,
            'weight_decay': 1e-4,

            # lr scheduler
            'scheduler_type': 'cosine',
            'min_rel_lr_in_cosine': 0.,
            'cosine_warmup_factor': 1. / 3,
            'cosine_restart_after_warmup': True,

            'train_transform': 'inception',
            'cosine_warmup_iters': 500,
            'warmup_steps': 0,
            'rms_alpha': 0.99,
            'smooth_label_eps': 0.1,
            'pred_tsv_to_json_extra': 1,
            'mobilenetv3_dropout_ratio': 0.2,
            'cutout_factor': 4,
            'dist_weight': 1.,

            'max_gen_length': 20,

            'splitbysplitsample_buffer_size': 1,
            'splitbysplitsample_group_size': 1,

            'prefetch_factor': 2,

            'ema_start_since': 0.75,
            'ema_step_every': 100,

            'cosine_num_cycle': 1,
            'cosine_gamma_cycle': 1.,
        }
        self.cfg = Config(self._default, kwargs)

        # output folder
        self.full_expid = self.cfg.full_expid or '_'.join(
            map(str, [self.cfg.data, self.cfg.net, self.cfg.expid]))
        self.output_folder = op.join('output', self.full_expid)
        self.model_folder = op.join(self.output_folder, 'snapshot')
        ensure_directory(self.model_folder)

        self.mpi_rank = get_mpi_rank()
        self.mpi_size= get_mpi_size()
        self.mpi_local_rank = get_mpi_local_rank()
        self.mpi_local_size = get_mpi_local_size()

        self.device_id = (self.mpi_local_rank if not self.cfg.debug_train else 0)

        # adapt the batch size based on the mpi_size
        self.is_master = self.mpi_rank == 0

        self._max_iter = None

        self.initialized = False

        if self.cfg.trainer == 'ds':
            # in the initialization of deepspeed, it will try to detect mpi env
            # if these variables are not set. In AML, we may launch the process
            # by another process, e.g. aml_server, in which the mpi auto detect
            # will fail. Thus, we set these env explicitly to bypass mpi env
            # detection.
            os.environ['RANK'] = str(self.mpi_rank)
            os.environ['LOCAL_RANK'] = str(self.mpi_local_rank)
            os.environ['WORLD_SIZE'] = str(self.mpi_size)
            os.environ['MASTER_ADDR'] = get_master_node_ip()
            os.environ['MASTER_PORT'] = str(self.cfg.dist_url_tcp_port)

    @property
    def max_iter(self):
        if self._max_iter is None:
            self._max_iter = self.parse_iter(self.cfg.max_iter)
        return self._max_iter

    def get_len_dataset(self, is_train):
        raise NotImplementedError('defined in sub class')

    def get_transform(self, is_train):
        raise NotImplementedError('defined in sub classes')

    def get_raw_model(self, is_train):
        raise NotImplementedError('sub class to implement')

    def predict_output_to_tsv_row(self, data, output):
        raise NotImplementedError('sub class to implement')

    def get_collate_fn(self, is_train):
        return None

    def append_predict_param(self, cc):
        if self.cfg.test_normalize_module:
            cc.append('NormBy{}'.format(self.cfg.test_normalize_module))
        if self.cfg.predict_extract:
            s = self.predict_extract
            if isinstance(self.predict_extract, list):
                s = '.'.join(self.predict_extract)
            cc.append('Extract{}'.format(s))
        if self.cfg.test_crop_position:
            cc.append(self.test_crop_position)
        if self.cfg.test_resize_size and self.cfg.test_resize_size != 224:
            cc.append('r{}'.format(self.cfg.test_resize_size))
        if self.cfg.predict_ema_decay:
            cc.append('ema{}'.format(self.cfg.predict_ema_decay))
        if self.cfg.pt_swa:
            cc.append('swa')
        if self.cfg.test_max_iter is not None:
            # this is used for speed test
            if self.test_mergebn:
                cc.append('mergebn')
            cc.append('max_iter{}'.format(self.test_max_iter))
            # we explicitly log the batch size here so that we can make sure it
            # is 1 or batch processing
            cc.append('BS{}'.format(self.test_batch_size))
            cc.append(self.device)
            if self.device == 'cpu' and self.cpu_num_threads:
                torch.set_num_threads(self.cpu_num_threads)
                cc.append('thread{}'.format(self.cpu_num_threads))
        if self.cfg.flush_denormal and self.device == 'cpu':
            # gpu is not supported
            r = torch.set_flush_denormal(True)
            assert r, 'not supported'
            cc.append('flush_denormal')
        if self.cfg.pred_file_hint is not None:
            cc.append(self.pred_file_hint)
        if self.cfg.test_crop_size != 224 and self.cfg.test_crop_size:
            cc.append('crop{}'.format(self.cfg.test_crop_size))

        # in vision-laugnage
        if self.cfg.max_gen_length != 20:
            cc.append('max_token{}'.format(self.cfg.max_gen_length))

        if self.cfg.test_respect_ratio_max is not None:
            cc.append('testMax{}'.format(self.cfg.test_respect_ratio_max))

        if self.cfg.crop_pct:
            cc.append('crpPct{}'.format(self.cfg.crop_pct))

    def get_model(self, is_train):
        model = self.get_raw_model(is_train)
        model = self.model_surgery(model, is_train)
        return model

    def model_surgery(self, model, is_train):
        convert_bn = self.cfg.convert_bn
        if convert_bn == 'L1':
            raise NotImplementedError
        elif convert_bn == 'L2':
            raise NotImplementedError
        elif convert_bn == 'GN':
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: torch.nn.GroupNorm(32, m.num_features),
                    )
        elif convert_bn == 'LNG': # layer norm by group norm
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: torch.nn.GroupNorm(1, m.num_features))
        elif convert_bn == 'ING': # Instance Norm by group norm
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: torch.nn.GroupNorm(m.num_features, m.num_features))
        elif convert_bn == 'GBN':
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: GroupBatchNorm(get_normalize_groups(
                        m.num_features, self.normalization_group,
                        self.normalization_group_size), m.num_features))
        elif convert_bn == 'SBN':
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d) or
                        isinstance(m, torch.nn.BatchNorm1d),
                    lambda m: torch.nn.SyncBatchNorm(m.num_features,
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=m.affine,
                        track_running_stats=m.track_running_stats))
            from qd.layers.batch_norm import FrozenBatchNorm2d
            model = replace_module(model,
                    lambda m: isinstance(m, FrozenBatchNorm2d),
                    lambda m: torch.nn.SyncBatchNorm(m.num_features,
                        eps=m.eps))
        elif convert_bn == 'FBN': # frozen batch norm
            def set_eval_return(m):
                m.eval()
                return m
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d) or
                                   isinstance(m, torch.nn.BatchNorm1d),
                    lambda m: set_eval_return(m))
        #elif convert_bn == 'NSBN':
            #if self.distributed:
                #from qd.layers.batch_norm import NaiveSyncBatchNorm
                #model = replace_module(model,
                        #lambda m: isinstance(m, torch.nn.BatchNorm2d),
                        #lambda m: NaiveSyncBatchNorm(m.num_features,
                            #eps=m.eps,
                            #momentum=m.momentum,
                            #affine=m.affine,
                            #track_running_stats=m.track_running_stats))
        elif convert_bn == 'CBN':
            from qd.layers.batch_norm import ConvergingBatchNorm
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: ConvergingBatchNorm(
                        policy=self.cbn_policy,
                        max_iter=self.max_iter,
                        gamma=self.cbn_gamma,
                        num_features=m.num_features,
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=True,
                        track_running_stats=m.track_running_stats,
                        ))
        else:
            assert convert_bn is None, convert_bn
        if self.cfg.convert_ln == 'LBN':
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.LayerNorm),
                    lambda m: LayerBatchNorm(
                        m.normalized_shape[0],
                        eps=m.eps,
                        affine=m.elementwise_affine
                        ))
        elif self.cfg.convert_ln == 'LBN_noE':
            attach_module_name_(model)
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.LayerNorm) and
                                   not m.name_from_root.endswith('bert.embeddings.LayerNorm'),
                    lambda m: LayerBatchNorm(
                        m.normalized_shape[0],
                        eps=m.eps,
                        affine=m.elementwise_affine
                        ))

        if self.cfg.fc_as_mlp:
            # this is used, normally for self-supervised learning scenarios
            from qd.torch_common import replace_fc_with_mlp_
            replace_fc_with_mlp_(model)
        if self.cfg.hswish2relu6:
            from qd.layers.mitorch_models.modules.activation import HardSwish
            model = replace_module(model,
                    lambda m: isinstance(m,
                                         HardSwish),
                    lambda m: torch.nn.ReLU6(inplace=True))
        if self.cfg.vis_adaptive_global_pool:
            from qd.layers.adapt_avg_pool2d import VisAdaptiveAvgPool2d
            model = replace_module(model,
                    lambda m: isinstance(m,
                                         nn.AdaptiveAvgPool2d),
                    lambda m: VisAdaptiveAvgPool2d())
        if self.cfg.freeze_bn:
            from qd.torch_common import freeze_bn_
            freeze_bn_(model)
        if self.cfg.standarize_conv2d:
            from qd.layers.standarized_conv import convert_conv2d_to_standarized_conv2d
            model = convert_conv2d_to_standarized_conv2d(model)
        if self.cfg.bn_momentum:
            from qd.torch_common import update_bn_momentum
            update_bn_momentum(model, self.cfg.bn_momentum)
        if self.cfg.c_bias_sigmoid_small is not None:
            from qd.torch_common import query_modules_by_name
            modules = query_modules_by_name(model, '.fc')
            assert len(modules) == 1
            fc = modules[0]
            if isinstance(self.cfg.c_bias_sigmoid_small, str):
                assert self.cfg.c_bias_sigmoid_small == 'C'
                self.c_bias_sigmoid_small = 1. / len(self.get_labelmap())
                logging.info('overwrite self.c_bias_sigmoid_small'
                             'to {}'.format(self.c_bias_sigmoid_small))
            nn.init.constant_(fc.bias, -math.log(1. / self.c_bias_sigmoid_small - 1))
        # assign a name to each module so that we can use it in each module to
        # print debug information
        attach_module_name_(model)
        if is_train:
            if self.cfg.device == 'cuda':
                if self.cfg.trainer == 'pl':
                    model = model.cuda()
                elif self.cfg.trainer == 'ds':
                    # in deepspeed, it will use apex amp rather than
                    # built-in amp. In apex amp, we should not wrap it with
                    # DistributedDataParallel
                    model = model.cuda()
                else:
                    assert self.cfg.trainer in [None, 'pre']
                    model = self.data_parallel_wrap(model)
        else:
            model.eval()
        return model

    def parse_iter(self, i):
        def to_iter(e):
            if type(e) is str and e.endswith('e'):
                num_train_images = len(self.get_len_dataset(is_train=True))
                iter_each_epoch = 1. * num_train_images / self.cfg.effective_batch_size
                return int(float(e[:-1]) * iter_each_epoch)
            elif isinstance(e, float) and e < 1:
                return int(self.max_iter * e)
            else:
                return int(e)
        return to_iter(i)

    def get_dataset(self, is_train):
        len_dataset = self.get_len_dataset(is_train)
        trans = self.get_transform(is_train)
        dataset = DatasetPlusTransform(len_dataset, trans)
        return dataset

    def get_sampler(self, is_train, dataset):
        #elif stage == 'train' and self.composite_rank_aware_sampler:
        if is_train:
            length_divisible = self.cfg.effective_batch_size // self.mpi_size
        else:
            length_divisible = 1

        if is_train and self.cfg.sampler_type == 'splitBysplit':
            sampler = samplers.SplitBySplitSampler(
                dataset,
                shuffle=self.cfg.train_shuffle,
                random_seed=self.cfg.random_seed,
                group_size=self.cfg.splitbysplitsample_group_size,
                prepare_t_versions=self.get_splitbysplit_sampler_prepare_t_versions(),
                disable_prepare=self.cfg.disable_splitbysplit_prepare,
            )
        elif is_train and self.cfg.sampler_type == 'ranksplit':
            from qd.data_layer.samplers import RankSplitSampler
            sampler = RankSplitSampler(dataset, shuffle=self.cfg.train_shuffle,
                                       random_seed=self.cfg.random_seed)
        elif is_train and self.cfg.sampler_type == 'nodesplit':
            from qd.data_layer.samplers import NodeSplitSampler
            sampler = NodeSplitSampler(dataset, shuffle=self.cfg.train_shuffle,
                                       random_seed=self.cfg.random_seed)
        else:
            sampler = samplers.DistributedSampler(
                dataset,
                shuffle=self.cfg.train_shuffle if is_train else False,
                length_divisible=length_divisible)
        return sampler

    def get_splitbysplit_sampler_prepare_t_versions(self):
        return [(None, None)]

    def get_batch_sampler(self, is_train, sampler, start_iter):
        bs = (self.cfg.effective_batch_size // self.mpi_size if is_train else
            self.cfg.test_batch_size)
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler,
            bs,
            drop_last=False,
        )
        if self.cfg.attach_iter_in_sampler:
            batch_sampler = AttachIterationNumberBatchSampler(
                batch_sampler, 0, self.max_iter)
        else:
            logging.info('recommended to set attach_iter_in_sampler as True')
        if is_train:
            batch_sampler = samplers.IterationBasedBatchSampler(
                batch_sampler, self.max_iter, start_iter
            )
        return batch_sampler

    def get_data_loader(self, is_train, start_iter):
        dataset = self.get_dataset(is_train)
        sampler = self.get_sampler(is_train, dataset)
        logging.info('sampler = {}'.format(sampler))
        batch_sampler = self.get_batch_sampler(is_train, sampler, start_iter)
        collate_fn = None
        collate_fn = self.get_collate_fn(is_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            prefetch_factor=self.cfg.prefetch_factor if self.cfg.num_workers > 0 else 2,
        )
        return loader

    def ensure_train(self):
        self._setup_logging()
        self._ensure_initialized(
            init_ddp=(self.cfg.trainer not in ['pl', 'ds']),
            init_ds=(self.cfg.trainer == 'ds')
        )

        last_model_file = self.get_checkpoint_file()
        logging.info('last model file = {}'.format(last_model_file))
        if op.isfile(last_model_file) and not self.cfg.force_train:
            logging.info('skip to train')
            return

        if self.mpi_rank == 0:
            save_parameters(self.cfg.overwrite, self.output_folder)

        logging.info(pformat(self.cfg.get_dict()))
        from qd.torch_common import get_torch_version_info
        logging.info('torch info = {}'.format(
            pformat(get_torch_version_info())))

        synchronize()

        train_result = self.train()

        if self.mpi_rank == 0 and not self.cfg.debug_train:
            # save the code after training
            from qd.qd_common import zip_qd, try_delete
            # we'd better to delete it since it seems like zip will read/write
            # if there is
            source_code = op.join(self.output_folder, 'source_code.zip')
            if op.isfile(source_code):
                try_delete(source_code)
            zip_qd(op.join(self.output_folder, 'source_code'))

        synchronize()

        return train_result

    def _setup_logging(self):
        # all ranker outputs the log to a file
        # only rank 0 print the log to console
        log_file = op.join(self.output_folder,
            'log_{}_rank{}.txt'.format(
                datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                self.mpi_rank))
        ensure_directory(op.dirname(log_file))
        file_handle = logging.FileHandler(log_file)
        format_str = '%(asctime)s.%(msecs)03d {}:%(process)d:%(filename)s:%(lineno)s %(funcName)10s(): %(message)s'.format(
            self.mpi_rank,
        )
        logger_fmt = logging.Formatter(format_str)
        file_handle.setFormatter(fmt=logger_fmt)

        root = logging.getLogger()
        root.handlers = []
        root.setLevel(logging.INFO)
        root.addHandler(file_handle)

        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logger_fmt)
        root.addHandler(ch)

    def get_optimizer(self, model):
        parameters = get_parameter_groups_general(self, model)

        if self.cfg.optimizer_type in [None, 'SGD', 'LARS']:
            from qd.opt.sgd import SGDVerbose
            optimizer = SGDVerbose(parameters,
                                   self.cfg.base_lr,
                                   momentum=self.cfg.momentum,
                                   # this is default decay, and will be
                                   # overwritten if we specified it in
                                   # parameters.
                                   weight_decay=self.cfg.weight_decay,
                                   nesterov=self.cfg.sgd_nesterov,
                                   )
        elif self.cfg.optimizer_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(
                parameters,
                self.cfg.base_lr,
                momentum=self.cfg.momentum,
                alpha=self.cfg.rms_alpha,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer_type in ['Adam']:
            optimizer = torch.optim.Adam(
                parameters,
                self.cfg.base_lr,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer_type in ['AdamW']:
            optimizer = torch.optim.AdamW(
                parameters,
                self.cfg.base_lr,
                weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer_type in ['MAdamW']:
            from qd.mask.solver import AdamW
            optimizer = AdamW(parameters,
                              lr=self.cfg.base_lr,
                              eps=1e-8)
        else:
            raise NotImplementedError(self.cfg.optimizer_type)
        if self.cfg.optimizer_type in ['LARS']:
            from torchlars import LARS
            optimizer = LARS(optimizer=optimizer)
        if self.cfg.ema_optimizer:
            from qd.opt.ema_optimizer import EMAOptimizer
            optimizer = EMAOptimizer(
                optimizer=optimizer,
                start_since=self.parse_iter(self.cfg.ema_start_since),
                step_every=self.parse_iter(self.cfg.ema_step_every)
            )
        return optimizer

    def get_lr_scheduler(self, optimizer):
        scheduler_type = self.cfg.scheduler_type
        if scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.parse_iter(self.cfg.step_lr),
            )
        elif scheduler_type == 'multi_step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[self.parse_iter(i) for i in self.cfg.stageiter],
                gamma=0.1,
            )
        elif scheduler_type == 'cosine':
            from qd.opt.WarmupCosineAnnealingLR import WarmupCosineAnnealingLR
            assert isinstance(self.max_iter, int)
            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                max_iter=self.max_iter,
                min_lr=self.cfg.min_rel_lr_in_cosine * self.cfg.base_lr,
                warmup_factor=self.cfg.cosine_warmup_factor,
                warmup_iters=self.parse_iter(self.cfg.cosine_warmup_iters),
                cosine_restart_after_warmup=self.cfg.cosine_restart_after_warmup
            )
        elif scheduler_type == 'Cos':
            from qd.opt.cosine_annearing_with_warmup import CosineAnnealingWarmupRestarts
            num_cycle = self.cfg.cosine_num_cycle
            first_cycle_steps = (self.max_iter + num_cycle - 1) // num_cycle
            if isinstance(self.cfg.warmup_steps, float) and  self.cfg.warmup_steps < 1:
                warmup_steps = int(self.cfg.warmup_steps*first_cycle_steps)
            else:
                warmup_steps = self.parse_iter(self.cfg.cosine_warmup_iters)
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=first_cycle_steps,
                cycle_mult=1.,
                max_lr=self.cfg.base_lr,
                min_lr=self.cfg.min_rel_lr_in_cosine * self.cfg.base_lr,
                warmup_steps=warmup_steps,
                gamma=self.cfg.cosine_gamma_cycle,
            )
        elif scheduler_type == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_iter,
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            assert isinstance(self.max_iter, int)
            patience = 3 * self.max_iter // self.cfg.effective_batch_size
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=patience, verbose=True)
        elif scheduler_type == "linear":
            from qd.mask.solver import WarmupLinearSchedule
            scheduler = WarmupLinearSchedule(
                optimizer,
                warmup_steps=self.parse_iter(self.cfg.warmup_steps),
                t_total=self.max_iter,
            )
        elif scheduler_type == 'linear_swa':
            from qd.opt.scheduler import WarmupLinearSWASchedule
            scheduler = WarmupLinearSWASchedule(
                optimizer,
                warmup_steps=self.parse_iter(self.cfg.warmup_steps),
                swa_start_steps=self.parse_iter(0.75),
                t_total=self.max_iter,
                swa_const_ratio=0.05,
            )
        else:
            raise NotImplementedError(scheduler_type)
        return scheduler

    def data_parallel_wrap(self, model):
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.device_id],
            # used for effiicient-net + faster-rcnn
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def create_checkpointer(self, model, optimizer, scheduler):
        save_to_disk = get_mpi_rank() == 0
        checkpointer = Checkpointer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=op.join(self.output_folder, 'snapshot'),
            save_to_disk=save_to_disk,
            suffix='pt',
        )
        return checkpointer

    #def fuse_cache(self):
        #if int(os.environ.get('QD_TSV_USE_FUSE', '0')):
            ## it is ok to run the following at the same time as we will handle concurrency in ensure_cache
            #files = self.get_fuse_cache_files()
            #from qd.cloud_storage import create_cloud_fuse
            #fuser = create_cloud_fuse()
            #fuser.ensure_cache(files)
            #synchronize()

    #def get_fuse_cache_files(self):
        #return []

    def train(self):
        #self.fuse_cache()

        model = self.get_model(is_train=True)
        optimizer = self.get_optimizer(model)
        logging.info(optimizer)

        scheduler = self.get_lr_scheduler(optimizer)

        if self.cfg.trainer == 'ds':
            config = self.get_deep_speed_config()
            import deepspeed
            model_engine, optimizer, _, scheduler = deepspeed.initialize(
                config_params=config,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
            )
        else:
            model_engine = None

        checkpointer = self.create_checkpointer(
            model,
            optimizer,
            scheduler,
        )

        # load basemodel or last pre-trained snapshot
        extra_param = checkpointer.recover_or_load(
            self.cfg.basemodel, model_only=True)
        start_iter = extra_param.get('iteration', 0)

        logging.info(scheduler)

        # use the maskrcnn trainer engine
        train_loader = self.get_data_loader(
            is_train=True,
            start_iter=start_iter,
        )

        self.do_train(train_loader, model, optimizer, scheduler, checkpointer,
                      start_iter, model_engine=model_engine)

        return checkpointer.get_checkpoint_file()

    def get_deep_speed_config(self):
        config_params = {
            'train_batch_size': self.cfg.effective_batch_size,
        }

        use_amp = self.cfg.use_amp
        use_fp16 = self.cfg.use_fp16
        assert not (use_amp and use_fp16)

        if use_amp:
            config_params['amp'] = {
                'enabled': True,
                'opt_level': 'O1',
            }

        if use_fp16:
            config_params['fp16'] = {
                'enabled': True,
            }

        gradient_clip = self.cfg.gradient_clip
        if gradient_clip:
            config_params['gradient_clipping'] = gradient_clip

        config_params['flops_profiler'] = {
            'enabled': True,
            'profile_step': 1,
            'module_depth': -1,
            'top_modules': 3,
            'detailed': True,
        }

        config_params['logging'] = {
            'steps_per_print': self.cfg.log_step,
        }
        if self.cfg.zero_opt_stage is not None:
            config_params['zero_optimization'] = {
                'stage': self.cfg.zero_opt_stage,
            }
            if self.cfg.zero_opt_stage > 0:
                config_params['fp16'] = {
                    'enabled': True
                }
            config_params['zero_allow_untested_optimizer'] = True
        logging.info(pformat(config_params))
        return config_params

    def do_train(self, loader, model, optimizer, scheduler, checkpointer,
                 start_iter, model_engine=None):
        device = torch.device(self.cfg.device)
        logging.info(model)
        training_modules = [n for n, m in model.named_modules() if m.training]
        eval_modules = [n for n, m in model.named_modules() if not m.training]
        logging.info('training module: {}'.format(pformat(training_modules)))
        logging.info('eval module: {}'.format(pformat(eval_modules)))
        logging.info('dataset = \n{}'.format(loader.dataset))

        if self.cfg.trainer == 'pl':
            from qd.layers.lightning_wrapper import LightningModule
            model = LightningModule(model, optimizer, scheduler)
            args = {}
            args['max_epochs'] = 1
            args['max_steps'] = self.max_iter
            if self.cfg.use_amp:
                args['precision'] = 16
            if self.cfg.gradient_clip is not None:
                args['gradient_clip_val'] = self.cfg.gradient_clip
            args['replace_sampler_ddp'] = False
            # do not use ddp
            args['accelerator'] = 'horovod'
            # 1 does not mean 1 gpu, but mean yes, to use gpu
            args['gpus'] = 1
            args['prepare_data_per_node'] = False
            args['log_every_n_steps'] = self.cfg.log_step
            args['val_check_interval'] = 1

            import pytorch_lightning as pl
            logging.info('pl args = {}'.format(pformat(args)))
            trainer = pl.Trainer(**args)
            trainer.fit(model, loader)
            # save the result
            checkpointer.save(get_model_sub_name(self.max_iter))
        elif self.cfg.trainer == 'ds':
            from qd.opt.trainer import do_train_by_deepspeed
            do_train_by_deepspeed(
                model=model,
                data_loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpointer=checkpointer,
                device=device,
                checkpoint_period=self.cfg.snapshot_steps,
                arguments={'iteration': start_iter},
                log_step=self.cfg.log_step,
                use_amp=self.cfg.use_amp,
                gradient_clip=self.cfg.gradient_clip,
                model_sub_name_fn=get_model_sub_name,
                zero_opt_stage=self.cfg.zero_opt_stage,
                use_fp16=self.cfg.use_fp16,
                async_loader=self.cfg.async_dataloader,
                no_flops_profiler=self.cfg.no_flops_profiler,
                model_engine=model_engine,
            )
        else:
            extra_param = {}
            if self.cfg.pt_swa:
                extra_param['pt_swa_lr'] = self.cfg.base_lr*self.cfg.pt_swa_lr_mult
            do_train_dict(
                model=model,
                data_loader=loader,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpointer=checkpointer,
                device=device,
                checkpoint_period=self.cfg.snapshot_steps,
                arguments={'iteration': start_iter},
                log_step=self.cfg.log_step,
                use_amp=self.cfg.use_amp,
                gradient_clip=self.cfg.gradient_clip,
                model_sub_name_fn=get_model_sub_name,
                async_loader=self.cfg.async_dataloader,
                pt_swa=self.cfg.pt_swa,
                **extra_param,
            )

    #def demo(self, image_path):
        #from qd.process_image import load_image
        #cv_im = load_image(image_path)
        #self.predict_one(cv_im)

    #def _get_load_model_demo(self):
        #if self.model is None:
            #model = self.get_test_model()
            #model_file = self._get_checkpoint_file()
            #self.load_test_model(model, model_file)
            #model = model.to(self.device)
            #self.model = model
            #model.eval()
        #return self.model

    #def predict_one(self, cv_im):
        #model = self._get_load_model_demo()
        #softmax_func = self._get_test_normalize_module()
        #from qd.layers import ForwardPassTimeChecker
        #model = ForwardPassTimeChecker(model)
        #transform = self.get_transform('test')
        #im = transform(cv_im)
        #im = im[None, :]
        #im = im.to(self.device)
        #output = model(im)
        #if softmax_func is not None:
            #output = softmax_func(output)
        #all_tops, all_top_indexes = output.topk(5, dim=1,
                #largest=True, sorted=False)

        #tops, top_indexes = all_tops[0], all_top_indexes[0]
        #labelmap = self.get_labelmap()
        #all_tag = [{'class': labelmap[i], 'conf': float(t)} for t, i in
                #zip(tops, top_indexes)]
        #return all_tag

    def get_checkpoint_file(self, iteration=None):
        if iteration is None and self.cfg.model_file is not None:
            return self.cfg.model_file
        if iteration is None:
            iteration = self.max_iter
        iteration = self.parse_iter(iteration)
        return op.join(
            self.model_folder,
            get_model_sub_name(iteration) + '.pt')

    def init_ddp(self):
        ensure_init_process_group(
            device_id=self.device_id,
            port=self.cfg.dist_url_tcp_port,
        )

    def init_ds(self):
        import deepspeed
        deepspeed.init_distributed(distributed_port=self.cfg.dist_url_tcp_port)

    def _ensure_initialized(self, init_ddp=True, init_ds=False):
        if self.initialized:
            return

        if self.cfg.file_system_sharing:
            logging.info('using file system for tensor sharing')
            torch.multiprocessing.set_sharing_strategy('file_system')

        if self.cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        torch.cuda.set_device(self.device_id)

        if init_ddp:
            self.init_ddp()

        if init_ds:
            self.init_ds()

        # sometimes, the init hangs, and thus we print some logs for
        # verification
        logging.info('initialized')
        # we need to synchronise before exit here so that all workers can
        # finish init_process_group(). If not, worker A might exit the
        # whole program first, but worker B still needs to talk with A. In
        # that case, worker B will never return and will hang there
        synchronize()
        init_random_seed(self.cfg.random_seed)
        self.initialized = True

    def get_predict_file(self, model_file=None):
        if model_file is None:
            model_file = self.get_checkpoint_file(iteration=self.max_iter)
        cc = [model_file, self.cfg.test_data, self.cfg.test_split]
        self.append_predict_param(cc)
        cc.append('predict')
        cc.append('tsv')
        return '.'.join(cc)

    def ensure_predict(self, model_file=None):
        if self.cfg.ignore_predict:
            logging.info('ignore to predict as instructed')
            return

        # deprecate epoch and iteration. use model_file, gradually
        self._ensure_initialized()
        #if iteration is not None:
            #assert model_file is None
            #logging.warn('use model_file rather than epoch or iteration, pls')
            #model_file = self.get_checkpoint_file(iteration=iteration)
        #else:
        if model_file is None:
            model_file = self.get_checkpoint_file()
            #assert model_file is not None
        predict_result_file = self.get_predict_file(model_file)
        if not op.isfile(model_file) and not op.isdir(model_file):
            logging.info('ignore to run predict since {} does not exist'.format(
                model_file))
            return predict_result_file
        if not worth_create(model_file, predict_result_file) and not self.cfg.force_predict:
            logging.info('ignore to do prediction {}'.format(predict_result_file))
            return predict_result_file

        self.predict(model_file, predict_result_file)

        return predict_result_file

    def load_test_model(self, model, model_file):
        if self.cfg.predict_ema_decay:
            out_model_file = op.splitext(model_file)[0] + '.ema{}.pt'.format(
                self.cfg.predict_ema_decay)
            if self.mpi_rank == 0 and not op.isfile(out_model_file):
                param = torch_load(model_file)
                from qd.opt.ema_optimizer import replace_ema_param
                replace_ema_param(param, decay=self.cfg.predict_ema_decay)
                torch_save(param, out_model_file)
            synchronize()
            model_file = out_model_file
        if self.cfg.pt_swa:
            out_model_file = op.splitext(model_file)[0] + '.swa.pt'
            if self.mpi_rank == 0 and not op.isfile(out_model_file):
                param = torch_load(model_file)
                param['model'] = param['swa_model']
                torch_save(param, out_model_file)
            synchronize()
            model_file = out_model_file
        checkpointer = Checkpointer(
            model=model,
            save_dir=self.output_folder,
        )
        checkpointer.load(model_file, load_if_has=False)

    #def wrap_feature_extract(self, model):
        #from qd.layers.feature_extract import FeatureExtract
        #model = FeatureExtract(model, self.predict_extract)
        #return model

    def get_rank_specific_tsv(self, f, rank):
        return '{}_{}_{}.tsv'.format(f, rank, self.mpi_size)

    def predict_iter(self, dataloader, model, meters):
        start = time.time()
        logging.info(dataloader.dataset)
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            if self.cfg.test_max_iter is not None and i >= self.cfg.test_max_iter:
                # this is used for speed test, where we only would like to run a
                # few images
                break
            meters.update(data=time.time() - start)
            start = time.time()
            data = recursive_to_device(data, self.cfg.device)
            meters.update(input_to_cuda=time.time() - start)
            start = time.time()
            output = self.predict_iter_forward(model, data)
            meters.update(model=time.time() - start)
            start = time.time()
            for row in self.predict_output_to_tsv_row(data, output):
                yield row
            if self.cfg.debug_feature:
                model.sumarize_feature()
            meters.update(write=time.time() - start)
            start = time.time()

    def predict_iter_forward(self, model, inputs):
        with torch.no_grad():
            return model(inputs)

    def feature_to_tsv_row(self, features, feature_names, keys):
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        if isinstance(keys, dict):
            keys = keys['key']
        for i, key in enumerate(keys):
            info = []
            for f, f_name in zip(features, feature_names):
                info.append({'feature': f[i].tolist(), 'name': f_name})
            yield key, json_dump(info)

    def post_load_model_surgery(self, model, model_file):
        if self.cfg.test_mergebn:
            from qd.layers import MergeBatchNorm
            model = MergeBatchNorm(model)
            logging.info('after merging bn = {}'.format(model))
        from qd.layers import ForwardPassTimeChecker
        # we need to first convert it to, e.g. cuda, especially for
        # pytorch-ligntning module. otherwise model.device will not be updated
        # if we convert it on top of ForwardPassTimeChecker
        model = model.to(self.cfg.device)
        model = ForwardPassTimeChecker(model)
        model.eval()
        #if self.predict_extract:
            #model = self.wrap_feature_extract(model)
        #if self.debug_feature:
            #from qd.layers.forward_pass_feature_cache import ForwardPassFeatureCache
            #model = ForwardPassFeatureCache(model)
        return model

    def is_train_finished(self):
        last_model = self.get_checkpoint_file()
        if not op.isfile(last_model) and \
                not op.islink(last_model) and \
                not op.isdir(last_model):
            logging.info('{} is not a file and not a folder'.format(
                last_model
            ))
            return False
        return True

    def predict(self, model_file, predict_result_file):
        if self.mpi_size > 1:
            sub_predict_file = self.get_rank_specific_tsv(predict_result_file,
                    self.mpi_rank)
        else:
            sub_predict_file = predict_result_file

        model = self.get_model(is_train=False)
        self.load_test_model(model, model_file)
        model = self.post_load_model_surgery(model, model_file)
        dataloader = self.get_data_loader(is_train=False, start_iter=0)

        #from maskrcnn_benchmark.utils.metric_logger import MetricLogger
        from qd.logger import MetricLogger
        meters = MetricLogger(delimiter="  ")
        logging.info('writing {}'.format(sub_predict_file))
        tsv_writer(self.predict_iter(dataloader, model, meters),
                   sub_predict_file)

        speed_yaml = sub_predict_file + '.speed.yaml'
        write_to_yaml_file(model.get_time_info(), speed_yaml)
        create_vis_net_file(speed_yaml,
                op.splitext(speed_yaml)[0] + '.vis.txt')
        logging.info(str(meters))

        if self.mpi_rank == 0:
            info_file = predict_result_file + '.info.yaml'
            write_to_yaml_file(self.cfg.overwrite, info_file)

        # we need to sync before merging all to make sure each rank finish its
        # own task
        synchronize()
        if self.mpi_size > 1 and get_mpi_rank() == 0:
            cache_files = [self.get_rank_specific_tsv(predict_result_file, i)
                for i in range(self.mpi_size)]
            before_reorder = predict_result_file + '.before.reorder.tsv'
            concat_tsv_files(cache_files, before_reorder)
            # in distributed testing, some images might be predicted by
            # more than one worker since the distributed sampler only
            # garrantee each image will be processed at least once, not
            # exactly once. Thus, we have to remove the duplicate
            # predictions.
            ordered_keys = dataloader.dataset.get_keys()
            reorder_tsv_keys(before_reorder, ordered_keys, predict_result_file)

            delete_tsv_files(cache_files)
            delete_tsv_files([before_reorder])

            # during prediction, we also computed the time cost. Here we
            # merge the time cost
            speed_cache_files = [c + '.speed.yaml' for c in cache_files]
            speed_yaml = predict_result_file + '.speed.yaml'
            from qd.qd_common import merge_speed_info
            merge_speed_info(speed_cache_files, speed_yaml)
            from qd.qd_common import try_delete
            for x in speed_cache_files:
                try_delete(x)
            vis_files = [op.splitext(c)[0] + '.vis.txt' for c in speed_cache_files]
            from qd.qd_common import merge_speed_vis
            merge_speed_vis(vis_files,
                    op.splitext(speed_yaml)[0] + '.vis.txt')
            for x in vis_files:
                try_delete(x)

        synchronize()
        return predict_result_file

    def get_evaluate_file(self, predict_file=None):
        if predict_file is None:
            predict_file = self.get_predict_file()
        assert predict_file.endswith('.tsv')
        cc = [op.splitext(predict_file)[0]]
        if self.cfg.evaluate_method != 'map':
            if self.cfg.evaluate_method is None:
                return
            cc.append(self.cfg.evaluate_method)
        if self.cfg.evaluate_method == 'neg_aware_gmap':
            if not self.apply_nms_gt:
                cc.append('noNMSGt')
            if not self.apply_nms_det:
                cc.append('noNMSDet')
            if not self.expand_label_det:
                cc.append('noExpandDet')
        if self.cfg.test_version:
            if self.cfg.test_version == -1:
                latest_version = TSVDataset(self.test_data).get_latest_version(
                        self.test_split, 'label')
                self.test_version = latest_version
                logging.info('inferred the latest version is {}'.format(
                    latest_version))
            cc.append('v{}'.format(self.cfg.test_version))
        if self.cfg.coco_eval_max_det is not None and self.cfg.coco_eval_max_det != 100:
            cc.append('MaxDet{}'.format(self.coco_eval_max_det))
        if self.cfg.pred_tsv_to_json_extra != 1 and \
                self.cfg.evaluate_method == 'coco_box':
            cc.append('{}'.format(self.pred_tsv_to_json_extra))
        cc.append('report')
        return '.'.join(cc)

    def ensure_evaluate(self, predict_file=None):
        if self.mpi_rank != 0:
            logging.info('skip because the rank {} != 0'.format(self.mpi_rank))
            return

        # if prediction is disabled, we will not proceed either.
        if self.cfg.ignore_evaluate or self.cfg.ignore_predict:
            logging.info('ignore evaluate as instructed')
            return

        # not other rank will exit and initalizing distributed will not go
        # through. No need to run initilaization here actually.
        #self._ensure_initialized()
        if not predict_file:
            model_file = self.get_checkpoint_file()
            predict_file = self.get_predict_file(model_file)
        evaluate_file = self.get_evaluate_file(predict_file)
        if evaluate_file is None:
            return
        if not worth_create(predict_file, evaluate_file) and not self.cfg.force_evaluate:
            logging.info('ignore {}'.format(evaluate_file))
        else:
            self.evaluate(predict_file, evaluate_file)

        # create index
        self.ensure_create_evaluate_meta_file(evaluate_file)
        return evaluate_file

    def evaluate(self, predict_file, evaluate_file):
        dataset = TSVDataset(self.cfg.test_data)

        if self.cfg.evaluate_method == 'map':
            from qd.deteval import deteval_iter
            other_param = copy.deepcopy(self.cfg.overwrite)
            if 'ovthresh' in other_param:
                del other_param['ovthresh']
            deteval_iter(
                    dataset.iter_data(self.cfg.test_split, 'label',
                        version=self.cfg.test_version),
                    predict_file,
                    report_file=evaluate_file,
                    ovthresh=self.cfg.ovthresh, # this is in self.kwargs already
                    **other_param)
        elif self.cfg.evaluate_method == 'attr':
            # only for visualgenome
            def gen_rows():
                for key, str_rects in tsv_reader(predict_file):
                    rects = json.loads(str_rects)
                    rects2 = []
                    for r in rects:
                        for l, s in zip(r['attr_labels'], r['attr_scores']):
                            rects2.append({'rect': r['rect'], 'class': str(l), 'conf': s})
                    yield key, json_dump(rects2)
            out_tsv = op.splitext(predict_file)[0] + '.attr.tsv'
            tsv_writer(gen_rows(), out_tsv)
            from qd.deteval import deteval_iter
            deteval_iter(
                    dataset.iter_data('test', 'attr',
                        version=None),
                    out_tsv,
                    report_file=evaluate_file,
                    ovthresh=[0.5],
                    force_evaluate=True)
        elif self.cfg.evaluate_method == 'coco_box':
            from qd.cocoeval import convert_gt_to_cocoformat
            from qd.cocoeval import convert_to_cocoformat
            from qd.cocoeval import coco_eval_json
            pred_tsv_to_json_extra = self.cfg.pred_tsv_to_json_extra
            gt_json = dataset.get_data(self.cfg.test_split, 'label.cocoformat',
                    version=self.cfg.test_version) + '.json'
            gt_iter = dataset.iter_data(self.cfg.test_split, 'label',
                        version=self.cfg.test_version)

            if not op.isfile(gt_json) or self.cfg.force_evaluate:
                convert_gt_to_cocoformat(gt_iter, gt_json)
            if pred_tsv_to_json_extra == 1:
                predict_json = predict_file + '.cocoformat.json'
            else:
                assert pred_tsv_to_json_extra == 0
                predict_json = predict_file + '.cocoformat.0.json'
            is_empty = False
            if worth_create(predict_file, predict_json) or self.cfg.force_evaluate:
                annotations = convert_to_cocoformat(predict_file, predict_json,
                                                    extra=pred_tsv_to_json_extra)
                if len(annotations) == 0:
                    is_empty = True
            else:
                from qd.qd_common import get_file_size
                if get_file_size(predict_json) < 100 and \
                        len(json.loads(read_to_buffer(predict_json))) == 0:
                    is_empty = True
            if is_empty:
                result = {'0.5-all': 0,
                        '0.75-all': 0,
                        'AR-all': 0,
                        'AR-all-1': 0,
                        'AR-all-10': 0,
                        'AR-large': 0,
                        'AR-medium': 0,
                        'AR-small': 0,
                        'all-all': 0,
                        'all-large': 0,
                        'all-medium': 0,
                        'all-small': 0}
            else:
                result = coco_eval_json(predict_json, gt_json,
                        maxDet=self.cfg.coco_eval_max_det)

            write_to_yaml_file(result, evaluate_file)
        elif self.cfg.evaluate_method == 'top1':
            iter_label = dataset.iter_data(self.cfg.test_split, 'label',
                    self.cfg.test_version)
            top1 = evaluate_topk(tsv_reader(predict_file), iter_label)
            logging.info('top1 = {}'.format(top1))
            write_to_yaml_file({'top1': top1}, evaluate_file)
        elif self.cfg.evaluate_method == 'neg_aware_gmap':
            from qd.evaluate.evaluate_openimages_google import evaluate
            truths = dataset.get_data(self.cfg.test_split, 'label')
            imagelabel_truths = dataset.get_data(self.cfg.test_split, 'imagelabel')
            assert op.isfile(truths), truths
            assert op.isfile(imagelabel_truths)
            result = evaluate(truths, imagelabel_truths, predict_file,
                    json_hierarchy_file=op.join(dataset._data_root, 'hierarchy.json'),
                    apply_nms_det=self.cfg.apply_nms_det,
                    expand_label_det=self.cfg.expand_label_det,
                    expand_label_gt=True,
                    apply_nms_gt=self.apply_nms_gt,
                    )
            from qd.qd_common import convert_to_yaml_friendly
            result = convert_to_yaml_friendly(result)
            logging.info(pformat(result))
            logging.info('mAP = {}'.format(result['map']))
            write_to_yaml_file(result, evaluate_file)
        else:
            logging.info('unknown evaluate method = {}'.format(self.cfg.evaluate_method))

    def monitor_train(self):
        self._ensure_initialized()
        while True:
            need_wait_models = self.pred_eval_intermediate_models()
            all_step = self.get_all_steps()
            all_eval_file = [self.get_evaluate_file(self.get_predict_file(self.get_checkpoint_file(iteration=i)))
                for i in all_step]
            iter_to_eval = dict((i, get_acc_for_plot(eval_file))
                    for i, eval_file in zip(all_step, all_eval_file) if
                        op.isfile(eval_file))
            self.update_acc_iter(iter_to_eval)
            if need_wait_models == 0:
                break
            time.sleep(5)

        if self.mpi_rank == 0:
            self.save_to_tensorboard()
        synchronize()

    def update_acc_iter(self, iter_to_eval):
        if self.mpi_rank == 0:
            xys = list(iter_to_eval.items())
            xys = sorted(xys, key=lambda x: x[0])
            xs = [x for x, _ in xys]
            if len(xys) > 0:
                keys = xys[0][1].keys()
                for k in keys:
                    # coco accuracy
                    ys = [y[k] for _, y in xys]
                    out_file = os.path.join(
                        self.output_folder,
                        'map_{}_{}_{}.png'.format(self.cfg.test_data,
                            self.cfg.test_split, k.replace('$', '_')))
                    logging.info('create {}'.format(out_file))
                    if op.isfile(out_file):
                        os.remove(out_file)
                    plot_to_file(xs, ys, out_file)
            else:
                logging.info('nothing plotted')
        synchronize()

    def save_to_tensorboard(self):
        all_step = self.get_all_steps()
        all_eval_file = [self.get_evaluate_file(self.get_predict_file(self.get_checkpoint_file(iteration=s)))
            for s in all_step]
        all_step_eval_result = [(s, get_acc_for_plot(e)) for s, e in zip(all_step,
            all_eval_file) if op.isfile(e)]

        tensorboard_folder = op.join('output', self.full_expid, 'tensorboard_data')
        from torch.utils.tensorboard import SummaryWriter
        ensure_remove_dir(tensorboard_folder)
        wt = SummaryWriter(log_dir=tensorboard_folder)
        tag_prefix = '{}_{}'.format(self.cfg.test_data, self.cfg.test_split)
        for step, eval_result in all_step_eval_result:
            for k in eval_result:
                wt.add_scalar(tag='{}_{}'.format(tag_prefix, k),
                        scalar_value=eval_result[k],
                        global_step=step)
        wt.close()

    def pred_eval_intermediate_models(self):
        ready_predict, all_step = self.get_intermediate_model_status()
        all_ready_predict_step = [step for step, status in zip(all_step, ready_predict) if status == 1]
        for step in all_ready_predict_step:
            model_file = self.get_checkpoint_file(iteration=step)
            pred = self.ensure_predict(model_file=model_file)
            self.ensure_evaluate(pred)
            synchronize()
        not_exist_steps = [step for step, status in zip(all_step, ready_predict) if status == 0]
        logging.info('not exist steps = {}'.format(not_exist_steps))
        need_wait_models = [x for x in ready_predict if x == 0]
        return len(need_wait_models)

    def get_intermediate_model_status(self):
        ready_predict = []
        all_step = self.get_all_steps()
        for step in all_step[:-1]:
            model_file = self.get_checkpoint_file(iteration=step)
            if not op.isfile(model_file) and not op.isdir(model_file):
                ready_predict.append(0)
                continue
            predict_result_file = self.get_predict_file(model_file)
            eval_file = self.get_evaluate_file(predict_result_file)
            if not worth_create(model_file, predict_result_file) and \
                    not worth_create(predict_result_file, eval_file):
                ready_predict.append(-1)
                continue
            ready_predict.append(1)
        if self.mpi_size > 1:
            # by default, we use nccl backend, which only supports gpu. Thus,
            # we should not use cpu here.
            ready_predict = torch.tensor(ready_predict).cuda()
            dist.broadcast(ready_predict, src=0)
            ready_predict = ready_predict.tolist()
        return ready_predict, all_step[:-1]

    def get_snapshot_steps(self):
        return self.cfg.snapshot_steps

    def get_all_steps(self):
        steps = self.get_snapshot_steps()
        curr = 0
        all_step = []
        while True:
            curr += steps
            if curr >= self.max_iter:
                all_step.append(self.max_iter)
                break
            all_step.append(curr)
        return all_step

    def ensure_create_evaluate_meta_file(self, evaluate_file):
        if self.cfg.evaluate_method == 'map':
            ensure_create_evaluate_meta_file(evaluate_file)

def get_acc_for_plot(eval_file):
    if 'coco_box' in eval_file:
        return load_from_yaml_file(eval_file)
    elif 'top1' in eval_file:
        return load_from_yaml_file(eval_file)
    elif 'vqa_acc' in eval_file:
        return load_from_yaml_file(eval_file)
    elif 'caption' in eval_file:
        return load_from_yaml_file(eval_file)
    else:
        if op.isfile(eval_file + '.map.json'):
            x = json.loads(read_to_buffer(eval_file + '.map.json'))
            from qd.qd_common import dict_get_all_path, dict_get_path_value
            return {p: dict_get_path_value(x, p) for p in dict_get_all_path(x)}
        return load_from_yaml_file(eval_file)

def ensure_create_evaluate_meta_file(evaluate_file):
    result = None
    simple_file = evaluate_file + '.map.json'
    if worth_create(evaluate_file, simple_file):
        if result is None:
            logging.info('data reading...')
            eval_result= read_to_buffer(evaluate_file)
            logging.info('json parsing...')
            result = json.loads(eval_result)
        s = {}
        for size_type in result:
            if size_type not in s:
                s[size_type] = {}
            for thresh in result[size_type]:
                if thresh not in s[size_type]:
                    s[size_type][thresh] = {}
                s[size_type][thresh]['map'] = \
                        result[size_type][thresh]['map']
        write_to_file(json.dumps(s, indent=4, sort_keys=True), simple_file)

    simple_file = evaluate_file + '.class_ap.json'
    if worth_create(evaluate_file, simple_file):
        if result is None:
            eval_result= read_to_buffer(evaluate_file)
            result = json.loads(eval_result)
        s = {}
        for size_type in result:
            if size_type not in s:
                s[size_type] = {}
            for thresh in result[size_type]:
                if thresh not in s[size_type]:
                    s[size_type][thresh] = {}
                s[size_type][thresh]['class_ap'] = \
                        result[size_type][thresh]['class_ap']
        write_to_file(json.dumps(s, indent=4, sort_keys=True), simple_file)

    simple_file = '{}.prec.threshold.tsv'.format(evaluate_file)
    if worth_create(evaluate_file, simple_file):
        if result is None:
            logging.info('data reading...')
            eval_result= read_to_buffer(evaluate_file)
            logging.info('json parsing...')
            result = json.loads(eval_result)
        _, max_key = max([(float(k), k) for k in result['overall']],
                key=lambda x: x[0])
        class_thresh = result['overall'][max_key]['class_thresh']
        precision_ths = None
        for l in class_thresh:
            precision_ths = class_thresh[l].keys()
            break
        if precision_ths:
            for precision_th in precision_ths:
                sub_simple_file = '{}.{}.prec{}.threshold.tsv'.format(
                        evaluate_file, max_key, precision_th)
                def gen_rows():
                    for l in class_thresh:
                        th_recall = class_thresh[l].get(precision_th, [1, 0])
                        yield l, th_recall[0], th_recall[1]
                tsv_writer(gen_rows(), sub_simple_file)
        from_file = '{}.{}.prec{}.threshold.tsv'.format(evaluate_file, max_key, 0.5)
        if op.isfile(from_file) and worth_create(from_file, simple_file):
            copyfile(from_file, simple_file)

from qd.data_layer.transform import BGR2RGB

def get_transform_image(self, is_train):
    # used by cls_uni_pipeline and caption_uni_pipeline (image encoder)
    train_transform = self.cfg.train_transform
    if train_transform == 'vit':
        # timm style
        transform = get_transform_vit_default(self, is_train=is_train)
    elif train_transform == 'deit':
        transform = get_transform_deit_default(self, is_train=is_train)
    elif train_transform == 'inception':
        transform = get_transform_incepion(self, is_train)
    elif train_transform == 'rand_cut':
        transform = get_transform_rand_cut(self, is_train)
        assert self.cfg.test_respect_ratio_max is None
    else:
        raise NotImplementedError(train_transform)
    return transform

def get_transform_image_norm(self):
    if self.cfg.data_normalize is None:
        from qd.data_layer.transform import get_default_mean, get_default_std
        normalize = transforms.Normalize(
            mean=get_default_mean(), std=get_default_std())
    elif self.cfg.data_normalize == 0.5:
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        raise NotImplementedError(self.cfg.data_normalize)
    return normalize

def get_transform_deit_default(self, is_train):
    # this hyperparameter is based on https://github.com/facebookresearch/deit
    normalize = get_transform_image_norm(self)
    if not is_train:
        trans = [
            # data loader will be pil, no need to convert
            #BGR2RGB(),
            #transforms.ToPILImage(),
        ]
        if self.cfg.test_respect_ratio_max:
            from qd.data_layer.transform import MinMaxResizeForTest
            trans.extend([
                MinMaxResizeForTest(self.cfg.test_crop_size, self.cfg.test_respect_ratio_max)
            ])
        else:
            trans.extend([
                transforms.Resize(int(math.floor(self.cfg.test_crop_size / self.cfg.crop_pct)), PIL.Image.BICUBIC),
                transforms.CenterCrop(self.cfg.test_crop_size),
            ])
        trans.extend([
            transforms.ToTensor(),
            normalize,
        ])
        transform = transforms.Compose(trans)
    else:
        from timm.data import create_transform
        scale = None
        if self.cfg.input_small_scale is not None:
            scale = (self.cfg.input_small_scale, 1.)
        hflip = 0.5
        if self.cfg.no_flip:
            hflip = 0.
        assert not self.cfg.no_color_jitter
        transform = create_transform(
            input_size=self.cfg.train_crop_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            mean=normalize.mean,
            std=normalize.std,
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            scale=scale,
            hflip=hflip,
        )
    return transform

def get_transform_vit_default(self, is_train):
    normalize = get_transform_image_norm(self)
    if not is_train:
        trans = [
            BGR2RGB(),
            transforms.ToPILImage(),
        ]
        if self.cfg.test_respect_ratio_max:
            from qd.data_layer.transform import MinMaxResizeForTest
            trans.extend([
                MinMaxResizeForTest(self.cfg.test_crop_size, self.cfg.test_respect_ratio_max)
            ])
        else:
            trans.extend([
                transforms.Resize(int(math.floor(self.cfg.test_crop_size / self.cfg.crop_pct)), PIL.Image.BICUBIC),
                transforms.CenterCrop(self.cfg.test_crop_size),
            ])
        trans.extend([
            transforms.ToTensor(),
            normalize,
        ])
        transform = transforms.Compose(trans)
    else:
        from qd.data_layer.transform import get_inception_train_transform
        transform = get_inception_train_transform(
            bgr2rgb=True,
            crop_size=self.cfg.train_crop_size,
            normalize=normalize,
            small_scale=self.cfg.input_small_scale,
            no_color_jitter=self.cfg.no_color_jitter,
            no_flip=self.cfg.no_flip,
            no_aspect_dist=self.cfg.no_aspect_dist,
            resize_crop=self.cfg.resize_crop,
            max_size=self.cfg.train_max_size,
        )
    return transform

def get_transform_incepion(self, is_train):
    if is_train:
        from qd.data_layer.transform import get_inception_train_transform
        transform = get_inception_train_transform(
            bgr2rgb=True, crop_size=self.cfg.train_crop_size)
    else:
        resize_size = self.cfg.test_resize_size
        if resize_size is None:
            resize_size = 256 * self.cfg.test_crop_size // 224
        from qd.data_layer.transform import get_inception_test_transform
        transform = get_inception_test_transform(
            bgr2rgb=True,
            resize_size=resize_size,
            crop_size=self.cfg.test_crop_size,
            crop_position=self.cfg.test_crop_position,
            with_crop=not self.cfg.no_crop,
            interpolation=self.cfg.interpolation,
            test_respect_ratio_max=self.cfg.test_respect_ratio_max,
        )

    return transform

def get_transform_rand_cut(self, is_train):
    if not is_train:
        return get_transform_incepion(self, is_train)

    from qd.data_layer.transform import get_data_normalize

    normalize = get_data_normalize()
    totensor = transforms.ToTensor()
    if self.cfg.min_size_range32 is None:
        all_trans = []
        all_trans.append(BGR2RGB())
        from qd.data_layer.rand_augmentation import rand_augment_transform

        # this is default
        config_str = 'rand-m9-mstd0.5'
        fillcolor = [0.5, 0.5, 0.5]
        hparams = dict(
            translate_const=int(self.cfg.train_crop_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in fillcolor]),
        )

        all_trans.extend([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(self.cfg.train_crop_size),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform(config_str, hparams),
            totensor,
            normalize,
            transforms.RandomApply([ImageCutout(1./self.cfg.cutout_factor)], p=0.5),
        ])
        data_augmentation = transforms.Compose(all_trans)
    else:
        first_trans = []
        first_trans.append(BGR2RGB())
        from qd.data_layer.rand_augmentation import rand_augment_transform

        # this is default
        config_str = 'rand-m9-mstd0.5'
        fillcolor = [0.5, 0.5, 0.5]
        hparams = dict(
            translate_const=int(self.cfg.train_crop_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in fillcolor]),
        )

        first_trans.extend([
            transforms.ToPILImage(),
        ])
        first_trans = transforms.Compose(first_trans)
        first_trans = ImageTransform2Dict(first_trans)
        from qd.data_layer.transform import RandomResizedCropMultiSize
        all_size = list(range(self.cfg.min_size_range32[0], self.cfg.min_size_range32[1] + 31, 32))
        if self.cfg.train_crop_size not in all_size:
            all_size.append(self.cfg.train_crop_size)
        second_trans = RandomResizedCropMultiSize(all_size)
        third_trans = [
            transforms.RandomHorizontalFlip(),
            rand_augment_transform(config_str, hparams),
            totensor,
            normalize,
            transforms.RandomApply([ImageCutout(1./self.cfg.cutout_factor)], p=0.5),
        ]
        third_trans = transforms.Compose(third_trans)
        third_trans = ImageTransform2Dict(third_trans)
        data_augmentation = transforms.Compose([
            first_trans, second_trans, third_trans])
    return data_augmentation


# --------------------------
def get_cls_criterion(self):
    if self.cfg.dataset_type in [
            'crop', 'single_dict', 'io']:
        if self.cfg.loss_type == 'NTXent':
            from qd.layers.ntxent_loss import NTXentLoss
            criterion = NTXentLoss(self.cfg.temperature, self.cfg.correct_loss)
        elif self.cfg.loss_type == 'NTXentQueue':
            from qd.layers.ntxent_loss import NTXentQueueLoss
            criterion = NTXentQueueLoss(self.cfg.temperature,
                                        self.cfg.queue_size,
                                        self.cfg.out_dim,
                                        self.cfg.queue_alpha,
                                        alpha_max=self.cfg.queue_alpha_max,
                                        alpha_policy=self.cfg.queue_alpha_policy,
                                        max_iter=self.cfg.max_iter,
                                        criterion_type=self.cfg.criterion_type,
                                        denominator_ce_factor=self.cfg.denominator_ce_factor
                                        )
        elif self.cfg.loss_type == 'SwAV':
            from qd.layers.ntxent_loss import SwAVQueueLoss
            criterion = SwAVQueueLoss(
                self.cfg.temperature,
                cluster_size=self.cfg.cluster_size,
                queue_size=self.cfg.queue_size,
                involve_queue_after=self.cfg.involve_queue_after,
                dim=self.cfg.out_dim)
        elif self.cfg.loss_type == 'SimpleQueue':
            from qd.layers.ntxent_loss import SimpleQueueLoss
            criterion = SimpleQueueLoss(self.cfg.temperature,
                                        self.cfg.queue_size,
                                        self.cfg.out_dim,
                                        self.cfg.queue_alpha,
                                        alpha_max=self.cfg.queue_alpha_max,
                                        alpha_policy=self.cfg.queue_alpha_policy,
                                        max_iter=self.cfg.max_iter,
                                        criterion_type=self.cfg.criterion_type,
                                        denominator_ce_factor=self.cfg.denominator_ce_factor
                                        )
        elif self.cfg.loss_type == 'NoisyDis':
            from qd.layers.ntxent_loss import NoisyDiscriminator
            criterion = NoisyDiscriminator(self.cfg.out_dim)
        elif self.cfg.loss_type == 'multi_ce':
            from qd.layers.loss import MultiCrossEntropyLoss
            criterion = MultiCrossEntropyLoss(weights=self.cfg.multi_ce_weights)
        elif self.cfg.loss_type == 'dist_ce':
            from qd.layers.loss import DistilCrossEntropyLoss
            criterion = DistilCrossEntropyLoss(self.cfg.dist_ce_weight)
        elif self.cfg.loss_type == 'smooth_ce':
            from qd.layers.loss import SmoothLabelCrossEntropyLoss
            criterion = SmoothLabelCrossEntropyLoss(eps=self.cfg.smooth_label_eps)
        elif self.cfg.loss_type == 'ExCE':
            from qd.layers.loss import ExclusiveCrossEntropyLoss
            criterion = ExclusiveCrossEntropyLoss(2.)
        elif self.cfg.loss_type == 'kl_ce':
            from qd.layers.loss import KLCrossEntropyLoss
            criterion = KLCrossEntropyLoss()
        elif self.cfg.loss_type == 'multi_klce':
            from qd.layers.loss import MultiKLCrossEntropyLoss
            criterion = MultiKLCrossEntropyLoss()
        elif self.cfg.loss_type == 'l2':
            from qd.layers.loss import L2Loss
            criterion = L2Loss()
        elif self.cfg.loss_type == 'mo_dist_ce':
            from qd.layers.loss import DistillCrossEntropyLoss
            criterion = DistillCrossEntropyLoss(
                num_image=self.cfg.get_num_training_images(),
                num_class=self.cfg.get_num_classes(),
                momentum=self.cfg.dist_ce_momentum,
                dist_weight=self.cfg.dist_weight,
            )
        elif self.cfg.loss_type == 'eff_fpn_ce':
            from qd.layers.loss import EfficientDetCrossEntropy
            criterion = EfficientDetCrossEntropy(
                no_reg=self.cfg.no_reg,
                sep=self.cfg.sep,
            )
        else:
            criterion = nn.CrossEntropyLoss().cuda()
    elif self.cfg.dataset_type in ['soft_assign']:
        from qd.layers.kl_div_logit_loss import KLDivLogitLoss
        criterion = KLDivLogitLoss()
    elif self.cfg.dataset_type == 'multi_hot':
        if self.cfg.loss_type == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss().cuda()
        elif self.cfg.loss_type == 'BCELogitsNormByPositive':
            from qd.layers.loss import BCELogitsNormByPositive
            criterion = BCELogitsNormByPositive()
        elif self.cfg.loss_type == 'MultiHotCrossEntropyLoss':
            criterion = MultiHotCrossEntropyLoss()
        else:
            raise Exception('not support value {}'.format(self.cfg.loss_type))
    elif self.cfg.dataset_type == 'multi_hot_neg':
        assert self.cfg.loss_type == 'BCEWithLogitsNegLoss'
        criterion = BCEWithLogitsNegLoss()
    else:
        raise NotImplementedError
    return criterion

def get_raw_timm_model(self, is_train):
    net = self.cfg.net[5:]
    import timm
    kwargs = {}
    if self.cfg.timm_attn_group_size:
        kwargs['group_size'] = self.cfg.timm_attn_group_size
    cfg_to_kwargs = [
        ('ip_attn_non_linear', 'non_linear'),
        ('ip_attn_init_to_sim', 'init_to_sim'),
        ('ip_attn_no_extra_linear', 'no_extra_linear'),
        ('qk_norm', 'qk_norm'),
        ('attention_type', 'attention_type'),
        ('drop_path', 'drop_path'),
        ('timm_attn_group_size', 'group_size'),
        ('elu_plus_n', 'elu_plus_n'),
    ]
    for k, v in cfg_to_kwargs:
        if self.cfg.get(k):
            kwargs[v] = self.cfg.get(k)
    logging.info(pformat(kwargs))
    model = timm.create_model(
        net,
        pretrained=self.cfg.pretrained,
        **kwargs,
    )
    if is_train:
        criterion = get_cls_criterion(self)
        model = ModelLoss(model, criterion)
    else:
        model = InputAsDict(model)
    return model

def get_parameter_groups_general(self, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = self.cfg.weight_decay
        lr = self.cfg.base_lr
        if self.cfg.bias_no_weight_decay and "bias" in key:
            weight_decay = 0.
        if self.cfg.ln_no_weight_decay and 'LayerNorm.weight' in key:
            weight_decay = 0
        if self.cfg.conv_no_weight_decay and key.endswith('conv.weight'):
            weight_decay = 0.
        if self.cfg.pos_no_weight_decay and key.endswith('.pos_embed'):
            weight_decay = 0.
        if self.cfg.cls_token_no_weight_decay and key.endswith('.cls_token'):
            weight_decay = 0.
        if self.cfg.lr_mult_classifier and '.classifier.' in key:
            # in vqa fine-tuning
            lr *= self.cfg.lr_mult_classifier
        params.append({
            'weight_decay': weight_decay,
            'lr': lr,
            'param': value,
            'param_name': key,
        })
    wlp = [((p['weight_decay'], p['lr']), p) for p in params]
    from qd.qd_common import list_to_dict
    wl_to_ps = list_to_dict(wlp, 0)
    ret = []
    for (curr_weight_decay, curr_lr), ps in wl_to_ps.items():
        p = {'weight_decay': curr_weight_decay, 'lr': curr_lr}
        p['params'] = [p['param'] for p in ps]
        p['param_names'] = [p['param_name'] for p in ps]
        ret.append(p)
    return ret

def get_parameter_groups(self, model):
    # use get_parameter_groups_general. this one is deprecated
    raise ValueError('use get_parameter_groups_general')
    lr = self.cfg.base_lr
    from collections import defaultdict
    decay_to_info = defaultdict(lambda: defaultdict(list))
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = self.cfg.weight_decay
        if self.cfg.bias_no_weight_decay and "bias" in key:
            weight_decay = 0.
        if self.cfg.ln_no_weight_decay and 'LayerNorm.weight' in key:
            weight_decay = 0
        if self.cfg.conv_no_weight_decay and key.endswith('conv.weight'):
            weight_decay = 0.
        if self.cfg.pos_no_weight_decay and key.endswith('.pos_embed'):
            weight_decay = 0.
        if self.cfg.cls_token_no_weight_decay and key.endswith('.cls_token'):
            weight_decay = 0.
        logging.info('{}: lr = {}; weight_decay = {}'.format(
            key, lr, weight_decay
        ))
        decay_to_info[weight_decay]['params'].append(value)
        decay_to_info[weight_decay]['param_names'].append(key)
    ps = []
    for w, info in decay_to_info.items():
        p = {'weight_decay': w, 'lr': lr}
        p.update(info)
        ps.append(p)
    return ps

def get_image_encoder(self, is_train, hidden_size):
    # used by clip_uni_pipeline
    encoder_type = self.cfg.image_encoder_type
    out_dim = hidden_size
    pretrained = self.cfg.image_encoder_pretrained
    if encoder_type.startswith('resnet'):
        param = {
            'pretrained': pretrained,
            'num_classes': out_dim,
            'out_adaptive_pools': self.cfg.out_adaptive_pools,
            'out_pools': self.cfg.out_pools,
        }
        return execute_func({
            'from': 'qd.layers.resnet_vl',
            'import': encoder_type,
            'param': param,
        })
    elif encoder_type.startswith('CLIP'):
        if encoder_type == 'CLIPresnet50':
            input_resolution = (self.cfg.train_crop_size if is_train else
                                self.cfg.test_crop_size)
            return execute_func({
                'from': 'qd.layers.CLIP.model',
                'import': 'ModifiedResNet',
                'param': {
                    'layers': (3, 4, 6, 3),
                    'output_dim': 1024,
                    'heads': 32,
                    'input_resolution': input_resolution,
                    'width': 64,
                },
            })
        elif encoder_type == 'CLIPViT_B_32':
            input_resolution = (self.cfg.train_crop_size if is_train else
                                self.cfg.test_crop_size)
            return execute_func({
                'from': 'qd.layers.CLIP.model',
                'import': 'VisualTransformer',
                'param': {
                    'input_resolution': input_resolution,
                    'patch_size': 32,
                    'width': 768,
                    'layers': 12,
                    'heads': 12,
                    'output_dim': self.cfg.embed_dim or 512,
                },
            })
        else:
            raise NotImplementedError
    elif encoder_type.startswith('timm_'):
        net = encoder_type[5:]
        return execute_func({
            'from': 'qd.pipelines.clip_uni_pipeline',
            'import': 'create_timm_image_encoder',
            'param': {
                'output_dim': self.cfg.embed_dim,
                'pooler_type': self.cfg.image_pooler_type,
                'model_name': net,
                'pretrained': pretrained,
                'output_grid': True,
            }
        })
    else:
        raise NotImplementedError

def update_joint_encoder_config(config, cfg):
    # used in vqa/caption unipipeline
    if 'vit' in cfg.text_encoder_type:
        # this is just to make sure we are using the right variables for
        # vit model
        config.timm_param = {}
        if cfg.drop_path_rate:
            config.timm_param['drop_path_rate'] = cfg.drop_path_rate
        if cfg.drop_rate:
            config.timm_param['drop_rate'] = cfg.drop_rate
        if cfg.attn_drop_rate:
            config.timm_param['attn_drop_rate'] = cfg.attn_drop_rate

        if cfg.fusion_timm_param_drop_out_all:
            if not hasattr(config, 'timm_param'):
                config.timm_param = {}
            config.timm_param['drop_rate'] = cfg.fusion_timm_param_drop_out_all
            config.timm_param['attn_drop_rate'] = cfg.fusion_timm_param_drop_out_all
            config.timm_param['drop_path_rate'] = cfg.fusion_timm_param_drop_out_all
        if cfg.norm_after:
            config.norm_after = cfg.norm_after
        if cfg.attention_type:
            config.timm_param['attention_type'] = cfg.attention_type

def get_image_encoder_model(self, is_train):
    if self.cfg.image_encoder_type.startswith('timm_'):
        net = self.cfg.image_encoder_type[5:]
        import timm
        model = timm.create_model(
            net,
            output_grid=True,
            pretrained=False,
        )
        if not is_train:
            model.eval()
        from qd.torch_common import InputAsDict
        model = InputAsDict(model)
    elif self.cfg.image_encoder_type.startswith('VitEmb_'):
        # VitEmb_base32_384
        net = self.cfg.image_encoder_type[len('VitEmb_'):]
        import timm
        model = timm.create_model(
            net,
            output_grid=True,
            pretrained=self.cfg.image_encoder_pretrained,
            img_size=self.cfg.train_crop_size,
            # during test, we will not do the sampling
            sample_token=(self.cfg.sample_token if is_train else None),
            sample_style=self.cfg.sample_style if is_train else None,
        )
        # clear out the following two modules
        model.norm = nn.Identity()
        model.blocks = nn.ModuleList()
        if not is_train:
            model.eval()
        from qd.torch_common import InputAsDict
        model = InputAsDict(model)
    elif self.cfg.image_encoder_type.startswith('EmbCLIPViT_B_32'):
        input_resolution = (self.cfg.train_crop_size if is_train else
                            self.cfg.test_crop_size)
        from qd.layers.CLIP.model import VisualTransformer
        model = VisualTransformer(
            input_resolution=input_resolution,
            patch_size=32,
            width=768,
            layers=12,
            heads=12,
            output_dim=512,
            output_grid=True,
        )
        model.transformer = nn.Identity()
        if not is_train:
            model.eval()
        return model
    elif self.cfg.image_encoder_type.startswith('vit'):
        # prefer to use VitEmb_; as this is too flexible and it is easy to
        # make mistakes.
        parts = list(self.cfg.image_encoder_type.split('_'))[1:]
        depth, embed_dim, patch_size, num_heads = 12, 386, 16, 12
        for p in parts:
            if p.startswith('d'):
                depth = int(p[1:])
            elif p.startswith('h'):
                embed_dim = int(p[1:])
            elif p.startswith('p'):
                patch_size = int(p[1:])
            elif p.startswith('a'):
                num_heads = int(p[1:])
            else:
                raise NotImplementedError
        if depth == 0:
            # image encoder has done projection
            assert self.cfg.ignore_project_image
            assert not self.cfg.use_img_layernorm
        model_kwargs = dict(patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                            num_heads=num_heads)
        img_size = self.cfg.train_crop_size if is_train else self.cfg.test_crop_size
        from timm.models.vision_transformer import VisionTransformer
        if self.cfg.image_encoder_ignore_norm:
            # use case, we ignore norm here. In joint fusion, it will be
            # passed to the ViT, which requires the input not be normed.
            model_kwargs['norm_layer'] = lambda x: nn.Identity()
        model = VisionTransformer(
            img_size=img_size, num_classes=-1, output_grid=True, **model_kwargs)
        if not is_train:
            model.eval()
        from qd.torch_common import InputAsDict
        model = InputAsDict(model)
    else:
        raise NotImplementedError(self.cfg.image_encoder_type)
    return model

def convert_shared_clip_model_to_vilt(model_file, out_model):
    model = torch_load(model_file)
    model = model['model']
    p1 = 'image_encoder.model.blocks'
    p2 = 'text_encoder.encoder.blocks'
    num = 0
    to_remove = []
    for k in model:
        if k.startswith(p1):
            k2 = p2 + k[len(p1):]
            v = model[k]
            v2 = model[k2]
            to_remove.append(k2)
            num += 1
            assert (v - v2).abs().sum() == 0
    for t in to_remove:
        del model[t]
    assert num > 0
    old2new = {
        'image_encoder.model.cls_token': 'image_encoder.module.cls_token',
        'image_encoder.model.pos_embed': 'image_encoder.module.pos_embed',
        'image_encoder.model.patch_embed.proj.weight': 'image_encoder.module.patch_embed.proj.weight',
        'image_encoder.model.patch_embed.proj.bias': 'image_encoder.module.patch_embed.proj.bias',
    }
    prefix_replace = {
        'image_encoder.model.blocks.': 'module.bert.encoder.blocks.',
        'text_encoder.embeddings.': 'module.bert.embeddings.',
        'image_encoder.model.head.': 'image_encoder.module.head.',
    }
    useless = [
        # this is LayerNorm after all blocks. we also do not use it
        'image_encoder.model.norm.',
        # this is for classification purpose
        'image_encoder.model.head.',
    ]
    out = {}
    to_remove = []
    for old_k in model:
        if old_k in old2new:
            new_k = old2new[old_k]
            out[new_k] = model[old_k]
            to_remove.append(old_k)
            continue
        matched = [(k1, k2) for k1, k2 in prefix_replace.items() if old_k.startswith(k1)]
        if len(matched) == 1:
            k1, k2 = matched[0]
            new_k = k2 + old_k[len(k1):]
            out[new_k] = model[old_k]
            to_remove.append(old_k)
            continue
        else:
            assert len(matched) == 0
    for t in to_remove:
        del model[t]
    assert all(any(k.startswith(u) for u in useless) for k in model)

    torch_save(out, out_model)

def evaluate_vqa(self, predict_file, evaluate_file):
    # in TaxVQAv2, it is test; in TaxVQA, it is test_std
    if self.cfg.test_split in ['test_dev', 'test_std', 'test']:
        # we only convert the pred to json and then we should manually
        # upload the json file
        out_file = predict_file + '.server.json'
        from qd.tsv_io import tsv_reader
        result = [json.loads(s) for _, s in tsv_reader(predict_file)]
        from qd.qd_common import write_to_file, json_dump
        write_to_file(json_dump(result), out_file)
    else:
        return evaluate_acc_vqa(self, predict_file, evaluate_file)

def evaluate_acc_vqa(self, predict_file, evaluate_file):
    from qd.tsv_io import TSVDataset
    dataset = TSVDataset(self.cfg.test_data)
    all_qa = [json.loads(s_cap) for key, s_cap in dataset.iter_data(
        self.cfg.test_split,
        'caption')]
    num_caps = [len(qa) for qa in all_qa]
    caption_linelist = [(idx_img, idx_cap) for idx_img, n in enumerate(num_caps) for idx_cap in range(n)]
    correctness = []
    from qd.tsv_io import tsv_reader
    for index, s_pred in tqdm(tsv_reader(predict_file)):
        pred = json.loads(s_pred)['answer']
        index = int(index)
        idx_img, idx_cap = caption_linelist[index]
        gt = all_qa[idx_img][idx_cap]['answers']
        if len(gt) == 0:
            # this case, we ignore it to follow the released code in oscar
            continue
        if pred in gt:
            idx = gt.index(pred)
            correctness.append(all_qa[idx_img][idx_cap]['confs'][idx])
        else:
            correctness.append(0.)
    acc = torch.tensor(correctness).mean()
    from qd.qd_common import write_to_yaml_file
    logging.info(acc)
    write_to_yaml_file({'acc': float(acc)}, evaluate_file)
