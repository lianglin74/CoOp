from qd.qd_pytorch import TwoCropsTransform
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import os.path as op
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import ImageFilter
from qd.qd_common import print_frame_info
from qd.qd_pytorch import GaussianBlur
from qd.data_layer.transform import ImageToImageDictTransform

import argparse
import copy
import os
import os.path as osp
import time

import torch
from mmcv import Config

from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel


class MMDetPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ps = (self.net + '.py').split('__')
        config_file = op.join(
            'src',
            'mmdetection',
            'configs',
            *ps,
        )
        cfg = Config.fromfile(config_file)
        # if it is distributed, this field will be ignored.
        cfg.gpu_ids = [0]
        cfg.seed = 666
        cfg.work_dir = op.join(self.output_folder, 'snapshot')
        self.distributed = True
        # set cfg.work_dir
        samples_per_gpu = self.effective_batch_size // self.mpi_size
        cfg.data.samples_per_gpu = samples_per_gpu
        cfg.data.workers_per_gpu = 8
        cfg.optimizer.lr = self.base_lr
        cfg.total_epochs = self.max_epoch
        cfg.data.workers_per_gpu = self.num_workers

        # replace teh config for data
        from qd.qd_common import dump_to_yaml_str
        cfg.data.train.type = 'MMTSVDataset'
        cfg.data.train.ann_file = dump_to_yaml_str({
            'data': self.data,
            'split': 'train',
        })
        cfg.data.val.type = 'MMTSVDataset'
        cfg.data.val.ann_file = dump_to_yaml_str({
            'data': self.test_data,
            'split': self.test_split,
        })
        cfg.data.test.type = cfg.data.val.type
        cfg.data.test.ann_file = cfg.data.val.ann_file

        if self.basemodel is not None:
            cfg.model.pretrained = self.basemodel
            cfg.load_from = self.basemodel

        self.cfg = cfg

    def train(self):
        cfg = self.cfg
        from pprint import pformat
        logging.info(pformat(dict(cfg)))
        model = build_detector(
            cfg.model,
            train_cfg=cfg.train_cfg,
            test_cfg=cfg.test_cfg)

        logging.info(model)

        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))
        model.CLASSES = datasets[0].CLASSES
        train_detector(
            model,
            datasets,
            cfg,
            distributed=self.distributed,
            validate=False, # for now, make it True
            )
        model_final = op.join(self.output_folder,
                              'snapshot',
                              'epoch_{}.pth'.format(self.max_epoch))
        if self.max_epoch == 0 and not op.isfile(model_final):
            from qd.torch_common import torch_save
            x = model.state_dict()
            x = {'state_dict': x}
            torch_save(x, model_final)
        last_iter = self._get_checkpoint_file(iteration=self.max_iter)
        if self.mpi_rank == 0:
            if not op.isfile(last_iter):
                import shutil
                shutil.copy(model_final, last_iter)
        from qd.qd_pytorch import synchronize
        synchronize()
        return last_iter

    def model_surgery(self, model):
        if self.device == 'cpu':
            from qd.qd_pytorch import replace_module
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.SyncBatchNorm),
                    lambda m: torch.nn.BatchNorm2d(m.num_features,
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=m.affine,
                        track_running_stats=m.track_running_stats))
        return model

    def get_test_model(self):
        cfg = self.cfg
        cfg.model.pretrained = None
        if cfg.model.get('neck'):
            if cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None
        cfg.data.test.test_mode = True
        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            from mmdet.core import wrap_fp16_model
            wrap_fp16_model(model)
        model = self.model_surgery(model)
        model.CLASSES = self.get_labelmap()

        if self.device == 'cuda':
            if not self.distributed:
                model = MMDataParallel(model, device_ids=[0])
            else:
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[self.mpi_local_rank],
                    broadcast_buffers=False)
        else:
            assert self.device == 'cpu'
            # Use torchvision ops for CPU mode instead
            for m in model.modules():
                from mmcv.ops import RoIAlign, RoIPool
                if isinstance(m, (RoIPool, RoIAlign)):
                    if not m.aligned:
                        # aligned=False is not implemented on CPU
                        # set use_torchvision on-the-fly
                        m.use_torchvision = True
        return model

    def load_test_model(self, model, model_file):
        from mmcv.runner import load_checkpoint
        load_checkpoint(model, model_file, map_location='cpu',
                        logger=logging.getLogger())

    def get_test_data_loader(self):
        cfg = self.cfg
        from mmdet.datasets import build_dataloader, build_dataset
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            # currently it has to be 1., predict_iter_forward also assume it is
            # 1 for cpu
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=self.distributed,
            shuffle=False)
        return data_loader

    def predict_output_to_tsv_row(self, output, keys, **kwargs):
        # there is only one image's result
        key = keys['img_metas'][0].data[0][0]['filename']
        rects = []
        labelmap = self.get_labelmap()
        for c_idx, c_rects in enumerate(output):
            for c_rect in c_rects:
                rect = {
                    'rect': list(map(float, c_rect[:4])),
                    'conf': float(c_rect[-1]),
                    'class': labelmap[c_idx]
                }
                rects.append(rect)
        from qd.qd_common import json_dump

        # debug
        #from qd.tsv_io import TSVDataset
        #from qd.qd_common import img_from_base64
        #dataset = TSVDataset(self.test_data)
        #img = img_from_base64(dataset.seek_by_key(key, 'test')[-1])
        #from qd.process_image import draw_rects, show_image
        #draw_rects([r for r in rects if r['conf'] > 0.3], img)
        #show_image(img)

        yield key, json_dump(rects)

    def predict_iter_forward(self, model, inputs):
        with torch.no_grad():
            if self.device == 'cpu':
                inputs['img_metas'] = inputs['img_metas'][0].data
            output = model(return_loss=False, rescale=True, **inputs)
        return output

    def _get_test_normalize_module(self):
        return

