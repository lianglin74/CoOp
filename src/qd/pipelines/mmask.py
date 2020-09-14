from qd.qd_pytorch import TwoCropsTransform
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import os.path as op
import logging
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

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
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import ImageFilter
from qd.qd_common import print_frame_info
from qd.qd_pytorch import GaussianBlur
from qd.qd_common import merge_dict_to_cfg
from qd.qd_common import load_from_yaml_file
from qd.qd_common import dict_update_nested_dict
from mmask.config import cfg
from qd.qd_common import (dict_has_path, dict_get_path_value,
                          dict_update_path_value)
from qd.qd_common import dump_to_yaml_str


class MMaskPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'bgr2rgb': True,
            'train_size_mode': 'random',
            'max_box': 300,
        })

        root = op.join('src', 'mmask', 'configs')
        config_file = op.join(root, self.net.replace('__', '/') + '.yaml')

        param = load_from_yaml_file(config_file)
        import copy
        self.default_net_param = copy.deepcopy(param)
        dict_update_nested_dict(self.kwargs, param, overwrite=False)

        if 'device' not in self.kwargs  and dict_has_path(self.kwargs, 'MODEL$DEVICE'):
            self.kwargs['device'] = dict_get_path_value(self.kwargs, 'MODEL$DEVICE')
        dict_update_path_value(self.kwargs,
                               'SOLVER$IMS_PER_BATCH',
                               self.effective_batch_size)
        max_iter = self.parse_iter(self.max_iter)
        dict_update_path_value(self.kwargs,
                               'SOLVER$MAX_ITER',
                               max_iter)
        if not dict_has_path(self.kwargs, 'SOLVER$STEPS'):
            dict_update_path_value(self.kwargs,
                                   'SOLVER$STEPS',
                                   (6*max_iter//9, 8*max_iter//9))
        train_arg = {'data': self.data,
                'split': 'train',
                'bgr2rgb': self.bgr2rgb}
        logging.info('bgr2rgb = {}; Should be true unless on purpose'.format(self.bgr2rgb))
        if self.MaskTSVDataset is not None:
            train_arg.update(self.MaskTSVDataset)
            assert 'bgr2rgb' not in self.MaskTSVDataset or \
                    self.MaskTSVDataset['bgr2rgb'] == self.bgr2rgb
        dict_update_path_value(self.kwargs,
                              'DATASETS$TRAIN',
                               ('${}'.format(dump_to_yaml_str(train_arg)),))
        test_arg = {'data': self.test_data,
                    'split': self.test_split,
                    'remove_images_without_annotations': False,
                    'bgr2rgb': self.bgr2rgb,
                    'max_box': self.max_box,
                    }
        dict_update_path_value(self.kwargs,
                               'DATASETS$TEST',
                               ('${}'.format(dump_to_yaml_str(test_arg)),))
        self.kwargs['OUTPUT_DIR'] = op.join('output', self.full_expid, 'snapshot')
        dict_update_path_value(self.kwargs,
                               'TEST$IMS_PER_BATCH',
                               self.test_batch_size * self.mpi_size)
        dict_update_path_value(self.kwargs,
                               'DATALOADER$NUM_WORKERS',
                               self.num_workers)
        if self.min_size_range32 is not None:
            min_size_train = tuple(range(self.min_size_range32[0], self.min_size_range32[1] + 32, 32))
            dict_update_path_value(self.kwargs, 'INPUT$MIN_SIZE_TRAIN',
                    min_size_train)
        if self.affine_resize:
            if self.affine_resize == 'AF':
                # AF: affine
                info = {'from': 'qd.qd_pytorch',
                        'import': 'DictTransformAffineResize',
                        'param': {'out_sizes': min_size_train}}
            elif self.affine_resize == 'RC':
                # RC: resize and crop
                info = {'from': 'qd.qd_pytorch',
                        'import': 'DictTransformResizeCrop',
                        'param': {'all_crop_size': min_size_train,
                                  'size_mode': self.train_size_mode}}
            else:
                raise NotImplementedError(self.affine_resize)
            dict_update_path_value(self.kwargs, 'INPUT$TRAIN_RESIZER',
                    dump_to_yaml_str(info))
        from qd.tsv_io import TSVDataset
        self.labelmap = TSVDataset(self.data).load_labelmap()
        #dict_update_path_value(self.kwargs,
                               #'MODEL$FCOS$NUM_CLASSES',
                               #len(self.labelmap)+1)
        dict_update_path_value(self.kwargs,
                               'MODEL$RETINANET$NUM_CLASSES',
                               len(self.labelmap)+1)
        dict_update_path_value(self.kwargs,
                               'MODEL$ROI_BOX_HEAD$NUM_CLASSES',
                               len(self.labelmap)+1)
        dict_update_path_value(self.kwargs, 'SOLVER$BASE_LR',
                               self.kwargs['base_lr'])
        if 'basemodel' not in self.kwargs and dict_has_path(self.kwargs, 'MODEL$WEIGHT'):
            dict_update_path_value(self.kwargs, 'basemodel',
                                   dict_get_path_value(self.kwargs,
                                                       'MODEL$WEIGHT'))

        merge_dict_to_cfg(self.kwargs, cfg)
        self.cfg = cfg
        logging.info('cfg = \n{}'.format(cfg))

    def append_predict_param(self, cc):
        super().append_predict_param(cc)
        if self.cfg.INPUT.MIN_SIZE_TEST != 800:
            cc.append('testInputSize{}'.format(self.cfg.INPUT.MIN_SIZE_TEST))
        if self.cfg.INPUT.MAX_SIZE_TEST != 1333:
            cc.append('testInputMax{}'.format(self.cfg.INPUT.MAX_SIZE_TEST))
        #if cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_ACTIVATE != 'softmax':
            #cc.append('{}'.format(cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_ACTIVATE))
        #if cfg.MODEL.ROI_HEADS.NMS_ON_MAX_CONF_AGNOSTIC:
            #cc.append('nmsMax')
        #if self.cfg.MODEL.RPN.FPN_POST_NMS_CONF_TH_TEST > 0:
            #cc.append('rpnTh{}'.format(
                #self.cfg.MODEL.RPN.FPN_POST_NMS_CONF_TH_TEST))
        default_fpn_post_nms_top_n_test = (dict_get_path_value(self.default_net_param,
            'MODEL$RPN$FPN_POST_NMS_TOP_N_TEST') if dict_has_path(self.default_net_param,
                'MODEL$RPN$FPN_POST_NMS_TOP_N_TEST') else 2000)
        if cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST != default_fpn_post_nms_top_n_test:
            cc.append('FPNpost{}'.format(cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST))
        if cfg.MODEL.ROI_HEADS.SCORE_THRESH != 0.05:
            cc.append('roidth{}'.format(cfg.MODEL.ROI_HEADS.SCORE_THRESH))
        if cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG != 100:
            cc.append('roidDets{}'.format(cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG))
        if cfg.TEST.DETECTIONS_PER_IMG != 100:
            cc.append('upto{}'.format(cfg.TEST.DETECTIONS_PER_IMG))

    def get_train_data_loader(self, start_iter):
        #from mmask.data import make_data_loader
        from maskrcnn_benchmark.data import make_data_loader
        data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=self.distributed,
            start_iter=start_iter,
        )
        return data_loader

    def get_test_data_loader(self):
        from maskrcnn_benchmark.data import make_data_loader
        loaders = make_data_loader(cfg,
                                  is_train=False,
                                  is_distributed=self.distributed)
        return loaders[0]


    def get_train_model(self):
        from mmask.modeling.detector import build_detection_model
        model = build_detection_model(cfg)
        model = self.model_surgery(model)
        return model

    def get_test_model(self):
        model = self.get_train_model()
        if self.device == 'cpu':
            # sync-bn does not support cpu
            from qd.torch_common import replace_module
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.SyncBatchNorm),
                    lambda m: torch.nn.BatchNorm2d(m.num_features,
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=m.affine,
                        track_running_stats=m.track_running_stats))
        return model

    def get_optimizer(self, model):
        from mmask.solver import make_optimizer
        optimizer = make_optimizer(cfg, model)
        return optimizer

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        from mmask.solver import make_lr_scheduler
        scheduler = make_lr_scheduler(cfg, optimizer)
        return scheduler

    def _get_test_normalize_module(self):
        return

    def predict_output_to_tsv_row(self, output, keys, **kwargs):
        from qd.qd_common import json_dump
        ds = kwargs['dataloader'].dataset
        if self.label_id_to_label is None:
            self.label_id_to_label = {i + 1: l for i, l in enumerate(self.labelmap)}
        from qd.torch_common import boxlist_to_list_dict
        for box_list, idx in zip(output, keys):
            key = ds.id_to_img_map[idx]
            wh_info = ds.get_img_info(idx)
            box_list = box_list.resize((wh_info['width'], wh_info['height']))
            rects = boxlist_to_list_dict(box_list, self.label_id_to_label,
                                         extra=0)

            #from qd.tsv_io import TSVDataset
            #from qd.qd_common import img_from_base64
            #from qd.process_image import draw_rects, show_image
            #dataset = TSVDataset('coco2017Full')
            #im = img_from_base64(dataset.seek_by_idx(split='test', idx=idx)[-1])

            #from qd.process_image import load_image
            #logging.info('debugging')
            #im = load_image('./src/FCOS/demo/images/COCO_val2014_000000128654.jpg')

            #draw_rects([r for r in rects if r['conf'] > 0.5], im)
            #show_image(im)

            yield key, json_dump(rects)

