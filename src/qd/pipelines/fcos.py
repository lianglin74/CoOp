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
from fcos_core.config import cfg
from qd.qd_common import (dict_has_path, dict_get_path_value,
                          dict_update_path_value)
from qd.qd_common import dump_to_yaml_str


class FCOSPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'bgr2rgb': True,
            'min_size_range32': (800, 800),
            'train_size_mode': 'random',
            'max_box': 300,
        })

        from fcos_core.config import _default_cfg
        # the cfg could be modified by other places and we have to do this
        # trick to restore to its default. Gradually, we can remove the
        # dependence on this global variable of cfg.
        cfg.merge_from_other_cfg(_default_cfg)
        if dict_has_path(kwargs, 'SOLVER$WARMUP_ITERS') and \
                isinstance(dict_get_path_value(kwargs, 'SOLVER$WARMUP_ITERS'),
                           str):
            x = dict_get_path_value(kwargs, 'SOLVER$WARMUP_ITERS')
            x = self.parse_iter(x)
            dict_update_path_value(kwargs, 'SOLVER$WARMUP_ITERS', x)

        fcos_root = op.join('aux_data', 'FCOS_configs')
        if self.net.startswith('retina'):
            config_file = op.join(fcos_root, 'retinanet', self.net + '.yaml')
        elif self.net.startswith('fcos'):
            config_file = op.join(fcos_root, 'fcos', self.net + '.yaml')
        elif self.net.startswith('reppoint'):
            config_file = op.join(fcos_root, 'reppoint', self.net + '.yaml')
        elif '__' in self.net:
            config_file = op.join(fcos_root, self.net.replace('__', '/') + '.yaml')
        else:
            config_file = op.join(fcos_root, self.net + '.yaml')

        param = load_from_yaml_file(config_file)
        import copy
        self.default_net_param = copy.deepcopy(param)
        self.custom_net_param = copy.deepcopy(self.kwargs)
        dict_update_nested_dict(self.kwargs, param, overwrite=False)

        fcos_on = self.net.startswith('fcos')
        dict_update_path_value(self.kwargs,
                               'MODEL$FCOS_ON',
                               fcos_on)
        if 'device' not in self.kwargs  and dict_has_path(self.kwargs, 'MODEL$DEVICE'):
            self.kwargs['device'] = dict_get_path_value(self.kwargs, 'MODEL$DEVICE')
        dict_update_path_value(self.kwargs,
                               'SOLVER$IMS_PER_BATCH',
                               self.effective_batch_size)
        max_iter = self.parse_iter(self.max_iter)
        dict_update_path_value(self.kwargs,
                               'SOLVER$MAX_ITER',
                               max_iter)
        if not dict_has_path(self.custom_net_param, 'SOLVER$STEPS'):
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
        min_size_train = tuple(range(self.min_size_range32[0], self.min_size_range32[1] + 32, 32))
        dict_update_path_value(self.kwargs, 'INPUT$MIN_SIZE_TRAIN', min_size_train)
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
        dict_update_path_value(self.kwargs,
                               'MODEL$FCOS$NUM_CLASSES',
                               len(self.labelmap)+1)
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

        if self.clear_to_remove:
            import qd.mask.structures.bounding_box as bd
            bd.TO_REMOVE = 0
        merge_dict_to_cfg(self.kwargs, cfg)
        self.cfg = cfg
        logging.info('cfg = \n{}'.format(cfg))

    def append_predict_param(self, cc):
        super().append_predict_param(cc)
        if self.cfg.INPUT.MIN_SIZE_TEST != 800:
            cc.append('testInputSize{}'.format(self.cfg.INPUT.MIN_SIZE_TEST))
        if self.cfg.INPUT.MAX_SIZE_TEST != 1333:
            cc.append('testInputMax{}'.format(self.cfg.INPUT.MAX_SIZE_TEST))
        if cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_ACTIVATE != 'softmax':
            cc.append('{}'.format(cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_ACTIVATE))
        if cfg.MODEL.ROI_HEADS.NMS_ON_MAX_CONF_AGNOSTIC:
            cc.append('nmsMax')
        if cfg.MODEL.ROI_HEADS.NM_FILTER == 3:
            # master branch-style maskrcnn code implementation. especially used
            # for feature extraction. That is, each region has only one box
            # while for non-extraction scenario, each regin can have multiple
            # boxes, each of which has different class names. In this case, use
            # NMS_ON_MAX_CONF_AGNOSTIC
            cc.append('MnmsMax')
        elif cfg.MODEL.ROI_HEADS.NM_FILTER == 4:
            cc.append('mlnms')
        else:
            assert cfg.MODEL.ROI_HEADS.NM_FILTER == 0
        if self.cfg.MODEL.RPN.FPN_POST_NMS_CONF_TH_TEST > 0:
            cc.append('rpnTh{}'.format(
                self.cfg.MODEL.RPN.FPN_POST_NMS_CONF_TH_TEST))
        default_fpn_post_nms_top_n_test = (dict_get_path_value(self.default_net_param,
            'MODEL$RPN$FPN_POST_NMS_TOP_N_TEST') if dict_has_path(self.default_net_param,
                'MODEL$RPN$FPN_POST_NMS_TOP_N_TEST') else 2000)
        default_pre_nms_top_n_test = (dict_get_path_value(self.default_net_param,
            'MODEL$RPN$PRE_NMS_TOP_N_TEST') if dict_has_path(self.default_net_param,
                'MODEL$RPN$PRE_NMS_TOP_N_TEST') else 6000)
        if cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST != default_pre_nms_top_n_test:
            cc.append('rpPre{}'.format(cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST))
        if cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST != default_fpn_post_nms_top_n_test:
            cc.append('FPNpost{}'.format(cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST))
        if cfg.MODEL.ROI_HEADS.SCORE_THRESH != 0.05:
            cc.append('roidth{}'.format(cfg.MODEL.ROI_HEADS.SCORE_THRESH))
        if cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG != 100:
            cc.append('roidDets{}'.format(cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG))
        if cfg.TEST.DETECTIONS_PER_IMG != 100:
            cc.append('upto{}'.format(cfg.TEST.DETECTIONS_PER_IMG))

        default_post_nms_top_n_test = (dict_get_path_value(self.default_net_param,
            'MODEL$RPN$POST_NMS_TOP_N_TEST') if dict_has_path(self.default_net_param,
                'MODEL$RPN$POST_NMS_TOP_N_TEST') else 1000)
        if cfg.MODEL.RPN.POST_NMS_TOP_N_TEST != default_post_nms_top_n_test:
            cc.append('rpPost{}'.format(cfg.MODEL.RPN.POST_NMS_TOP_N_TEST))
        if cfg.MODEL.ROI_HEADS.NMS != 0.5:
            cc.append('roidNMS{}'.format(cfg.MODEL.ROI_HEADS.NMS))

        #if cfg.MODEL.RETINANET.INFERENCE_TH != 0.05:
            #cc.append('retinath{}'.format(cfg.MODEL.RETINANET.INFERENCE_TH))
        #if cfg.MODEL.RETINANET.PRE_NMS_TOP_N != 1000:
            #cc.append('retprenms{}'.format(cfg.MODEL.RETINANET.PRE_NMS_TOP_N))

        if cfg.MODEL.RPN.NMS_POLICY.TYPE == 'nms':
            if cfg.MODEL.RPN.NMS_THRESH != cfg.MODEL.RPN.NMS_POLICY.THRESH:
                cfg.MODEL.RPN.NMS_POLICY.THRESH = cfg.MODEL.RPN.NMS_THRESH
        if cfg.MODEL.RPN.NMS_POLICY.TYPE != 'nms':
            cc.append(cfg.MODEL.RPN.NMS_POLICY.TYPE)
        if cfg.MODEL.RPN.NMS_THRESH != 0.7:
            cc.append('rpnnmspolicy{}'.format(cfg.MODEL.RPN.NMS_THRESH))
        if cfg.MODEL.RPN.NMS_POLICY.ALPHA != 0.5:
            cc.append('rpA{}'.format(cfg.MODEL.RPN.NMS_POLICY.ALPHA))
        if cfg.MODEL.RPN.NMS_POLICY.GAMMA != 0.5:
            cc.append('rpG{}'.format(cfg.MODEL.RPN.NMS_POLICY.GAMMA))
        if cfg.MODEL.RPN.NMS_POLICY.NUM != 2:
            cc.append('rpN{}'.format(cfg.MODEL.RPN.NMS_POLICY.NUM))

        if cfg.MODEL.RPN.NMS_POLICY.ALPHA2 != 0.1:
            cc.append('rpA2{}'.format(cfg.MODEL.RPN.NMS_POLICY.ALPHA2))
        if cfg.MODEL.RPN.NMS_POLICY.GAMMA2 != 0.1:
            cc.append('rpG2{}'.format(cfg.MODEL.RPN.NMS_POLICY.GAMMA2))
        if cfg.MODEL.RPN.NMS_POLICY.NUM2 != 1:
            cc.append('rpN2{}'.format(cfg.MODEL.RPN.NMS_POLICY.NUM2))
        if cfg.MODEL.RPN.NMS_POLICY.COMPOSE_FINAL_RERANK:
            cc.append('rpnnmsRerank')

        #if cfg.MODEL.ROI_HEADS.NMS_POLICY.TYPE != 'nms':
            #cc.append(cfg.MODEL.ROI_HEADS.NMS_POLICY.TYPE)
        #if cfg.MODEL.ROI_HEADS.NMS_POLICY.THRESH != 0.5:
            #cc.append('roinmspolicy{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.THRESH))

        #if cfg.MODEL.ROI_HEADS.NMS_POLICY.ALPHA != 0.5:
            #cc.append('roinmsAlpha{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.ALPHA))
        #if cfg.MODEL.ROI_HEADS.NMS_POLICY.GAMMA != 0.5:
            #cc.append('roinmsGamma{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.GAMMA))
        #if cfg.MODEL.ROI_HEADS.NMS_POLICY.NUM != 2:
            #cc.append('roinmsNum{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.NUM))

        #if cfg.MODEL.ROI_HEADS.NMS_POLICY.ALPHA2 != 0.1:
            #cc.append('roinmsAlpha2{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.ALPHA2))
        #if cfg.MODEL.ROI_HEADS.NMS_POLICY.GAMMA2 != 0.1:
            #cc.append('roinmsGamma2{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.GAMMA2))
        #if cfg.MODEL.ROI_HEADS.NMS_POLICY.NUM2 != 1:
            #cc.append('roinmsNum2{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.NUM2))
        #if cfg.MODEL.ROI_HEADS.NMS_POLICY.COMPOSE_FINAL_RERANK:
            #cc.append('roinmsRerank')

        #if cfg.MODEL.RETINANET.NMS_POLICY.TYPE != 'nms':
            #cc.append(cfg.MODEL.RETINANET.NMS_POLICY.TYPE)
        #if cfg.MODEL.RETINANET.NMS_POLICY.THRESH != 0.4:
            #cc.append('rnpolicy{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.THRESH))

        #if cfg.MODEL.RETINANET.NMS_POLICY.ALPHA != 0.4:
            #cc.append('rnAlpha{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.ALPHA))
        #if cfg.MODEL.RETINANET.NMS_POLICY.GAMMA != 0.4:
            #cc.append('rnGamma{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.GAMMA))
        #if cfg.MODEL.RETINANET.NMS_POLICY.NUM != 1:
            #cc.append('rnNum{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.NUM))

        #if cfg.MODEL.RETINANET.NMS_POLICY.ALPHA2 != 0.1:
            #cc.append('rnAlpha2{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.ALPHA2))
        #if cfg.MODEL.RETINANET.NMS_POLICY.GAMMA2 != 0.1:
            #cc.append('rnGamma2{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.GAMMA2))
        #if cfg.MODEL.RETINANET.NMS_POLICY.NUM2 != 0:
            #cc.append('rnNum2{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.NUM2))
        #if cfg.MODEL.RETINANET.NMS_POLICY.THRESH2 != 0.4:
            #cc.append('rnTh2{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.THRESH2))
        #if cfg.MODEL.RETINANET.NMS_POLICY.COMPOSE_FINAL_RERANK:
            #cc.append('rnRerank')
            #if cfg.MODEL.RETINANET.NMS_POLICY.COMPOSE_FINAL_RERANK_TYPE != 'nms':
                #cc.append(cfg.MODEL.RETINANET.NMS_POLICY.COMPOSE_FINAL_RERANK_TYPE)
        if cfg.INPUT.SMART_RESIZE_ON_MIN_IN_TEST:
            cc.append('smartresize')

        if self.bn_train_mode_test:
            cc.append('trainmode')

        if cfg.MODEL.ROI_HEADS.MIN_DETECTIONS_PER_IMG != 0:
            cc.append('min{}'.format(cfg.MODEL.ROI_HEADS.MIN_DETECTIONS_PER_IMG))

    def get_train_data_loader(self, start_iter):
        #from fcos_core.data import make_data_loader
        from qd.mask.data import make_data_loader
        data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=self.distributed,
            start_iter=start_iter,
        )
        return data_loader

    def get_test_data_loader(self):
        from qd.mask.data import make_data_loader
        loaders = make_data_loader(cfg,
                                  is_train=False,
                                  is_distributed=self.distributed)
        return loaders[0]

    def predict_iter_forward(self, model, inputs):
        with torch.no_grad():
            return model(inputs)

    def get_train_model(self):
        from fcos_core.modeling.detector import build_detection_model
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
        from fcos_core.solver import make_optimizer
        optimizer = make_optimizer(cfg, model)
        return optimizer

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        from fcos_core.solver import make_lr_scheduler
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
            rects = boxlist_to_list_dict(
                box_list,
                self.label_id_to_label,
                extra=0,
                encode_np_fields=['box_features'],
            )
            if any('box_features' in r for r in rects):
                # we need to re-name the key as feature to be compatible with
                # the pre-training logic
                for r in rects:
                    assert 'feature' not in r
                    # if there is on box which has box_features. all others
                    # should have
                    r['zlib_feature'] = r['box_features']
                    del r['box_features']
            for r in rects:
                if 'attr_feature' in r:
                    del r['attr_feature']

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

