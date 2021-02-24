# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import math
from qd.mask.modeling import registry
from torch import nn


@registry.ROI_BOX_PREDICTOR.register("FastRCNNPredictor")
class FastRCNNPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None

        num_inputs = in_channels

        num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_score = nn.Linear(num_inputs, num_classes)
        num_bbox_reg_classes = 2 if config.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(num_inputs, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        return cls_logit, bbox_pred


@registry.ROI_BOX_PREDICTOR.register("FPNPredictor")
class FPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FPNPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        representation_size = in_channels

        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = 2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)
        if cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_LOSS in ['BCE', 'BCEByPos']:
            c_bias_sigmoid_small = 1. / num_classes
            logging.info('setting bias so that the conf is {}'.format(c_bias_sigmoid_small))
            nn.init.constant_(self.cls_score.bias, -math.log(1. / c_bias_sigmoid_small - 1))

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

@registry.ROI_BOX_PREDICTOR.register("MDFPNPredictor")
class MDFPNPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        representation_size = in_channels
        from qd.qd_common import load_from_yaml_file
        domain_infos = load_from_yaml_file(cfg.MODEL.ROI_BOX_HEAD.DOMAINMAP_FILE)
        all_num = [len(domain_info['labelmap']) + 1 for domain_info in domain_infos]

        self.cls_scores = nn.ModuleList(
            [nn.Linear(representation_size, n) for n in all_num]
        )
        assert cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
        num_bbox_reg_classes = 2
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        for cls_score in self.cls_scores:
            nn.init.normal_(cls_score.weight, std=0.01)
            nn.init.constant_(cls_score.bias, 0)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            x = x.view(x.size(0), -1)
        scores = [m(x) for m in self.cls_scores]
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def make_roi_box_predictor(cfg, in_channels):
    func = registry.ROI_BOX_PREDICTOR[cfg.MODEL.ROI_BOX_HEAD.PREDICTOR]
    return func(cfg, in_channels)
