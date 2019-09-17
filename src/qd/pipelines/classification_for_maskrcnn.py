import logging
from pprint import pformat
from qd.qd_pytorch import ModelPipeline
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
from qd.qd_maskrcnn import MaskRCNNPipeline
from torch.nn import Module
from maskrcnn_benchmark.config import cfg
from torch import nn
import maskrcnn_benchmark
import torch


def frozen_to_batch_norm2d(module):
    module_output = module
    info = {'num_convert_bn': 0, 'num_convert_gn': 0}
    if isinstance(module,
            maskrcnn_benchmark.layers.batch_norm.FrozenBatchNorm2d):
        module_output = torch.nn.BatchNorm2d(module.bias.size())
        module_output.weight.data = module.weight.data.clone().detach()
        module_output.bias.data = module.bias.data.clone().detach()

        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        info['num_convert_bn'] += 1
    for name, child in module.named_children():
        child, child_info = frozen_to_batch_norm2d(child)
        module_output.add_module(name, child)
        for k, v in child_info.items():
            info[k] += v
    del module
    return module_output, info

class BackboneToPredict(Module):
    def __init__(self, cfg, num_class):
        super(BackboneToPredict, self).__init__()

        assert cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT == 0

        from maskrcnn_benchmark.modeling.backbone import build_backbone
        backbone = build_backbone(cfg)
        self.body = backbone.body

        self.body, info = frozen_to_batch_norm2d(self.body)
        logging.info(pformat(info))


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in list(self.body.modules())[::-1]:
            if hasattr(m, 'weight'):
                feature_size = len(m.weight)
                break
        # in torchvision's resnet model, there is no initialization for fc.
        # Here, we do the same and not initialize it.
        self.fc = nn.Linear(feature_size, num_class)

    def forward(self, x):
        x = self.body(x)[-1]
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class ClassificationForMaskRCNN(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super(ClassificationForMaskRCNN, self).__init__(**kwargs)

        # this function will change the parameter in cfg
        MaskRCNNPipeline(**kwargs)

    def _get_model(self, pretrained, num_class):
        model = BackboneToPredict(cfg, num_class)
        logging.info(model)
        assert len(self.basemodel) == 0
        return model

