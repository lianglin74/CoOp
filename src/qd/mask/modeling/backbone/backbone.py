# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from qd.mask.modeling import registry
from qd.mask.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import mobilenet


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
@registry.BACKBONES.register("R-152-C4")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model

class TimmViTBackbone(nn.Module):
    def __init__(self, model, skip_first, patch_size):
        super().__init__()
        self.module = model
        self.skip_first = skip_first
        self.patch_size = patch_size

    def forward(self, x):
        y = self.module(x)
        if self.skip_first:
            y = y[:, 1:]
        h, w = x.shape[2:]
        h2, w2 = h // 32, w // 32
        y = y.reshape(y.shape[0], h2, w2, y.shape[-1])
        y = y.permute((0, 3, 1, 2))
        return [y]

@registry.BACKBONES.register("timm_vit_base_patch32_384")
def build_timm_backbone(cfg):
    model_name = cfg.MODEL.BACKBONE.CONV_BODY[5:]
    import timm
    model = timm.create_model(
        model_name=model_name,
        pretrained=True,
        output_grid=True,
    )
    if model_name == 'vit_base_patch32_384':
        patch_size = 32
        out_channels = 768
    else:
        raise NotImplementedError
    model = TimmViTBackbone(model, skip_first=True, patch_size=patch_size)
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("MNV2-FPN-RETINANET")
def build_mnv2_fpn_backbone(cfg):
    body = mobilenet.MobileNetV2(cfg)
    in_channels_stage2 = body.return_features_num_channels
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2[1],
            in_channels_stage2[2],
            in_channels_stage2[3],
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(out_channels, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("efficient-det-0")
def build_efficientdet0_backbone(cfg):
    import qd.layers.efficient_det as efficient_det
    efficient_det.g_simple_padding = True
    start_from = cfg.MODEL.BACKBONE.EFFICIENT_DET_START_FROM
    model = efficient_det.EffNetFPN(
        compound_coef=0,
        start_from=start_from)
    return model

@registry.BACKBONES.register("efficient-det")
def build_efficientdet_backbone(cfg):
    import qd.layers.efficient_det as efficient_det
    efficient_det.g_simple_padding = True
    compound = cfg.MODEL.BACKBONE.EFFICIENT_DET_COMPOUND
    start_from = cfg.MODEL.BACKBONE.EFFICIENT_DET_START_FROM
    model = efficient_det.EffNetFPN(
        compound_coef=compound,
        start_from=start_from,
        bifpn_version=cfg.MODEL.BACKBONE.EFFICIENT_DET_BIFPN_VERSION,
    )
    return model

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)

