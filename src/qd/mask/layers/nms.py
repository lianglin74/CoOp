# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
import torch
import torchvision


def nms(*args):
    from qd.mask.config import cfg
    if cfg.MODEL.USE_TORCHVISION_NMS:
        _nms = torchvision.ops.nms
    else:
        if torch._C._get_tracing_state():
            _nms = torchvision.ops.nms
        else:
            from fcos_core import _C
            _nms = _C.nms
    return _nms(*args)

def ml_nms(*args, **kwargs):
    from fcos_core import _C
    return _C.ml_nms(*args, **kwargs)
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
