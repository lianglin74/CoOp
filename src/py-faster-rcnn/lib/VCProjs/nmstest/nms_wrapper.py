# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from gpu_nms import gpu_nms;
from cpu_nms import cpu_nms;


def nms(dets, thresh, force_cpu=False):
    if dets.shape[0] == 0:
        return []
    if not force_cpu:
        return gpu_nms(dets, thresh, device_id=0)
    else:
        return cpu_nms(dets, thresh)
