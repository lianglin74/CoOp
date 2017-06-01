"""Parse log for loss."""

from __future__ import division
from __future__ import print_function

from glob import glob
import numpy as np
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))
import deteval

def parse_loss(log_path):
    """Parse caffe loss from log file."""
    with open(log_path) as fin:
        str_buff = fin.read()
    iter_loss = re.findall(r"Iteration\s+(\d+).+loss\s+=\s+([0-9\.]+)", str_buff)
    iters, losses = zip(*iter_loss)
    iters = map(lambda x: float(x), iters)
    losses = map(lambda x: float(x), losses)
    iters = np.array(iters, dtype=np.uint64)
    losses = np.array(losses, dtype=np.float32)
    return iters, losses

def mAP_generator(test_data, job_path):
    out_tsvs = glob(os.path.join(job_path, "test_iter_*.tsv"))
    out_tsv_maps = [(
        int(re.findall("test_iter_(\d+).tsv", out_tsv)[0]), out_tsv)
        for out_tsv in out_tsvs]
    for current_iter, out_tsv in sorted(out_tsv_maps, key=lambda x: x[0]):
        # Get mAP report from outtsv over IoU threshold 0.5.
        report = deteval.eval(test_data, out_tsv, 0.5)
        yield (current_iter, report["map"])
