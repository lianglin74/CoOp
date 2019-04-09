import os
import shutil
import json

import torch

from tagging.scripts import train, pred
from qd import tsv_io, qd_common

torch.manual_seed(2018)

def main():
    config_root = "data/brand_output/configs/"

    config_file = os.path.join(config_root, "train_brand1048_nobg.yaml")
    outdir = "data/brand_output/brand1048_resnet18_nobg_wd/snapshot/"
    train.main(
        [config_file,
        # "--debug",
        "--arch", "resnet18",
        "--pretrained",
        "--weight-decay", "1e-3",
        "-f",
        "--workers", str(32),
        "-b", str(128),
        '--print-freq', str(100),
        '--epochs', "150",
        "--output-dir", outdir])

    config_file = os.path.join(config_root, "train_brand1048_nobg.yaml")
    outdir = "data/brand_output/brand1048_resnet18_nobg_bneval/snapshot/"
    train.main(
        [config_file,
        # "--debug",
        "--arch", "resnet18",
        # '--data_aug',
        # "--fixpartialfeature",
        # "--bn_no_weight_decay",
        "--BatchNormEvalMode",
        "--pretrained",
        # "--lr", "0.001", "--lr-policy", "CONSTANT",
        "-f",
        "--workers", str(32),
        "-b", str(128),
        # "--resume", os.path.join(config_root, "snapshot1/None-0400.pth.tar"),
        '--print-freq', str(100),
        '--epochs', "150",
        "--output-dir", outdir])

def philly_main():
    from qd.philly import philly_upload_dir
    import os.path as op

    src_dirs = []
    dest_dirs = []
    # src_dirs.append('/home/xiaowh/repos/quickdetection/')
    # dest_dirs.append('xiaowh/code/')
    src_dirs.append('/raid/data/brand1048/')
    dest_dirs.append('xiaowh/code/quickdetection/quickdetection/data/')
    for src_dir, dest_dir in zip(src_dirs, dest_dirs):
        philly_upload_dir(src_dir, dest_dir, vc='input', cluster='wu1',
            blob=True)


if __name__ == "__main__":
    qd_common.init_logging()
    main()
    # philly_main()
