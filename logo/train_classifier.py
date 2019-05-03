import os
import shutil
import json

import torch

from qd_classfier.scripts import train, pred
from qd import tsv_io, qd_common

torch.manual_seed(2018)

def test():
    config_root = "data/brand_output/configs/"

    config_file = os.path.join(config_root, "train_nobg.yamllst")
    outdir = "data/brand_output/test/snapshot/"
    train.main(
        [config_file,
        # "--debug",
        "--arch", "resnet18",
        '--ccs_loss_param', '2.0',
        '--data_aug', '0',
        '--enlarge_bbox', '2',
        "--bn_no_weight_decay",
        '--weight-decay', '1e-3',
        "--pretrained",
        '--balance_sampler',
        "-f",
        "--workers", str(64),
        "-b", str(2),
        # "--resume", os.path.join(config_root, "snapshot1/None-0400.pth.tar"),
        '--print-freq', str(100),
        '--epochs', "1",
        "--output-dir", outdir])

def philly_main():
    from qd.philly import philly_upload_dir
    import os.path as op

    src_dirs = []
    dest_dirs = []
    # src_dirs.append('/home/xiaowh/repos/quickdetection/')
    # dest_dirs.append('xiaowh/code/')
    src_dirs.append('/raid/data/brand1048')
    dest_dirs.append('xiaowh/data/')
    for src_dir, dest_dir in zip(src_dirs, dest_dirs):
        philly_upload_dir(src_dir, dest_dir, vc='input', cluster='wu1',
            blob=True)


if __name__ == "__main__":
    qd_common.init_logging()
    # philly_main()
