import os
import shutil
import json

import torch

from tagging.scripts import train, pred
from qd import tsv_io, qd_common

torch.manual_seed(2018)

def test():
    config_root = "data/brand_output/configs/"

    config_file = os.path.join(config_root, "train_brand1048_nobg.yaml")
    outdir = "data/brand_output/test/snapshot/"
    train.main(
        [config_file,
        # "--debug",
        "--arch", "resnet18",
        '--data_aug', '0',
        '--enlarge_bbox', '2',
        "--bn_no_weight_decay",
        '--weight-decay', '1e-3',
        "--pretrained",
        '--balance_class',
        "-f",
        "--workers", str(64),
        "-b", str(256),
        # "--resume", os.path.join(config_root, "snapshot1/None-0400.pth.tar"),
        '--print-freq', str(1),
        '--epochs', "1",
        "--output-dir", outdir])

def main():
    config_root = "data/brand_output/configs/"

    # config_file = os.path.join(config_root, "train_brand1048_nobg.yaml")
    # outdir = "data/brand_output/brand1048_resnet18_nobg_aug/snapshot4/"
    # train.main(
    #     [config_file,
    #     # "--debug",
    #     "--arch", "resnet18",
    #     '--data_aug', '4',
    #     '--enlarge_bbox', '2',
    #     "--bn_no_weight_decay",
    #     '--weight-decay', '1e-3',
    #     "--pretrained",
    #     # '--balance_class',
    #     "-f",
    #     "--workers", str(64),
    #     "-b", str(256),
    #     # "--resume", os.path.join(config_root, "snapshot1/None-0400.pth.tar"),
    #     '--print-freq', str(100),
    #     '--epochs', "120",
    #     "--output-dir", outdir])

    config_file = os.path.join(config_root, "train_brand1048_nobg.yaml")
    outdir = "data/brand_output/brand1048_resnet18_nobg_fixpartial/snapshot/"
    train.main(
        [config_file,
        # "--debug",
        "--arch", "resnet18",
        '--fixpartialfeature',
        '--data_aug', '0',
        '--enlarge_bbox', '2',
        "--bn_no_weight_decay",
        '--weight-decay', '1e-3',
        "--pretrained",
        # '--balance_class',
        "-f",
        "--workers", str(64),
        "-b", str(256),
        # "--resume", os.path.join(config_root, "snapshot1/None-0400.pth.tar"),
        '--print-freq', str(100),
        '--epochs', "120",
        "--output-dir", outdir])

    config_file = os.path.join(config_root, "train_brand1048_nobg.yaml")
    outdir = "data/brand_output/brand1048_resnet18_nobg_ccs/snapshot_0.5/"
    train.main(
        [config_file,
        # "--debug",
        "--arch", "resnet18",
        '--ccs_loss_param', '0.5',
        '--data_aug', '0',
        '--enlarge_bbox', '2',
        "--bn_no_weight_decay",
        '--weight-decay', '1e-3',
        "--pretrained",
        # '--balance_class',
        "-f",
        "--workers", str(64),
        "-b", str(256),
        # "--resume", os.path.join(config_root, "snapshot1/None-0400.pth.tar"),
        '--print-freq', str(100),
        '--epochs', "120",
        "--output-dir", outdir])

    config_file = os.path.join(config_root, "train_brand1048_nobg.yaml")
    outdir = "data/brand_output/brand1048_resnet18_nobg_ccs/snapshot_1.0/"
    train.main(
        [config_file,
        # "--debug",
        "--arch", "resnet18",
        '--ccs_loss_param', '1.0',
        '--data_aug', '0',
        '--enlarge_bbox', '2',
        "--bn_no_weight_decay",
        '--weight-decay', '1e-3',
        "--pretrained",
        # '--balance_class',
        "-f",
        "--workers", str(64),
        "-b", str(256),
        # "--resume", os.path.join(config_root, "snapshot1/None-0400.pth.tar"),
        '--print-freq', str(100),
        '--epochs', "120",
        "--output-dir", outdir])

    config_file = os.path.join(config_root, "train_brand1048_nobg.yaml")
    outdir = "data/brand_output/brand1048_resnet18_nobg_ccs/snapshot_2.0/"
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
        # '--balance_class',
        "-f",
        "--workers", str(64),
        "-b", str(256),
        # "--resume", os.path.join(config_root, "snapshot1/None-0400.pth.tar"),
        '--print-freq', str(100),
        '--epochs', "120",
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
    # test()
    main()
    # philly_main()
