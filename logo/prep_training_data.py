import os

from logo import classifier
from qd import tsv_io, qd_common

outdir = "data/brand_output/configs/"
labelmap = "data/brand_output/TaxLogoV1_7_darknet19_448_C_Init.best_model9748_maxIter.75eEffectBatchSize128_bb_only/classifier/add_sports/labelmap.txt"

def get_train_config(dataset_name, version, labelmap,
            det_expid=None, use_region_proposal=True):
    config_file = os.path.join(outdir, "train_{}.yaml".format(dataset_name))
    dataset = tsv_io.TSVDataset(dataset_name)

    if use_region_proposal:
        label_file = classifier.prepare_training_data(det_expid, dataset_name, outdir, gt_split="train",
                version=version, enlarge_bbox=1.5)
    else:
        label_file = dataset.get_data("train", t="label", version=version)

    config = {"train": {
        "tsv": dataset.get_data("train"),
        "label": label_file,
        "labelmap": labelmap
    },
    "val": {
        "tsv": dataset.get_data("test"),
        "label": dataset.get_data("test", t="label", version=version),
        "labelmap": labelmap
    }}
    qd_common.write_to_yaml_file(config, config_file)
    return config_file

if __name__ == "__main__":
    tsv_io.tsv_writer([[get_train_config("brand1048", 4, labelmap)],
            [get_train_config("sports_missingSplit", -1, labelmap, use_region_proposal=False)]],
            os.path.join(outdir, "train.yamllst"))
