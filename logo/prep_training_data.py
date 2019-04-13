import collections
import copy
import json
import os

import classifier, constants
from qd import tsv_io, qd_common
from qd.qd_common import calculate_iou, write_to_yaml_file, init_logging, ensure_directory, int_rect, is_valid_rect
from qd.tsv_io import TSVDataset, TSVFile, tsv_writer
from qd.yolotrain import yolo_predict

class DetectionFile(object):
    """ Wrapper of bbox prediction file, columns are image_key, json list of bbox
    """
    def __init__(self, tsv_file):
        self.tsv = TSVFile(tsv_file)
        self._key2idx = {self.tsv.seek_first_column(i): i for i in range(self.tsv.num_rows())}

    def __getitem__(self, key):
        idx = self._key2idx[key]
        parts = self.tsv.seek(idx)
        assert(parts[0] == key)
        return json.loads(parts[1])

def get_hw(dataset, split, key):
    parts = dataset.seek_by_key(key, split, t="hw")
    assert(len(parts) == 2)
    assert(parts[0] == key)
    nums = parts[1].split(' ')
    assert(len(nums) == 2)
    return int(nums[0]), int(nums[1])

def gen_background_labels(det_file, gt_dataset_name, split, version,
            iou_range=(0.0, 0.3), conf_range=(0.0, 1.0), enlarge_bbox=1.0):
    rp = DetectionFile(det_file)
    gt_dataset = TSVDataset(gt_dataset_name)

    bg_cands = []
    for imgkey, coded_rects in gt_dataset.iter_data(split, t='label', version=version):
        im_h, im_w = get_hw(gt_dataset, split, imgkey)
        gt_bboxes = json.loads(coded_rects)
        rp_bboxes = rp[imgkey]
        for b in rp_bboxes:
            if not is_valid_rect(b["rect"]):
                continue
            if b["conf"] < conf_range[0] or b["conf"] > conf_range[1]:
                continue

            new_rect = int_rect(b["rect"], enlarge_factor=enlarge_bbox, im_h=im_h, im_w=im_w)
            overlaps = [calculate_iou(new_rect, gtbox["rect"]) for gtbox in gt_bboxes]
            sorted_overlaps = sorted([(i, v) for i, v in enumerate(overlaps)], key=lambda t: t[1], reverse=True)
            if len(overlaps) == 0:
                continue
            max_iou_idx, max_iou = sorted_overlaps[0]

            if max_iou >= iou_range[0] and max_iou <= iou_range[1]:
                # background candidate
                b["class"] = constants.BACKGROUND_LABEL
                bg_cands.append((imgkey, b))
    print("get {} background labels".format(len(bg_cands)))
    key2labels = collections.defaultdict(list)
    for imgkey, b in bg_cands:
        key2labels[imgkey].append(b)
    return key2labels

def gen_proposed_labels(det_file, gt_dataset_name, split, version,
            iou_range=(0.5, 1.0), conf_range=(0.0, 1.0), enlarge_bbox=1.0):
    rp = DetectionFile(det_file)
    gt_dataset = TSVDataset(gt_dataset_name)

    key2labels = collections.defaultdict(list)
    num_total_regions = 0
    for imgkey, coded_rects in gt_dataset.iter_data(split, t='label', version=version):
        gt_bboxes = json.loads(coded_rects)
        rp_bboxes = rp[imgkey]
        for b in rp_bboxes:
            rect = b["rect"]
            if not is_valid_rect(rect):
                continue
            if b["conf"] < conf_range[0] or b["conf"] > conf_range[1]:
                continue
            overlaps = [calculate_iou(rect, gtbox["rect"]) for gtbox in gt_bboxes]
            sorted_overlaps = sorted([(i, v) for i, v in enumerate(overlaps)], key=lambda t: t[1], reverse=True)
            if len(overlaps) == 0:
                continue
            max_iou_idx, max_iou = sorted_overlaps[0]
            if max_iou >= iou_range[0] and max_iou <= iou_range[1]:
                if len(sorted_overlaps)>1 and sorted_overlaps[1][1]>=iou_range[0]:
                    # skip if the region covers >1 instances
                    continue
                b["class"] = gt_bboxes[max_iou_idx]["class"]
                key2labels[imgkey].append(b)
                num_total_regions += 1
    print("get {} positive samples".format(num_total_regions))
    return key2labels


def prepare_training_data(det_expid, gt_dataset_name, split, version, outdataset_name,
            pos_iou_range=(0.5, 1.0), pos_conf_range=(0.0, 1.0),
            neg_iou_range=(0.0, 0.1), neg_conf_range=(0.0, 0.3),
            add_gt=True):
    """
    Merge ground truth bbox with region proposal bbox
    region proposal is annotated with the corresponding class if IoU>pos_iou,
    annotated as __background if max(IoU)<neg_iou
    """
    # generate region proposal
    det_file, _ = yolo_predict(full_expid=det_expid, test_data=gt_dataset_name, test_split=split)

    label_dicts = []
    generate_info = []
    if neg_iou_range and neg_iou_range[0] < neg_iou_range[1]:
        kwargs = {"det_file": det_file, "gt_dataset_name": gt_dataset_name, "split": split,
                "version": version, "iou_range": neg_iou_range, "conf_range": neg_conf_range,
                "enlarge_bbox": 1.0}
        key2labels = gen_background_labels(**kwargs)
        label_dicts.append(key2labels)
        generate_info.append(["gen_background_labels", json.dumps(kwargs)])
    if pos_iou_range and pos_iou_range[0] < pos_iou_range[1]:
        kwargs = {"det_file": det_file, "gt_dataset_name": gt_dataset_name, "split": split,
                "version": version, "iou_range": pos_iou_range, "conf_range": pos_conf_range,
                "enlarge_bbox": 1.0}
        key2labels = gen_proposed_labels(**kwargs)
        label_dicts.append(key2labels)
        generate_info.append(["gen_proposed_labels", json.dumps(kwargs)])

    gt_dataset = TSVDataset(gt_dataset_name)
    dataset = TSVDataset(outdataset_name)
    def gen_labels():
        for key, coded_rects in gt_dataset.iter_data(split, 'label', version=version):
            if add_gt:
                labels = json.loads(coded_rects)
            else:
                labels = []
            for key2labels in label_dicts:
                labels.extend(key2labels[key])
            yield key, json.dumps(labels, separators=(',', ':'))
    dataset.update_data(gen_labels(), split, 'label', generate_info=generate_info)
    print("generate new label file: {}".format(dataset.get_data(split, 'label', version=-1)))


def get_train_config(outdir, dataset_name, version, labelmap,
            det_expid=None, use_region_proposal=True):
    from logo import classifier
    config_file = os.path.join(outdir, "train_{}.yaml".format(dataset_name))
    dataset = TSVDataset(dataset_name)

    if use_region_proposal:
        label_file = prepare_training_data(det_expid, dataset_name, outdir, gt_split="train",
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

def test():
    det_expid = "TaxLogoV1_7_darknet19_448_C_Init.best_model9748_maxIter.75eEffectBatchSize128_bb_only"
    gt_dataset_name = "brand1048"
    gt_dataset = TSVDataset(gt_dataset_name)
    split = "train"
    version = 4
    outdataset_name = "brand1048_add_bg"

    prepare_training_data(det_expid, gt_dataset_name, split, version, outdataset_name,
            pos_iou_range=None, pos_conf_range=(0.0, 1.0),
            neg_iou_range=(0.0, 0.1), neg_conf_range=(0.2, 0.3),
            add_gt=True)

if __name__ == "__main__":
    # outdir = "data/brand_output/configs/"
    # labelmap = "data/brand_output/TaxLogoV1_7_darknet19_448_C_Init.best_model9748_maxIter.75eEffectBatchSize128_bb_only/classifier/add_sports/labelmap.txt"

    # tsv_io.tsv_writer([[get_train_config(outdir, "brand1048", 4, labelmap)],
    #         [get_train_config(outdir, "sports_missingSplit", -1, labelmap, use_region_proposal=False)]],
    #         os.path.join(outdir, "train.yamllst"))

    test()
