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

def prepare_training_data(det_expid, gt_dataset_name, outdir, gt_split="train",
            version=0, pos_iou=0.5, neg_iou=(0.1, 0.3), enlarge_bbox=2.5):
    """
    Merge ground truth bbox with region proposal bbox
    region proposal is annotated with the corresponding class if IoU>pos_iou,
    annotated as __background if max(IoU)<neg_iou
    """
    # generate region proposal
    detpred_file, _ = yolo_predict(full_expid=det_expid, test_data=gt_dataset_name, test_split=gt_split)

    # merge region proposal and ground truth
    outfile = os.path.join(outdir, "region_proposal/{}.{}.gt_rp.{}_{}_{}.tsv".format(
            gt_dataset_name, gt_split, pos_iou, neg_iou[0], neg_iou[1]))
    gt_dataset = TSVDataset(gt_dataset_name)

    rp = DetectionFile(detpred_file)
    rp_candidates = collections.defaultdict(list)  # imgkey: list of bboxes
    class2count = collections.defaultdict(int)  # class: count
    bg_cands = []  # tuple of imgkey, bbox
    num_gt = 0
    num_total_regions = 0

    for imgkey, coded_rects in gt_dataset.iter_data(gt_split, t='label', version=version):
        # HACK
        if imgkey == "http://www.mimifroufrou.com/scentedsalamander/images/Le-Male-2009-Billboard-B.jpg":
            continue

        im_h, im_w = get_hw(imgkey)
        gt_bboxes = json.loads(coded_rects)
        num_gt += len(gt_bboxes)
        for idx in range(len(gt_bboxes)):
            cur_bbox = copy.deepcopy(gt_bboxes[idx])
            cur_bbox["rect"] = int_rect(cur_bbox["rect"], enlarge_factor=1.0, im_h=im_h, im_w=im_w)
            enlarged_rect = int_rect(cur_bbox["rect"], enlarge_factor=enlarge_bbox, im_h=im_h, im_w=im_w)
            overlaps = [calculate_iou(enlarged_rect, gtbox["rect"]) for i, gtbox in enumerate(gt_bboxes) if i!=idx]
            # enlarge bbox only if it does not overlap other boxes
            if len(overlaps) > 0 and max(overlaps) < neg_iou[0]:
                cur_bbox["rect"] = enlarged_rect
            if not is_valid_rect(cur_bbox["rect"]):
                print("invalid rect")
                continue
            rp_candidates[imgkey].append(cur_bbox)
            num_total_regions += 1

        rp_bboxes = rp[imgkey]
        for b in rp_bboxes:
            new_rect = int_rect(b["rect"], enlarge_factor=enlarge_bbox, im_h=im_h, im_w=im_w)
            b["rect"] = new_rect
            if not is_valid_rect(b["rect"]):
                continue
            overlaps = [calculate_iou(new_rect, gtbox["rect"]) for gtbox in gt_bboxes]
            sorted_overlaps = sorted([(i, v) for i, v in enumerate(overlaps)], key=lambda t: t[1], reverse=True)
            if len(overlaps) == 0:
                continue
            max_iou_idx, max_iou = sorted_overlaps[0]

            if max_iou >= neg_iou[0] and max_iou <= neg_iou[1]:
                # background candidate
                b["class"] = constants.BACKGROUND_LABEL
                bg_cands.append((imgkey, b))
            elif max_iou > pos_iou:
                if len(sorted_overlaps)>1 and sorted_overlaps[1][1]>pos_iou:
                    # skip if the region covers >1 instances
                    continue
                b["class"] = gt_bboxes[sorted_overlaps[0][0]]["class"]
                rp_candidates[imgkey].append(b)
                num_total_regions += 1

    bg_cands = sorted(bg_cands, key=lambda t: t[1]["obj"], reverse=True)
    # skip top 1% to avoid false negative
    bg_lidx = int(0.01 * len(bg_cands))
    bg_ridx = min(len(bg_cands), int(bg_lidx+num_total_regions*2))
    for i in range(bg_lidx, bg_ridx):
        k, b = bg_cands[i]
        rp_candidates[k].append(b)
    print("added #background: {}, #gt: {}, #proposal: {}".format(bg_ridx-bg_lidx, num_gt, num_total_regions-num_gt))

    def gen_labels():
        for imgkey, coded_rects in gt_dataset.iter_data(gt_split, t='label', version=version):
            yield imgkey, json.dumps(rp_candidates[imgkey], separators=(',', ':'))

    tsv_writer(gen_labels(), outfile)
    return outfile


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

    det_file, _ = yolo_predict(full_expid=det_expid, test_data=gt_dataset_name, test_split=split)
    kwargs = {"det_file": det_file, "gt_dataset_name": gt_dataset_name, "split": split,
            "version": version, "iou_range": (0.0, 0.1), "conf_range": (0.0, 1.0),
            "enlarge_bbox": 1.0}
    # key2labels = gen_background_labels(det_file, gt_dataset_name, split, version,
    #         iou_range=(0.0, 0.3), conf_range=(0.0, 1.0), enlarge_bbox=2.0)
    key2labels = gen_background_labels(**kwargs)

    dataset = TSVDataset("brand1048_add_bg")
    def gen_labels():
        for key, _, _ in gt_dataset.iter_data(split):
            labels = key2labels[key] if key in key2labels else []
            yield key, json.dumps(labels, separators=(',', ':'))
    dataset.update_data(gen_labels(), split, 'label',
            generate_info=[["gen_background_labels"], [json.dumps(kwargs)]])

if __name__ == "__main__":
    # outdir = "data/brand_output/configs/"
    # labelmap = "data/brand_output/TaxLogoV1_7_darknet19_448_C_Init.best_model9748_maxIter.75eEffectBatchSize128_bb_only/classifier/add_sports/labelmap.txt"

    # tsv_io.tsv_writer([[get_train_config(outdir, "brand1048", 4, labelmap)],
    #         [get_train_config(outdir, "sports_missingSplit", -1, labelmap, use_region_proposal=False)]],
    #         os.path.join(outdir, "train.yamllst"))

    test()
