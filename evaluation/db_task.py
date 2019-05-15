import collections
try:
    # Python 2/3 compatibility
    import izip as zip
except ImportError:
    pass
import json
import logging
import os
import os.path as op

from qd.qd_common import calculate_iou
from qd.tsv_io import TSVDataset, tsv_reader
from qd.db import BoundingBoxVerificationDB

CFG_IOU = "iou"
CFG_REQUIRED_FIELDS = "required"
TASK_CONFIGS = {
    "uhrs_tag_verification": {CFG_IOU: 0, CFG_REQUIRED_FIELDS: ["class"]},
    "uhrs_bounding_box_verification": {CFG_IOU: 0.5, CFG_REQUIRED_FIELDS: ["class", "rect"]},
    "uhrs_logo_verification": {CFG_IOU: 0.5, CFG_REQUIRED_FIELDS: ["class", "rect"]}
}

def submit_to_verify(gt_dataset_name, gt_split, pred_file, collection_name,
            conf_thres=0, gt_version=-1, is_urgent=False):
    """ Submits UHRS verification task to qd database
    Args:
        gt_dataset_name: name of the ground truth dataset
        gt_split: split of the ground truth dataset
        pred_file: TSV file, columns are image_key, json list of dict.
                For detection results, dict must contain "class", "rect".
                For tag reuslts, dict must contain "class".
        collection_name: choose from
                uhrs_tag_verification -> for tagging
                uhrs_bounding_box_verification -> for general OD
                uhrs_logo_verification -> for logo detection
        conf_thres: confidence threshold of prediction results
    """
    cfg = TASK_CONFIGS[collection_name]
    priority_tier = BoundingBoxVerificationDB.urgent_priority_tier if is_urgent else 2
    gt_dataset = TSVDataset(gt_dataset_name)

    key2pred = {p[0]: json.loads(p[1]) for p in tsv_reader(pred_file)}
    num_total_bboxes = 0
    num_verify = 0
    all_bb_tasks = []

    gt_iter = gt_dataset.iter_data(gt_split, t='label', version=gt_version)
    url_iter = gt_dataset.iter_data(gt_split, t='key.url', version=gt_version)

    for gt_parts, url_parts in zip(gt_iter, url_iter):
        assert(gt_parts[0] == url_parts[0])
        key = gt_parts[0]
        if key not in key2pred:
            continue
        gt_bboxes = json.loads(gt_parts[1])
        url = url_parts[1]
        for bbox in key2pred[key]:
            if not all([k in bbox for k in cfg[CFG_REQUIRED_FIELDS]]):
                raise ValueError("missing fields in bbox: {}. Must include: {}"
                            .format(str(bbox), str(cfg[CFG_REQUIRED_FIELDS])))
            num_total_bboxes += 1
            # NOTE: if exists a gt bbox, s.t., "class" is equal in case-insensitive mode,
            # AND "rect" IoU > thres, then the bbox will be treated correct, and will NOT
            # be verified
            is_in_gt = False
            for gt_bbox in gt_bboxes:
                if bbox["class"].lower() == gt_bbox["class"].lower() and \
                            (cfg[CFG_IOU] == 0 or calculate_iou(bbox["rect"], gt_bbox["rect"]) > cfg[CFG_IOU]):
                    is_in_gt = True
                    break
            if is_in_gt:
                continue

            if "conf" not in bbox or bbox["conf"] >= conf_thres:
                num_verify += 1
                if "rect" not in bbox:
                    bbox["rect"] = "DUMMY"  # for tag results
                all_bb_tasks.append({
                    "url": url,
                    "split": gt_split,
                    "key": key,
                    "data": gt_dataset_name,
                    'priority_tier': priority_tier,
                    'priority': 1,
                    "rect": bbox
                })

    logging.info("#total bboxes: {}, #in gt: {}, #submit: {}"
            .format(num_total_bboxes, num_total_bboxes-num_verify, num_verify))
    cur_db = BoundingBoxVerificationDB(db_name = 'qd', collection_name = collection_name)
    cur_db.request_by_insert(all_bb_tasks)

def download_merge_correct_to_gt(gt_dataset_name, gt_split, collection_name):
    # query completed results
    cur_db = BoundingBoxVerificationDB(db_name = 'qd', collection_name = collection_name)
    data_split_to_key_rects, all_id = cur_db.get_completed_uhrs_result(
            extra_match={'data': gt_dataset_name, 'split': gt_split})
    key_rects = data_split_to_key_rects[(gt_dataset_name, gt_split)]

    # get bboxes verified as correct
    key2rects = collections.defaultdict(list)
    for key, rect in key_rects:
        if is_uhrs_consensus_correct(rect["uhrs"]):
            key2rects[key].append(rect)
    def gen_correct_labels():
        for key in key2rects:
            yield key, key2rects[key]

    # merge correct bboxes to ground truth
    from evaluation.eval_utils import add_label_to_dataset
    dataset = TSVDataset(gt_dataset_name)
    add_label_to_dataset(dataset, gt_split, TASK_CONFIGS[collection_name][CFG_IOU],
                         gen_correct_labels(), label_key_type="key")

def is_uhrs_consensus_correct(uhrs_res):
    """ From UHRS feedback, "1" -> Yes, "2" -> No, "3" -> Can't judge
    Consensus correct means more people select "1" than "2"
    """
    if "1" in uhrs_res:
        if "2" not in uhrs_res and uhrs_res["1"] > 0:
            return True
        elif "2" in uhrs_res and uhrs_res["1"] > uhrs_res["2"]:
            return True
    return False
