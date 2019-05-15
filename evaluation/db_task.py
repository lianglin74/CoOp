try:
    # Python 2/3 compatibility
    import izip as zip
except ImportError:
    pass
import json
import logging
import os
import os.path as op

from evaluation.utils import search_bbox_in_list
from qd.tsv_io import TSVDataset, tsv_reader
from qd.db import BoundingBoxVerificationDB

CFG_IOU = "iou"
CFG_REQUIRED_FIELDS = "required"
TASK_CONFIGS = {
    "uhrs_tag_verification": {CFG_IOU: 0, CFG_REQUIRED_FIELDS: ["class"]},
    "uhrs_bounding_box_verification": {CFG_IOU: 0.5, CFG_REQUIRED_FIELDS: ["class", "rect"]},
    "uhrs_logo_verification": {CFG_IOU: 0.5, CFG_REQUIRED_FIELDS: ["class", "rect"]}
}

def submit_task(gt_dataset_name, gt_split, pred_file, collection_name,
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
            if search_bbox_in_list(bbox, gt_bboxes, cfg[CFG_IOU]) < 0:
                if "conf" not in bbox or bbox["conf"] >= thres:
                    num_verify += 1
                    all_bb_tasks.append({
                        "url": url,
                        "split": gt_split,
                        "key": key,
                        "data": gt_dataset_name,
                        'priority_tier': priority_tier,
                        'priority': 1,
                        "rect": bbox
                    })

    logging.info("#total bboxes: {}, #submit: {}".format(num_total_bboxes, num_verify))
    cur_db = BoundingBoxVerificationDB(db_name = 'qd', collection_name = collection_name)
    cur_db.request_by_insert(all_bb_tasks)

