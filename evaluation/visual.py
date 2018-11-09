import json
import os
import uuid

from eval_utils import DetectionFile, GroundTruthConfig, _thresholding_detection
from generate_task import write_task_file, pack_task_with_honey_pot
from uhrs import UhrsTaskManager
from utils import read_from_file, write_to_file, get_max_iou_idx
import _init_paths
from process_tsv import get_img_url


def get_wrong_pred(pred_file, gt_file, labelmap=None, outfile=None, min_conf=0.5, iou=0.5):
    if labelmap:
        target_classes = set(cols[0].lower() for cols in read_from_file(labelmap))
    else:
        target_classes = None
    pred = DetectionFile(pred_file, min_conf=min_conf)
    gt = DetectionFile(gt_file)
    all_wrong_pred = []  # imgkey, pred_bboxes, gt_bboxes
    num_gt, num_pred, num_false_pos, num_missing = 0, 0, 0, 0
    for imgkey in pred:
        pred_bboxes = _thresholding_detection(pred[imgkey], thres_dict=None, display_dict=None, obj_threshold=0,
                            conf_threshold=min_conf, blacklist=None, labelmap=target_classes)
        gt_bboxes = _thresholding_detection(gt[imgkey], thres_dict=None, display_dict=None, obj_threshold=0,
                            conf_threshold=0, blacklist=None, labelmap=target_classes)
        visited = set()
        false_pos = []
        missing = []
        for b in pred_bboxes:
            # false positive detection
            idx_list, max_iou = get_max_iou_idx(b, gt_bboxes)
            if max_iou < iou or all([idx in visited for idx in idx_list]):
                false_pos.append(b)
            else:
                for idx in idx_list:
                    if idx not in visited:
                        visited.add(idx)
                        break
        # missing label
        if len(visited) < len(gt_bboxes):
            for i in range(len(gt_bboxes)):
                if i not in visited:
                    missing.append(gt_bboxes[i])

        num_gt += len(gt_bboxes)
        num_pred += len(pred_bboxes)
        num_false_pos += len(false_pos)
        num_missing += len(missing)
        if len(false_pos)>0 or len(missing)>0:
            all_wrong_pred.append([imgkey, json.dumps(false_pos), json.dumps(missing), json.dumps(pred_bboxes), json.dumps(gt_bboxes)])

    print("#gt: {}, #pred: {}, #correct: {}, precision: {}, recall: {}".format(
            num_gt, num_pred, num_pred-num_false_pos,
            float(num_pred-num_false_pos)/num_pred, float(num_pred-num_false_pos)/num_gt))
    if all_wrong_pred and outfile:
        write_to_file(all_wrong_pred, outfile)
    return all_wrong_pred


def manual_check(gt_config_file, source_name, labelmap, dirpath, min_conf=0.5, iou=0.5):
    """
    Manually check false positive prediction via UHRS UI
    """
    config = GroundTruthConfig(gt_config_file)
    task_data = []
    for dataset in config.datasets():
        gt_file = config.gt_file(dataset)
        pred_file = config.baseline_file(dataset, source_name)
        wrong_pred = get_wrong_pred(pred_file, gt_file, labelmap, min_conf=min_conf, iou=iou)
        for it in wrong_pred:
            key = it[0]
            image_url = get_img_url(key)
            key = '_'.join([dataset, source_name, key])
            bboxes = json.loads(it[1])
            for bbox in bboxes:
                task_data.append({"uuid": str(uuid.uuid4()), "image_key": key,
                                  "image_url": image_url,
                                  "objects_to_find": bbox["class"], "bboxes": [bbox]})
    outdata = []
    start = 0
    while start < len(task_data):
        end = min(start + 20, len(task_data))
        outdata.append(task_data[start:end])
        start = end

    task_hitapp = "internal_verify_box"
    task_id_name_log = os.path.join(dirpath, 'task_id_name_log')
    if os.path.isfile(task_id_name_log):
        os.remove(task_id_name_log)
    uhrs_client = UhrsTaskManager(task_id_name_log)
    upload_dir = os.path.join(dirpath, 'upload')
    filename = "manual_false_pos_{}".format(source_name)
    write_task_file(outdata, os.path.join(upload_dir, filename))
    uhrs_client.upload_tasks_from_folder(task_hitapp, upload_dir,
                                         prefix=filename, num_judges=1)
