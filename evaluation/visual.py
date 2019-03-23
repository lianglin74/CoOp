import json
import os
import uuid

from evaluation.eval_utils import DetectionFile, GroundTruthConfig, _thresholding_detection, merge_gt
from evaluation.generate_task import generate_task_files
from evaluation.analyze_task import analyze_verify_box_task
from evaluation.uhrs import UhrsTaskManager
from evaluation.utils import read_from_file, write_to_file, get_max_iou_idx, list_files_in_dir
from scripts.process_tsv import get_img_url2
from qd.qd_common import ensure_directory
from qd.tsv_io import tsv_reader, tsv_writer, TSVDataset


def get_wrong_pred(pred_file, gt_file, labelmap=None, outfile=None, min_conf=0.5, iou=0.5, gt_conf=0.0):
    if labelmap:
        target_classes = set(cols[0].lower() for cols in read_from_file(labelmap))
    else:
        target_classes = None
    pred = DetectionFile(pred_file, conf_threshold=min_conf, labelmap=target_classes)
    gt = DetectionFile(gt_file, conf_threshold=gt_conf, labelmap=target_classes)

    all_wrong_pred = []  # imgkey, pred_bboxes, gt_bboxes
    num_gt, num_pred, num_false_pos, num_missing = 0, 0, 0, 0
    for imgkey in pred:
        pred_bboxes = pred[imgkey]
        gt_bboxes = gt[imgkey]
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


def manual_check(source_name, gt_config_file, task_dir, min_conf=0.5, iou=0.5, labelmap=None):
    """
    Manually check false positive prediction via UHRS UI
    """
    # prepare working directory
    dirpath = os.path.join(task_dir, source_name)
    ensure_directory(dirpath)

    task_hitapp = "internal_verify_box"
    task_id_name_log = os.path.join(dirpath, 'task_id_name_log')
    upload_dir = os.path.join(dirpath, 'upload')
    ensure_directory(upload_dir)
    download_dir = os.path.join(dirpath, 'download')
    ensure_directory(download_dir)
    res_file = os.path.join(dirpath, "eval_result.tsv")
    if os.path.isfile(task_id_name_log):
        os.remove(task_id_name_log)

    # filter data that need to be verified
    config = GroundTruthConfig(gt_config_file)
    task_data = []
    for dataset in config.datasets():
        gt_file = config.gt_file(dataset)
        pred_file = config.baseline_file(dataset, source_name)
        wrong_pred = get_wrong_pred(pred_file, gt_file, labelmap, min_conf=min_conf, iou=iou)
        for it in wrong_pred:
            key = it[0]
            image_url = get_img_url2(key)
            bboxes = json.loads(it[1])  # false positive
            task_data.append([dataset, json.dumps(bboxes), image_url])

    task_label_file = os.path.join(dirpath, "eval_labels.tsv")
    write_to_file(task_data, task_label_file)
    fname = "manual_check_{}".format(source_name)
    generate_task_files("VerifyBox", task_label_file, None, outbase=os.path.join(upload_dir, fname),
                        num_tasks_per_hit=20, num_hp_per_hit=0)

    uhrs_client = UhrsTaskManager(task_id_name_log)
    uhrs_client.upload_tasks_from_folder(task_hitapp, upload_dir,
                                         prefix=fname, num_judges=1)

    uhrs_client.wait_until_task_finish(task_hitapp)
    uhrs_client.download_tasks_to_folder(task_hitapp, download_dir)
    download_files = list_files_in_dir(download_dir)

    analyze_verify_box_task(
            download_files, "uhrs", res_file, outfile_rejudge=None,
            worker_quality_file=None, min_num_judges_per_hit=1)
    for dataset in config.datasets():
        merge_gt(dataset, gt_config_file, [res_file], 0.8)
