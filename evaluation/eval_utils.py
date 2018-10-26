import collections
import copy
import json
import logging
import os
import yaml

import _init_paths
from process_tsv import get_img_url
from utils import read_from_file, write_to_file
from utils import search_bbox_in_list, is_valid_bbox


def parse_config_info(rootpath, config):
    """Parses config information used to load detection results

    Args:
        rootpath: path to the directory storing files used in config info
        config: a dict containing config setting used to filter the detection
            result, the keys in the dict are:
            result: required, result file name. The first column is image key,
                the second column is json list of bboxes.
            threshold: optional, tsv file with per-class thresholds. The first
                column is term, the second column is confidence threshold.
                Terms not in this file are treated as threshold=1.0
            display: optional, tsv file showing the display name for each term.
                Terms not in this file use their original names.
            conf_threshold: optional, default 0, minimum confidence to pass
                threhosld
            obj_threshold: optional, default 0, minimum objectness to pass
                threshold, will be ignored if the prediction result doesn't
                have "obj" field
            blacklist: optional, file to block some categories
    Returns:
        config dict with absolute path, default values filled
    """
    if "result" not in config:
        raise Exception("can not find result file in config")
    if not os.path.isfile(config["result"]):
        config["result"] = os.path.join(rootpath, config["result"])
    if "threshold" in config and config["threshold"]:
        if not os.path.isfile(config["threshold"]):
            config["threshold"] = os.path.join(rootpath, config["threshold"])
    else:
        config["threshold"] = None

    if "display" in config and config["display"]:
        if not os.path.isfile(config["display"]):
            config["display"] = os.path.join(rootpath, config["display"])
    else:
        config["display"] = None

    if "blacklist" in config and config["blacklist"]:
        if not os.path.isfile(config["blacklist"]):
            config["blacklist"] = os.path.join(rootpath, config["blacklist"])
    else:
        config["blacklist"] = None

    if "conf_threshold" not in config:
        config["conf_threshold"] = 0
    if "obj_threshold" not in config:
        config["obj_threshold"] = 0
    return config


def load_gt(gt_filepath):
    gt = {}
    for cols in read_from_file(gt_filepath):
        imgkey, bboxes = cols[0:2]
        assert(imgkey not in gt)
        gt[imgkey] = json.loads(bboxes)
    return gt


def load_result(rootpath, config):
    """Loads detection results

    Args:
        rootpath: path to directory
        config: a dict of thresholding and other setting for detection

    Returns:
        Dict mapping image key to detection results after thresholding and
        displayname applied
    """
    config = parse_config_info(rootpath, config)
    result = {}
    if config["threshold"]:
        threshold_dict = {p[0]: float(p[1]) for p in read_from_file(
            config["threshold"])}
    else:
        threshold_dict = None
    if config["display"]:
        display_dict = {p[0]: p[1] for p in read_from_file(
            config["display"], check_num_cols=2)}
    else:
        display_dict = None
    if config["blacklist"]:
        blacklist = set(l[0].lower() for l in read_from_file(
            config["blacklist"], check_num_cols=1))
    else:
        blacklist = None
    for cols in read_from_file(config["result"]):
        imgkey, bboxes = cols[0:2]
        assert(imgkey not in result)
        result[imgkey] = _thresholding_detection(
                            json.loads(bboxes),
                            threshold_dict, display_dict,
                            config["obj_threshold"], config["conf_threshold"],
                            blacklist)
    return result


def _thresholding_detection(bbox_list, thres_dict, display_dict,
                            obj_threshold, conf_threshold, blacklist):
    res = []
    for b in bbox_list:
        if "obj" in b and b["obj"] < obj_threshold:
            continue
        if b["conf"] < conf_threshold:
            continue
        term = b["class"]
        if blacklist and term.lower() in blacklist:
            continue
        if thres_dict and term in thres_dict and b["conf"] < thres_dict[term]:
            continue
        if display_dict and term in display_dict:
            b['class'] = display_dict[term]
        res.append(b)
    return res


def _load_all_gt(gt_config_file):
    gt_bboxes = {}
    gt_files = {}
    gt_root = os.path.split(gt_config_file)[0]
    with open(gt_config_file, 'r') as fp:
        gt_cfg = yaml.load(fp)

    for dataset_name, dataset in gt_cfg.items():
        gt_file = os.path.join(gt_root, dataset['groundtruth']["original"])
        gt_files[dataset_name] = gt_file
        gt_bboxes[dataset_name] = load_gt(gt_file)
    return gt_bboxes, gt_files


def filter_gt(gt_config_file, bbox_iou, blacklist=None,
              override_min_conf=None):
    """
    Remove ground truth that does not appear in any of the baseline outputs
    Args:
        gt_config_file: filepath to the config yaml file
        bbox_iou: the IoU threshold to consider two bboxes as the same
        blacklist: file to block some categories
        override_min_conf: default None, if specified, the conf_threshold and
            per-class thresholds in baseline_info will be ignored, all gt will
            be filtered using this number
    """
    gt_root = os.path.split(gt_config_file)[0]
    with open(gt_config_file, 'r') as fp:
        gt_cfg = yaml.load(fp)
    filtered_dirpath = os.path.join(gt_root, "filtered")
    if not os.path.exists(filtered_dirpath):
        os.mkdir(filtered_dirpath)

    for dataset_name, dataset in gt_cfg.items():
        out_gt_file = os.path.join(gt_root, dataset["groundtruth"]["filtered"])
        all_gt_file = os.path.join(gt_root, dataset["groundtruth"]["original"])
        all_gt_ids = [cols[0] for cols in read_from_file(all_gt_file)]
        all_gt = {cols[0]: json.loads(cols[1]) for cols in
                  read_from_file(all_gt_file)}
        cur_gt = collections.defaultdict(list)
        for baseline_info in dataset['baselines']:
            baseline = baseline_info["name"]
            if override_min_conf:
                baseline_info["conf_threshold"] = override_min_conf
                if "threshold" in baseline_info:
                    del baseline_info["threshold"]
            if blacklist:
                baseline_info["blacklist"] = blacklist
            filtered_res_file = os.path.join(
                filtered_dirpath, os.path.split(baseline_info["result"])[-1])
            result = load_result(gt_root, baseline_info)
            filtered_res_data = [[k, json.dumps(result[k])] for k in result]
            write_to_file(filtered_res_data, filtered_res_file)
            for imgkey, bbox_list in result.items():
                for b in bbox_list:
                    # skip if bbox is already in current ground truth
                    if search_bbox_in_list(b, cur_gt[imgkey], bbox_iou) >= 0:
                        continue
                    # add bbox to ground truth if it's correct
                    if search_bbox_in_list(b, all_gt[imgkey], bbox_iou) >= 0:
                        cur_gt[imgkey].append(b)
        cur_gt_data = [[imgkey, json.dumps(cur_gt[imgkey])]
                       for imgkey in all_gt_ids]
        write_to_file(cur_gt_data, out_gt_file)
        logging.info("filter ground truth in {}: #_before: {}, #_after: {}"
                     .format(dataset_name,
                             sum(len(v) for k, v in all_gt.items()),
                             sum(len(v) for k, v in cur_gt.items())))


def process_prediction_to_verify(gt_config_file, rootpath, file_info_list,
                                 outfile, bbox_matching_iou):
    '''
    Processes prediction results for human verification
    Args:
        gt_config_file: path to ground truth config yaml file
        rootpath: path to the prediction results folder
        file_info_list: list of dict, including
            dataset: the dataset name on which detector is run
            source: the detection model name
            result: tsv file with image_key, prediction_bboxes_list
            conf_threhold: optional, float, default is 0
            display: optional, tsv file of term, display name
        outfile: tsv file with datasetname_key, bboxes, image_url
    '''
    # load existing ground truth labels
    all_gt, _ = _load_all_gt(gt_config_file)

    num_bbox_to_submit = 0
    num_bbox_total = 0
    output_data = []
    for fileinfo in file_info_list:
        dataset = fileinfo["dataset"]
        source = fileinfo["source"]
        if dataset not in all_gt:
            raise Exception("unknow dataset: {}".format(dataset))
        result = load_result(rootpath, fileinfo)
        logging.info("load prediction results from: {}"
                     .format(fileinfo["result"]))
        for imgkey, bboxes in result.items():
            num_bbox_total += len(bboxes)
            new_bboxes = []
            for b in bboxes:
                if not is_valid_bbox(b):
                    logging.error("invalid bbox: {}".format(str(b)))
                    continue
                # if prediction is not in ground truth, submit to verify
                if search_bbox_in_list(b, all_gt[dataset][imgkey],
                                       bbox_matching_iou) < 0:
                    new_bboxes.append(b)
            if len(new_bboxes) > 0:
                num_bbox_to_submit += len(new_bboxes)
                image_url = get_img_url(imgkey)
                output_data.append(['_'.join([dataset, source, imgkey]),
                                    json.dumps(new_bboxes), image_url])

    write_to_file(output_data, outfile)
    logging.info("#total bbox: {}, #to submit: {}, file: {}"
                 .format(num_bbox_total, num_bbox_to_submit, outfile))


def merge_gt(gt_config_file, res_files, bbox_matching_iou):
    '''
    Processes human evaluation results to add into existing ground truth labels
    '''
    # load old ground truth labels
    all_gt, gt_files = _load_all_gt(gt_config_file)

    # merge new results into ground truth
    added_count = collections.defaultdict(int)
    for res_file in res_files:
        for parts in read_from_file(res_file, check_num_cols=2):
            assert(int(parts[0]) in [1, 2, 3])
            # consensus yes
            if int(parts[0]) == 1:
                for task in json.loads(parts[1]):
                    dataset, source, imgkey = task["image_key"].split('_', 2)
                    existing_bboxes = all_gt[dataset][imgkey]
                    for new_bbox in task["bboxes"]:
                        if search_bbox_in_list(new_bbox, existing_bboxes,
                                               bbox_matching_iou) < 0:
                            existing_bboxes.append(new_bbox)
                            added_count[dataset] += 1

    for dataset, count in added_count.items():
        outfile = gt_files[dataset]
        logging.info("added #gt {} to: {}".format(count, outfile))
        temp = [[k, json.dumps(all_gt[dataset][k])] for k in all_gt[dataset]]
        if os.path.isfile(outfile):
            if os.path.isfile(outfile+".old"):
                os.remove(outfile+".old")
            os.rename(outfile, outfile+".old")
        write_to_file(temp, outfile)


def populate_pred(infile, outfile):
    map_add = {"man": "person", "woman": "person"}
    count_add = 0
    with open(outfile, 'w') as fout:
        for cols in read_from_file(infile, check_num_cols=2):
            bboxes = json.loads(cols[1])
            to_add = []
            for b in bboxes:
                term = b["class"].lower()
                if term in map_add:
                    tmp = copy.deepcopy(b)
                    tmp["class"] = map_add[term]
                    if search_bbox_in_list(tmp, bboxes, 0.8) < 0:
                        to_add.append(tmp)
                        count_add += 1
            fout.write("{}\t{}\n".format(cols[0], json.dumps(bboxes + to_add)))
    print("added {} bboxes".format(count_add))


def tune_threshold(gt, result, iou_threshold, target_prec, target_class):
    target_class = target_class.lower()
    scores = []
    num_correct = 0
    for imgkey in result:
        for bbox in result[imgkey]:
            if bbox["class"].lower() == target_class:
                if search_bbox_in_list(bbox, gt[imgkey], iou_threshold) >= 0:
                    scores.append((1, bbox["conf"]))
                    num_correct += 1
                else:
                    scores.append((0, bbox["conf"]))
    scores = sorted(scores, key=lambda t:t[1])
    cur_correct = float(num_correct)
    num_total = len(scores)
    for idx, (truth, score) in enumerate(scores):
        prec = cur_correct / (num_total-idx)
        if prec >= target_prec:
            return score, prec, cur_correct / num_correct
        cur_correct -= truth
    return 1.0, 0.0, 0.0
