import collections
import copy
import json
import logging
import os
from shutil import copyfile
import yaml

import _init_paths
from process_tsv import get_img_url
from utils import read_from_file, write_to_file
from utils import search_bbox_in_list, is_valid_bbox, get_max_iou_idx


class DetectionFile(object):
    def __init__(self, predict_file, key_col_idx=0, bbox_col_idx=1, min_conf=0):
        self.tsv_file = predict_file
        self.key_col_idx = key_col_idx
        self.bbox_col_idx = bbox_col_idx
        self.conf_threshold = min_conf

        self._fp = None
        self._keyidx = None

    def __iter__(self):
        self._ensure_keyidx_loaded()
        for key in self._keyidx:
            yield key

    def __contains__(self, key):
        self._ensure_keyidx_loaded()
        return key in self._keyidx

    def __getitem__(self, key):
        self._ensure_keyidx_loaded()
        if key not in self._keyidx:
            return []
        self._fp.seek(self._keyidx[key])
        cols = self._fp.readline().strip().split('\t')
        bboxes = json.loads(cols[self.bbox_col_idx])
        return _thresholding_detection(
                    bboxes, thres_dict=None, display_dict=None, obj_threshold=0,
                    conf_threshold=self.conf_threshold, blacklist=None)

    def _ensure_keyidx_loaded(self):
        self._ensure_tsv_opened()
        if self._keyidx is None:
            self._keyidx = {}
            fpos = 0
            fsize = os.fstat(self._fp.fileno()).st_size
            while fpos != fsize:
                key = self._fp.readline().strip().split('\t')[self.key_col_idx]
                self._keyidx[key] = fpos
                fpos = self._fp.tell()

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'rb')


class GroundTruthConfig(object):
    def __init__(self, config_file, check_format=False):
        self.config_file = config_file
        self._load_config()
        if check_format:
            check_config(config_file)
        self.rootpath = os.path.split(self.config_file)[0]

    def datasets(self):
        return self.config.keys()

    def baselines(self, dataset):
        return [b["name"] for b in self.config[dataset]["baselines"]]

    def gt_file(self, dataset, filtered=False):
        gt_type = "filtered" if filtered else "original"
        fpath = self.config[dataset]["groundtruth"][gt_type]
        return os.path.join(self.rootpath, fpath)

    def baseline_file(self, dataset, name):
        fpath = self._baseline_info(dataset, name)["result"]
        return os.path.join(self.rootpath, fpath)

    def _baseline_info(self, dataset, name):
        for b in self.config[dataset]["baselines"]:
            if b["name"] == name:
                return b
        raise Exception("unknown baseline: {} in {}".format(name, dataset))

    def _load_config(self):
        with open(self.config_file, 'r') as fp:
            self.config = yaml.load(fp)


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
            config["display"])}
    else:
        display_dict = None
    if config["blacklist"]:
        blacklist = set(l[0].lower() for l in read_from_file(
            config["blacklist"]))
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
        if "conf" in b and b["conf"] < conf_threshold:
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
                                 outfile, pos_iou_threshold, neg_iou_threshold,
                                 baseline_conf=0.5, include_labelmap=None):
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
        pos_iou_threshold: IoU with a ground truth to be treated as correct
        neg_iou_threshold: IoU with a verified wrong bbox to be treated as wrong
        baseline_conf: the verified confidence score in baselines,
            i.e., if it is 0.5, all bboxes with conf>=0.5 were verified
    '''
    include_classes = None
    if include_labelmap:
        include_classes = set(l[0].lower() for l in read_from_file(include_labelmap))
    # load existing ground truth labels
    all_gt, _ = _load_all_gt(gt_config_file)
    with open(gt_config_file, 'r') as fp:
        gt_cfg = yaml.load(fp)
    gt_root = os.path.split(gt_config_file)[0]

    num_bbox_to_submit = 0
    num_bbox_pos = 0
    num_bbox_neg = 0
    num_bbox_total = 0
    output_data = []
    for fileinfo in file_info_list:
        # load prediction results
        dataset = fileinfo["dataset"]
        source = fileinfo["source"]
        if dataset not in gt_cfg:
            raise Exception("unknow dataset: {}".format(dataset))
        result = load_result(rootpath, fileinfo)
        logging.info("load prediction results from: {}"
                     .format(fileinfo["result"]))

        # load ground truth and verified baselines
        gt_tsv = DetectionFile(os.path.join(gt_root, gt_cfg[dataset]['groundtruth']["original"]))
        baselines_tsv = []
        if gt_cfg[dataset]["baselines"]:
            for baseline in gt_cfg[dataset]["baselines"]:
                baselines_tsv.append(DetectionFile(os.path.join(gt_root, baseline["result"]), min_conf=baseline_conf))

        for imgkey, bboxes in result.items():
            new_bboxes = []
            for b in bboxes:
                if include_classes and b["class"].lower() not in include_classes:
                    continue
                if not is_valid_bbox(b):
                    logging.error("invalid bbox: {}".format(str(b)))
                    continue

                num_bbox_total += 1
                # if prediction is in ground truth, continue
                if search_bbox_in_list(b, gt_tsv[imgkey], pos_iou_threshold) >= 0:
                    num_bbox_pos += 1
                    continue
                # if prediction is in verified in baselines, continue
                is_verified = False
                for base_tsv in baselines_tsv:
                    if search_bbox_in_list(b, base_tsv[imgkey], neg_iou_threshold) >= 0:
                        is_verified = True
                        break
                if is_verified:
                    num_bbox_neg += 1
                    continue

                new_bboxes.append(b)
            if len(new_bboxes) > 0:
                num_bbox_to_submit += len(new_bboxes)
                image_url = get_img_url(imgkey)
                output_data.append(['_'.join([dataset, source, imgkey]),
                                    json.dumps(new_bboxes), image_url])

    if output_data:
        write_to_file(output_data, outfile)
    logging.info("#total bbox: {}, #verified correct: {}, #verified wrong: {}, #to submit: {}, file: {}"
                 .format(num_bbox_total, num_bbox_pos, num_bbox_neg, num_bbox_to_submit, outfile))
    return num_bbox_to_submit


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


def tune_threshold_for_target(gt_config_file, dataset, res_file, iou_threshold, prec_file):
    """
    prec_file: TSV file of class, target precision
    Returns: threshold dict of label: threshold
    """
    gt_config = GroundTruthConfig(gt_config_file)
    gt_file = gt_config.gt_file(dataset)
    gt = DetectionFile(gt_file)
    res = DetectionFile(res_file)
    thres_dict = {}
    for cols in read_from_file(prec_file):
        thres, prec, recall = _tune_threshold_helper(gt, res, iou_threshold, float(cols[1]), cols[0])
        thres_dict[cols[0]] = thres
    return thres_dict


def _tune_threshold_helper(gt, result, iou_threshold, target_prec, target_class):
    """
    gt: key -> a list of bbox
    result: key -> a list of bbox
    """
    target_class = target_class.lower()
    scores = []
    num_correct = 0
    num_gt = 0
    for imgkey in gt:
        gt_bboxes = [b for b in gt[imgkey] if b["class"].lower() == target_class]
        num_gt += len(gt_bboxes)
        if imgkey not in result:
            continue
        pred_bboxes = [b for b in result[imgkey] if b["class"].lower() == target_class]
        visited = set()
        for b in pred_bboxes:
            idx_list, max_iou = get_max_iou_idx(b, gt_bboxes)
            if max_iou < iou_threshold or all([idx in visited for idx in idx_list]):
                scores.append((0, b["conf"]))
            else:
                for idx in idx_list:
                    if idx not in visited:
                        visited.add(idx)
                        break
                scores.append((1, b["conf"]))

    scores = sorted(scores, key=lambda t:t[1])
    cur_correct = float(sum([p[0] for p in scores]))
    num_pred = len(scores)
    for idx, (truth, score) in enumerate(scores):
        prec = cur_correct / (num_pred-idx)
        if prec >= target_prec:
            return score, prec, cur_correct / num_gt
        cur_correct -= truth
    return 1.0, 0.0, 0.0


def check_config(config_file):
    """
    Check if the config file structure is valid
    """
    root = os.path.split(config_file)[0]
    with open(config_file) as fp:
        config = yaml.load(fp)

    for d in config:
        models = [os.path.join(root, it["result"]) for it in config[d]["baselines"]]
        models = [DetectionFile(f) for f in models]

        gts = [os.path.join(root, it) for it in config[d]["groundtruth"].values()]
        gts = [DetectionFile(f) for f in gts]

        all_keys = set()
        for i, gt in enumerate(gts):
            if i == 0:
                for key in gt:
                    assert(key not in all_keys)
                    all_keys.add(key)
            else:
                for key in gt:
                    assert(key in all_keys)

        for m in models:
            for key in m:
                assert(key in all_keys)


def add_config_baseline(gt_config_file, dataset, baseline_name, filepath_dict, min_conf=0.5):
    config = GroundTruthConfig(gt_config_file)
    assert(dataset in config.datasets())
    assert("result" in filepath_dict)
    baseline = {"name": baseline_name, "conf_threshold": min_conf}
    for f_type in filepath_dict:
        fpath = filepath_dict[f_type]
        froot, fname = os.path.split(fpath)
        if froot != config.rootpath:
            copyfile(fpath, os.path.join(config.rootpath, fname))
        baseline[f_type] = fname
    if "display" not in baseline:
        baseline["display"] = "displayname.v10.3.tsv"
    new_config_dict = config.config
    new_config_dict[dataset]["baselines"].append(baseline)

    # write new config file
    with open(gt_config_file+".tmp", 'w') as outfile:
        yaml.dump(new_config_dict, outfile, default_flow_style=False)
    os.remove(gt_config_file)
    os.rename(gt_config_file+".tmp", gt_config_file)
