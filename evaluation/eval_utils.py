import collections
import copy
import json
import logging
import os
from shutil import copyfile
import yaml

from scripts import tsv_io
from scripts.tsv_io import TSVDataset
from scripts.process_tsv import get_img_url2
from scripts.qd_common import write_to_yaml_file
from evaluation.utils import read_from_file, write_to_file
from evaluation.utils import search_bbox_in_list, is_valid_bbox, get_max_iou_idx, get_bbox_matching_map


class DetectionFile(object):
    def __init__(self, predict_file, key_col_idx=0, bbox_col_idx=1, conf_threshold=0,
                 threshold=None, display=None, obj_threshold=0, blacklist=None, **kwargs):
        self.tsv_file = predict_file
        self.key_col_idx = key_col_idx
        self.bbox_col_idx = bbox_col_idx

        self._fp = None
        self._keyidx = None

        self._threshold_dict = None
        self._display_dict = None
        self._blacklist = None
        self._obj_threshold = obj_threshold
        self._conf_threshold = conf_threshold

        if threshold:
            self._threshold_dict = {p[0]: float(p[1]) for p in read_from_file(threshold)}
        if display:
            self._display_dict = {p[0]: p[1] for p in read_from_file(display)}
        if blacklist:
            self._blacklist = set(l[0].lower() for l in read_from_file(blacklist))

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
                    bboxes, thres_dict=self._threshold_dict, display_dict=self._display_dict,
                    obj_threshold=self._obj_threshold, conf_threshold=self._conf_threshold,
                    blacklist=self._blacklist)

    def _ensure_keyidx_loaded(self):
        self._ensure_tsv_opened()
        if self._keyidx is None:
            self._keyidx = collections.OrderedDict()
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
        self.rootpath = os.path.split(os.path.realpath(self.config_file))[0]

    def datasets(self):
        return self.config.keys()

    def baselines(self, dataset):
        res = []
        if self.config[dataset].get("baselines", None):
            for b in self.config[dataset]["baselines"]:
                res.append(b["name"])
        return res

    def gt_dataset(self, dataset):
        gt = self.config[dataset]["groundtruth"]["original"]
        if not isinstance(gt, dict):
            raise Exception("{} is not a valid dataset".format(dataset))
        tsv_dataset = TSVDataset(gt["dataset"])
        # HACK to access dataset on Windows
        if os.name == "nt":
            tsv_dataset._data_root = os.path.join("//vigdgx02/raid_data/", gt["dataset"])
        return tsv_dataset, gt["split"]

    def gt_file(self, dataset, filtered=False):
        gt_type = "filtered" if filtered else "original"
        gt = self.config[dataset]["groundtruth"][gt_type]
        if isinstance(gt, dict):
            tsv_dataset, split = self.gt_dataset(dataset)
            fpath = tsv_dataset.get_data(split, 'label', version=-1)
            return os.path.realpath(fpath)
        else:
            return os.path.join(self.rootpath, gt)

    def baseline_file(self, dataset, name):
        return self.baseline_info(dataset, name)["result"]

    def baseline_info(self, dataset, name):
        for baseline in self.config[dataset]["baselines"]:
            if baseline["name"] == name:
                return parse_config_info(self.rootpath, baseline)
        raise Exception("unknown baseline: {} in {}".format(name, dataset))

    def add_baseline(self, dataset, baseline_name, pred_file, min_conf):
        if not os.path.isabs(pred_file):
            assert(os.path.isfile(os.path.join(self.rootpath, pred_file)))
        else:
            assert(os.path.isfile(pred_file))

        if "baselines" not in self.config[dataset]:
            self.config[dataset]["baselines"] = []
        bases = self.config[dataset]["baselines"]
        assert(isinstance(bases, list))
        # check if baseline already exists
        for b in bases:
            if b["name"] == baseline_name:
                raise Exception("{} already exists".format(baseline_name))
        bases.append(
            {"name": baseline_name, "result": pred_file,
            "conf_threshold": min_conf})
        write_to_yaml_file(self.config, self.config_file)
        self._load_config()

    def _load_config(self):
        with open(self.config_file, 'r') as fp:
            self.config = yaml.load(fp)


def parse_config_info(rootpath, config):
    """Parses config information used to load detection results

    Args:
        rootpath: path to the directory storing files used in config info
        config: a dict containing config setting used to filter the detection
            result. The file path can be relative or absolute. Keys in the dict are:
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
    def get_path(p):
        if os.path.isabs(p):
            return p
        else:
            return os.path.join(rootpath, p)
    if "result" not in config:
        raise Exception("can not find result file in config")
    if not os.path.isfile(config["result"]):
        config["result"] = get_path(config["result"])
    if "threshold" in config:
        if not os.path.isfile(config["threshold"]):
            config["threshold"] = get_path(config["threshold"])

    if "display" in config:
        if not os.path.isfile(config["display"]):
            config["display"] = get_path(config["display"])

    if "blacklist" in config:
        if not os.path.isfile(config["blacklist"]):
            config["blacklist"] = get_path(config["blacklist"])
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


def _thresholding_detection(bbox_list, thres_dict, display_dict, obj_threshold,
                            conf_threshold, blacklist, labelmap=None):
    res = []
    for b in bbox_list:
        if "obj" in b and b["obj"] < obj_threshold:
            continue
        if "conf" in b and b["conf"] < conf_threshold:
            continue
        term = b["class"]
        if blacklist and term.lower() in blacklist:
            continue
        if labelmap and term.lower() not in labelmap:
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
    gt_cfg = GroundTruthConfig(gt_config_file)
    filtered_dirpath = os.path.join(gt_root, "filtered")
    if not os.path.exists(filtered_dirpath):
        os.mkdir(filtered_dirpath)

    for dataset_name in gt_cfg.datasets():
        out_gt_file = gt_cfg.gt_file(dataset_name, filtered=True)
        all_gt_file = gt_cfg.gt_file(dataset_name, filtered=False)
        all_gt_ids = [cols[0] for cols in read_from_file(all_gt_file)]
        all_gt = {cols[0]: json.loads(cols[1]) for cols in
                  read_from_file(all_gt_file)}
        cur_gt = collections.defaultdict(list)
        for baseline in gt_cfg.baselines(dataset_name):
            baseline_info = gt_cfg.baseline_info(dataset_name, baseline)
            if override_min_conf:
                baseline_info["conf_threshold"] = override_min_conf
                if "threshold" in baseline_info:
                    del baseline_info["threshold"]
            if blacklist:
                baseline_info["blacklist"] = blacklist
            filtered_res_file = os.path.join(
                filtered_dirpath, os.path.split(baseline_info["result"])[-1])
            result = DetectionFile(baseline_info["result"], **baseline_info)
            filtered_res_data = [[k, json.dumps(result[k])] for k in result]
            write_to_file(filtered_res_data, filtered_res_file)
            for imgkey in result:
                for b in result[imgkey]:
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


def process_prediction_to_verify(gt_config_file, dataset_name, pred_name, pred_type,
                                 outfile, pos_iou_threshold, neg_iou_threshold,
                                 include_labelmap=None):
    '''
    Processes prediction results for human verification
    Args:
        gt_config_file: path to ground truth config yaml file
        dataset_name: gt_config[dataset_name]
        pred_name: gt_config[dataset_name]["baselines"]["name"]
        pred_type: VerifyImage (tagging results) or VerifyBox(detection results)
        outfile: tsv file with img_info, bboxes, image_url
        pos_iou_threshold: IoU with a ground truth to be treated as correct
        neg_iou_threshold: IoU with a verified wrong bbox to be treated as wrong
    '''
    tag_only = True if pred_type=="VerifyImage" else False
    if tag_only:
        pos_iou_threshold = 0
        neg_iou_threshold = 0

    include_classes = None
    if include_labelmap:
        include_classes = set(l[0].lower() for l in read_from_file(include_labelmap))
    # load existing ground truth labels
    gt_cfg = GroundTruthConfig(gt_config_file)
    if dataset_name not in gt_cfg.datasets():
        raise Exception("unknow dataset: {}".format(dataset))

    num_bbox_to_submit = 0
    num_bbox_pos = 0
    num_bbox_neg = 0
    num_bbox_total = 0
    output_data = []

    # load prediction results
    pred_config = gt_cfg.baseline_info(dataset_name, pred_name)
    pred_file = pred_config["result"]
    pred_tsv = DetectionFile(pred_file, **pred_config)
    logging.info("load prediction results from: {}, {}".format(pred_file, str(pred_config)))

    # load ground truth and verified baselines
    gt_dataset, gt_split = gt_cfg.gt_dataset(dataset_name)
    baselines_tsv = []
    for baseline in gt_cfg.baselines(dataset_name):
        if baseline == pred_name:
            continue
        base_cfg = gt_cfg.baseline_info(dataset_name, baseline)
        baselines_tsv.append(DetectionFile(gt_cfg.baseline_file(dataset_name, baseline), **base_cfg))

    for imgkey, coded_rects in gt_dataset.iter_data(gt_split, 'label', version=-1):
        if imgkey not in pred_tsv:
            continue
        # get result bboxes
        bboxes = []
        for b in pred_tsv[imgkey]:
            if include_classes and b["class"].lower() not in include_classes:
                continue
            if not tag_only and not is_valid_bbox(b):
                logging.error("invalid bbox: {}".format(str(b)))
                continue
            bboxes.append(b)
        num_bbox_total += len(bboxes)

        verified = set()

        if tag_only:
            assert(all(["rect" not in b for b in bboxes]))
            gt_bboxes = [b for b in json.loads(coded_rects) if "rect" not in b]
        else:
            assert(all(["rect" in b for b in bboxes]))
            gt_bboxes = [b for b in json.loads(coded_rects) if "rect" in b]
        # if prediction is in ground truth, it is verified as correct
        idx_map = get_bbox_matching_map(bboxes, gt_bboxes, pos_iou_threshold)
        verified.update([idx for idx in range(len(bboxes)) if idx_map[idx]])
        num_pos_cur = len(verified)
        num_bbox_pos += num_pos_cur

        # if prediction is in baseline but not in gt, it is verified as wrong
        for idx in range(len(bboxes)):
            if idx in verified:
                continue
            for base_tsv in baselines_tsv:
                if tag_only:
                    base_bboxes = [b for b in base_tsv[imgkey] if "rect" not in b]
                else:
                    base_bboxes = [b for b in base_tsv[imgkey] if "rect" in b]
                if search_bbox_in_list(bboxes[idx], base_bboxes, neg_iou_threshold) >= 0:
                    verified.add(idx)
                    break
        num_bbox_neg += len(verified) - num_pos_cur

        if len(verified) < len(bboxes):
            new_bboxes = []
            for idx in range(len(bboxes)):
                if idx not in verified:
                    new_bboxes.append(bboxes[idx])

            num_bbox_to_submit += len(new_bboxes)
            image_url = get_img_url2(imgkey)
            output_data.append([dataset_name, json.dumps(new_bboxes), image_url])

    if output_data:
        write_to_file(output_data, outfile)
    logging.info("#total bbox: {}, #verified correct: {}, #verified wrong: {}, #to submit: {}, file: {}"
                 .format(num_bbox_total, num_bbox_pos, num_bbox_neg, num_bbox_to_submit, outfile))
    return num_bbox_to_submit


def merge_gt(dataset_name, gt_config_file, res_files, bbox_matching_iou):
    '''
    Processes human evaluation results to add into existing ground truth labels
    res_file: aggregated results from UHRS. The first column is 1/2/3, which
        stands for Yes/No/Can'tJudge. The second column is a list of dict, each
        dict has "image_url", "bboxes".
    '''
    # load old ground truth dataset
    gt_cfg = GroundTruthConfig(gt_config_file)
    gt_dataset, gt_split = gt_cfg.gt_dataset(dataset_name)

    def gen_labels():
        for res_file in res_files:
            for parts in read_from_file(res_file, check_num_cols=2):
                assert(int(parts[0]) in [1, 2, 3])
                # consensus yes
                if int(parts[0]) == 1:
                    for task in json.loads(parts[1]):
                        yield task["image_url"], task["bboxes"]

    add_label_to_dataset(gt_dataset, gt_split, bbox_matching_iou, gen_labels(),
            label_key_type="url",
            info_str="add labels from: {}".format(', '.join(res_files)))


def add_label_to_dataset(dataset, split, iou_threshold, labels,
                         label_key_type="url", info_str=None):
    """
    Adds the bbox labels to existing dataset. New labels of IoU>threshold with
    other labels will not be added
    Args:
        dataset: TSVDataset
        labels: iterable of labels. Each label contains imgkey/url, list of bboxes
        label_key_type: choose from url or key
    """
    assert(label_key_type=="url" or label_key_type=="key")
    existing_res = {}
    url_key_map = {}
    for key, coded_rects in dataset.iter_data(split, 'label', version=-1):
        existing_res[key] = json.loads(coded_rects)
        url_key_map[get_img_url2(key)] = key

    num_added = 0
    for parts in labels:
        imgkey = parts[0]
        if label_key_type == "url":
            imgkey = url_key_map[imgkey]
        for new_bbox in json.loads(parts[1]):
            if search_bbox_in_list(new_bbox, existing_res[imgkey], iou_threshold)<0:
                existing_res[imgkey].append(new_bbox)
                num_added += 1

    if num_added > 0:
        def gen_rows():
            for key, _ in dataset.iter_data(split, 'label', version=-1):
                yield key, json.dumps(existing_res[key], separators=(',', ':'))
        info = [("num_added", str(num_added))]
        if info_str:
            info.append(info_str)
        dataset.update_data(gen_rows(), split, "label", generate_info=info)
    else:
        logging.info("no new add")


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
