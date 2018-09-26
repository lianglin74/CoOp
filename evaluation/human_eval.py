from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import yaml

from utils import read_from_file
from utils import search_bbox_in_list
from utils import write_to_file

parser = argparse.ArgumentParser(
    description='Human evaluation for object detection')
parser.add_argument('--gt', default='./groundtruth', type=str,
                    help='''path to ground truth folder or yaml config file,
                    default is ./groundtruth''')
parser.add_argument('--set', default='', type=str,
                    help='''dataset name to be evaluated on, default is MIT1K
                    and Instagram''')
parser.add_argument('--iou_threshold', default=0.5, type=float,
                    help='IoU threshold for bounding boxes, default is 0.5')
parser.add_argument('--result', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--result_conf', default=0.0, type=float,
                    help='confidence threshold for result, default is 0')


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
        display_dict = {p[0]: p[1] for p in read_from_file(config["display"])}
    else:
        display_dict = None
    for cols in read_from_file(config["result"]):
        imgkey, bboxes = cols[0:2]
        assert(imgkey not in result)
        result[imgkey] = _thresholding_detection(
                            json.loads(bboxes),
                            threshold_dict, display_dict,
                            config["obj_threshold"], config["conf_threshold"])
    return result


def _thresholding_detection(bbox_list, thres_dict, display_dict,
                            obj_threshold, conf_threshold):
    res = []
    for b in bbox_list:
        if "obj" in b and b["obj"] < obj_threshold:
            continue
        if b["conf"] < conf_threshold:
            continue
        term = b["class"]
        if thres_dict:
            if term not in thres_dict or b["conf"] < thres_dict[term]:
                continue
        if display_dict and term in display_dict:
            b['class'] = display_dict[term]
        res.append(b)
    return res


def eval_result(gt, result, iou_threshold):
    """Calculates accumulated statitics on all results

    Args:
        gt: dict of image key and ground truth pairs
        result: dict of image key and detection result pairs
        iou_threshold: IoU threshold to consider two boxes as matching

    Returns:
        Tuple of statistics
    """
    num_gt_bboxes = sum([len(bboxes) for _, bboxes in gt.items()])
    num_result_bboxes = sum([len(bboxes) for _, bboxes in result.items()])

    num_correct_bboxes = 0
    categories = set()
    for imgkey, bboxes in result.items():
        for b in bboxes:
            categories.add(b["class"].lower())
            if search_bbox_in_list(b, gt[imgkey], iou_threshold) >= 0:
                num_correct_bboxes += 1

    precision = float(num_correct_bboxes) / num_result_bboxes * 100
    recall = float(num_correct_bboxes) / num_gt_bboxes * 100

    return num_gt_bboxes, num_result_bboxes, num_correct_bboxes, \
        precision, recall, len(categories)


def eval_result_per_class(gt, result, iou_threshold):
    """Calculates statistics for each class in the detection results

    Returns:
        List of tuple: class name, #ground_truth, #result, #correct,
        precision, recall
    """
    num_result_bboxes_per_class = collections.defaultdict(int)
    num_correct_bboxes_per_class = collections.defaultdict(int)
    for imgkey, bboxes in result.items():
        for b in bboxes:
            num_result_bboxes_per_class[b["class"].lower()] += 1
            if search_bbox_in_list(b, gt[imgkey], iou_threshold) >= 0:
                num_correct_bboxes_per_class[b["class"].lower()] += 1
    return num_result_bboxes_per_class, num_correct_bboxes_per_class


def eval_dataset(gt_root, dataset_name, dataset, iou_threshold,
                 use_filtered_gt=True):
    """Prints the table summarizing evaluation results on the dataset.
    Writes per-class evaluation results to directory gt_root/per_class_pr

    Args:
        gt_root: dirctory path
        dataset_name: str of dataset name
        dataset: dict of config information for the dataset
        iou_threshold: IoU threshold for bboxes
        use_filtered_gt: indicate whether ground truth labels should only
            include detection output under current thresholding settings
    """
    print('|*{0}*\t\t|*{1}*|*{2:}*|*{3:}*|*{4}@{6}*|*{5}@{6}*|*{7}*|'.format(
        dataset_name,
        '#_gt_bboxes', '#_result_bboxes', '#_correct_bboxes',
        'prec', 'recall', iou_threshold, '#_result_categories'))

    if use_filtered_gt:
        gt_file = dataset['groundtruth']["filtered"]
    else:
        gt_file = dataset['groundtruth']["original"]
    gt_file = os.path.join(gt_root, gt_file)
    gt = load_gt(gt_file)
    num_gt_per_class = collections.defaultdict(int)
    for _, bboxes in gt.items():
        for b in bboxes:
            num_gt_per_class[b["class"].lower()] += 1
    df_per_class = pd.DataFrame({"category": [c for c in num_gt_per_class]})
    df_per_class["#gt"] = pd.Series([num_gt_per_class[c]
                                    for c in df_per_class["category"]])

    # evaluate all baselines
    for baseline, baseline_info in dataset['baselines'].items():
        result = load_result(gt_root, baseline_info)
        num_gt_bboxes, num_result_bboxes, num_correct_bboxes, \
            precision, recall, num_unique_categories = eval_result(
                                                    gt, result, iou_threshold)
        print('|{:15}\t|{:10}\t|{:10}\t|{:10}\t|{:10.2f}\t|{:10.2f}\t|{:10}|'
              .format(
                  baseline, num_gt_bboxes, num_result_bboxes,
                  num_correct_bboxes, precision, recall,
                  num_unique_categories))

        num_res_per_class, num_cor_per_class = eval_result_per_class(
            gt, result, iou_threshold)
        pr = []
        for c in df_per_class["category"]:
            if c not in num_cor_per_class:
                pr.append([0, 0])
            else:
                pr.append([float(num_cor_per_class[c]) / num_res_per_class[c],
                           float(num_cor_per_class[c]) / num_gt_per_class[c]])
        col_name = baseline + " precision (recall)"
        df_per_class[col_name] = pd.Series(["{:.2f} ({:.2f})".format(
                                            p * 100, r * 100)
                                            for p, r in pr])
    df_per_class = df_per_class.sort_values(by="#gt", ascending=False)
    df_per_class.to_csv(os.path.join(gt_root,
                        "stats/" + dataset_name + "_category_pr.tsv"),
                        sep='\t', index=False)


def draw_pr_curve(gt_root, dataset_name, dataset, iou_threshold,
                  start_from_conf=0.3):
    """Draws precision recall curve for all models on the dataset
    """
    gt_file = os.path.join(gt_root, dataset['groundtruth']["original"])
    gt = load_gt(gt_file)
    num_gt = sum(len(v) for k, v in gt.items())

    fig, ax = plt.subplots()
    for baseline, baseline_info in dataset['baselines'].items():
        # the chosen conf threshold to present final prediction
        if "conf_threshold" in baseline_info:
            chosen_threshold = baseline_info["conf_threshold"]
        else:
            chosen_threshold = start_from_conf
        baseline_info["conf_threshold"] = start_from_conf
        result = load_result(gt_root, baseline_info)
        y_score_gt = []
        for imgkey, bboxes in result.items():
            for b in bboxes:
                if search_bbox_in_list(b, gt[imgkey], iou_threshold) >= 0:
                    y_score_gt.append((b["conf"], 1))
                else:
                    y_score_gt.append((b["conf"], 0))
        y_score_gt = sorted(y_score_gt, key=lambda t: t[0], reverse=True)
        precision_list = []
        recall_list = []
        num_correct, num_result = 0, 0
        # the index of precision-recall pair at chosen threshold
        chosen_idx = -1
        for score, is_correct in y_score_gt:
            num_result += 1
            if is_correct:
                num_correct += 1
            precision_list.append(float(num_correct) / num_result)
            recall_list.append(float(num_correct) / num_gt)
            if score <= chosen_threshold and chosen_idx < 0:
                chosen_idx = num_result - 1
        # in case the chosen threshold is smaller than all output confidence
        if chosen_idx < 0:
            chosen_idx = num_result - 1
        # skip the precsion, recall calculated with less than 100 data points
        # as they are not stable
        start_idx = 100
        plt.plot(recall_list[start_idx:], precision_list[start_idx:], '-D',
                 markevery=[chosen_idx - start_idx], label=baseline)
    ax.margins(0.05)
    plt.legend()
    plt.xlabel("Recall")
    plt.xticks([])
    plt.ylabel("Precision")
    plt.title(dataset_name)
    plt.show()
    fig.savefig(os.path.join(gt_root, dataset_name + "_pr_curve.png"))


def main(args):
    if isinstance(args, dict):
        args = argparse.Namespace()

    if not args.gt.endswith('yaml'):
        args.gt = os.path.join(args.gt, 'config.yaml')

    gt_root = os.path.split(args.gt)[0]
    with open(args.gt, 'r') as fp:
        gt_cfg = yaml.load(fp)

    for dataset_name, dataset in gt_cfg.items():
        if len(args.set) == 0 or args.set.lower() == dataset_name.lower():
            if args.result:
                # if new result is given, add to existing models
                dataset["baselines"]["new result"] = {}
                dataset["baselines"]["new result"]["result"] = args.result
                dataset["baselines"]["new result"]["conf_threshold"] \
                    = args.result_conf
            eval_dataset(gt_root, dataset_name, dataset, args.iou_threshold)
            draw_pr_curve(gt_root, dataset_name, dataset, args.iou_threshold)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
