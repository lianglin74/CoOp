from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import yaml

import _init_paths
from eval_utils import filter_gt, load_gt, load_result
from qd_common import init_logging
from utils import search_bbox_in_list


parser = argparse.ArgumentParser(
    description='Human evaluation for object detection')
parser.add_argument('--gt', default='./groundtruth/config.yaml', type=str,
                    help='''path to yaml config file in the ground truth folder,
                    default is ./groundtruth/config.yaml''')
parser.add_argument('--blacklist', default='', type=str,
                    help='''blacklist filename, blocking categories you don't
                    want to evaluate here''')
parser.add_argument('--set', default='', type=str,
                    help='''dataset name to be evaluated on, default is MIT1K
                    and Instagram''')
parser.add_argument('--iou_threshold', default=0.5, type=float,
                    help='IoU threshold for bounding boxes, default is 0.5')
parser.add_argument('--result', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--result_conf', default=0.0, type=float,
                    help='confidence threshold for result, default is 0')
parser.add_argument('--filter_gt', action='store_true',
                    help='choose if the ground truth should be filtered by given baselines')


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
                 use_filtered_gt=False):
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
    print('\n|*{0}*\t\t|*{1}*|*{2:}*|*{3:}*|*{4}@{6}*|*{5}@{6}*|*{7}*|'
          .format(dataset_name, '#_gt_bboxes', '#_result_bboxes',
                  '#_correct_bboxes', 'prec', 'recall', iou_threshold,
                  '#_result_categories'))

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
    for baseline_info in dataset['baselines']:
        result = load_result(gt_root, baseline_info)
        num_gt_bboxes, num_result_bboxes, num_correct_bboxes, \
            precision, recall, num_unique_categories = eval_result(
                                                    gt, result, iou_threshold)
        print('|{:15}\t|{:10}\t|{:10}\t|{:10}\t|{:10.2f}\t|{:10.2f}\t|{:10}|'
              .format(baseline_info["name"], num_gt_bboxes, num_result_bboxes,
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
        col_name = baseline_info["name"] + " precision (recall)"
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
    for baseline_info in dataset['baselines']:
        if "deprecated" in baseline_info:
            continue
        # the chosen conf threshold to present final prediction
        if "conf_threshold" in baseline_info:
            chosen_threshold = baseline_info["conf_threshold"]
        else:
            chosen_threshold = start_from_conf
        baseline_info["conf_threshold"] = start_from_conf
        if "threshold" in baseline_info:
            del baseline_info["threshold"]
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
                 markevery=[chosen_idx - start_idx], label=baseline_info["name"])
    ax.margins(0.05)
    plt.legend()
    plt.xlabel("Recall")
    plt.xticks([])
    plt.ylabel("Precision")
    plt.title(dataset_name)
    fig.savefig(os.path.join(gt_root, dataset_name + "_pr_curve.png"))


def main(args):
    gt_root = os.path.split(args.gt)[0]
    with open(args.gt, 'r') as fp:
        gt_cfg = yaml.load(fp)

    for dataset_name, dataset in gt_cfg.items():
        if len(args.set) == 0 or args.set.lower() == dataset_name.lower():
            if args.result:
                # if new result is given, add to existing models
                dataset["baselines"].append(
                    {"name": "new result", "result": args.result,
                     "conf_threshold": args.result_conf})
            if args.blacklist:
                for config in dataset["baselines"]:
                    config["blacklist"] = args.blacklist
            eval_dataset(gt_root, dataset_name, dataset, args.iou_threshold, use_filtered_gt=args.filter_gt)
            draw_pr_curve(gt_root, dataset_name, dataset, args.iou_threshold)


def parse_args():
    args = parser.parse_args()
    if isinstance(args, dict):
        args = argparse.Namespace()
    if not args.gt.endswith('yaml'):
        raise Exception("Please specify the config yaml file.")
    return args

if __name__ == '__main__':
    init_logging()
    args = parse_args()
    if args.filter_gt:
        filter_gt(args.gt, args.iou_threshold, args.blacklist)
    main(args)
