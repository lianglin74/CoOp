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
from evaluation.eval_utils import filter_gt, load_gt, load_result, GroundTruthConfig, DetectionFile
from scripts.qd_common import init_logging
from evaluation import utils


parser = argparse.ArgumentParser(
    description='Human evaluation for object detection')
parser.add_argument('--config', default='./groundtruth/config.yaml', type=str,
                    help='''path to yaml config file of ground truth and predictions,
                    default is ./groundtruth/config.yaml''')
parser.add_argument('--blacklist', default='', type=str,
                    help='''blacklist filename, blocking categories you don't
                    want to evaluate here''')
parser.add_argument('--set', default=None, nargs='+',
                    help='datasets to be evaluated, default is all the dataset in config')
parser.add_argument('--iou_threshold', default=0.5, type=float,
                    help='IoU threshold for bounding boxes, default is 0.5')
parser.add_argument('--filter_gt', action='store_true',
                    help='choose if the ground truth should be filtered by given baselines')


def eval_result(gt, result, iou_threshold):
    """Calculates accumulated statitics on all results

    Args:
        gt: dict of image key: ground truth bboxes
        result: dict of image key: detection result bboxes
        iou_threshold: IoU threshold to consider two boxes as matching

    Returns:
        Tuple of statistics
    """
    num_gt_bboxes = 0
    num_result_bboxes = 0
    num_correct_bboxes = 0
    categories = set()
    for imgkey in gt:
        gt_bboxes = gt[imgkey]
        num_gt_bboxes += len(gt_bboxes)
        if imgkey not in result:
            continue
        res_bboxes = result[imgkey]
        num_result_bboxes += len(res_bboxes)

        idx_map = utils.get_bbox_matching_map(res_bboxes, gt_bboxes, iou_threshold,
                                              allow_multiple_to_one=True)
        num_correct_bboxes += len(idx_map)
        for b in res_bboxes:
            categories.add(b["class"].lower())

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
    for imgkey in result:
        res_bboxes = result[imgkey]
        gt_bboxes = gt[imgkey]
        idx_map = utils.get_bbox_matching_map(res_bboxes, gt_bboxes, iou_threshold,
                                              allow_multiple_to_one=True)
        for i, b in enumerate(res_bboxes):
            c = b["class"].lower()
            num_result_bboxes_per_class[c] += 1
            if i in idx_map:
                num_correct_bboxes_per_class[c] += 1
    return num_result_bboxes_per_class, num_correct_bboxes_per_class


def eval_dataset(gt_config_file, dataset_name, iou_threshold,
                 use_filtered_gt=False, blacklist=None):
    """Prints the table summarizing evaluation results on the dataset.
    Writes per-class evaluation results to directory gt_root/per_class_pr

    Args:
        gt_config_file: yaml file
        dataset_name: str of dataset name
        iou_threshold: IoU threshold for bboxes
        use_filtered_gt: indicate whether ground truth labels should only
            include detection output under current thresholding settings
    """
    print('\n|*{0}*\t\t|*{1}*|*{2:}*|*{3:}*|*{4}@{6}*|*{5}@{6}*|*{7}*|'
          .format(dataset_name, '#_gt_bboxes', '#_result_bboxes',
                  '#_correct_bboxes', 'prec', 'recall', iou_threshold,
                  '#_result_categories'))
    gt_cfg = GroundTruthConfig(gt_config_file)
    gt_file = gt_cfg.gt_file(dataset_name, filtered=use_filtered_gt)
    gt_tsv = DetectionFile(gt_file, blacklist=blacklist)
    gt = {}
    for imgkey in gt_tsv:
        gt[imgkey] = [b for b in gt_tsv[imgkey] if "rect" in b]
    num_gt_per_class = collections.defaultdict(int)
    for imgkey in gt:
        for b in gt[imgkey]:
            num_gt_per_class[b["class"].lower()] += 1
    df_per_class = pd.DataFrame({"category": [c for c in num_gt_per_class]})
    df_per_class["#gt"] = pd.Series([num_gt_per_class[c]
                                    for c in df_per_class["category"]])

    # evaluate all baselines
    for baseline in gt_cfg.baselines(dataset_name):
        base_file = gt_cfg.baseline_file(dataset_name, baseline)
        base_info = gt_cfg.baseline_info(dataset_name, baseline)
        base_info["blacklist"] = blacklist
        result = DetectionFile(base_file, sort_by_conf=True, **base_info)
        num_gt_bboxes, num_result_bboxes, num_correct_bboxes, \
            precision, recall, num_unique_categories = eval_result(
                                                    gt, result, iou_threshold)
        print('|{:15}\t|{:10}\t|{:10}\t|{:10}\t|{:10.2f}\t|{:10.2f}\t|{:10}|'
              .format(baseline, num_gt_bboxes, num_result_bboxes,
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
    gt_root = os.path.split(gt_config_file)[0]
    stats_dir = os.path.join(gt_root, "stats")
    if not os.path.exists(stats_dir):
        os.mkdir(stats_dir)
    df_per_class.to_csv(os.path.join(stats_dir,
                        "{}_category_pr.tsv".format(dataset_name)),
                        sep='\t', index=False)


def draw_pr_curve(gt_config_file, dataset_name, iou_threshold,
                  start_from_conf=0.3, blacklist=None):
    """Draws precision recall curve for all models on the dataset
    """
    gt_cfg = GroundTruthConfig(gt_config_file)
    gt_file = gt_cfg.gt_file(dataset_name)
    gt_tsv = DetectionFile(gt_file, blacklist=blacklist)
    gt = {}
    num_gt = 0
    for imgkey in gt_tsv:
        gt[imgkey] = [b for b in gt_tsv[imgkey] if "rect" in b]
        num_gt += len(gt[imgkey])

    fig, ax = plt.subplots()
    for baseline in gt_cfg.baselines(dataset_name):
        baseline_info = gt_cfg.baseline_info(dataset_name, baseline)
        if "deprecated" in baseline_info:
            continue
        # the chosen conf threshold to present final prediction
        if "conf_threshold" in baseline_info:
            chosen_threshold = baseline_info["conf_threshold"]
            start_from_conf = min(start_from_conf, chosen_threshold)
        else:
            chosen_threshold = start_from_conf
        baseline_info["conf_threshold"] = start_from_conf
        if "threshold" in baseline_info:
            del baseline_info["threshold"]
        baseline_info["blacklist"] = blacklist
        result = DetectionFile(baseline_info["result"], sort_by_conf=True, **baseline_info)
        y_score_gt = []
        for imgkey in result:
            res_bboxes = result[imgkey]
            idx_map = utils.get_bbox_matching_map(res_bboxes, gt[imgkey], iou_threshold,
                                                  allow_multiple_to_one=True)
            for i, b in enumerate(res_bboxes):
                if i in idx_map:
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
            num_correct += is_correct
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
    gt_root = os.path.split(gt_config_file)[0]
    stats_dir = os.path.join(gt_root, "stats")
    if not os.path.exists(stats_dir):
        os.mkdir(stats_dir)
    fig.savefig(os.path.join(stats_dir, dataset_name + "_pr_curve.png"))


def main(args):
    dataset_list = args.set
    if dataset_list is None:
        gt_cfg = GroundTruthConfig(args.config)
        dataset_list = gt_cfg.datasets()
    for dataset_name in dataset_list:
        eval_dataset(args.config, dataset_name, args.iou_threshold, use_filtered_gt=args.filter_gt,
                     blacklist=args.blacklist)
        draw_pr_curve(args.config, dataset_name, args.iou_threshold, blacklist=args.blacklist)


def parse_args():
    args = parser.parse_args()
    if isinstance(args, dict):
        args = argparse.Namespace()
    if not args.config.endswith('yaml'):
        raise Exception("Please specify the config yaml file.")
    return args

if __name__ == '__main__':
    init_logging()
    args = parse_args()
    if args.filter_gt:
        filter_gt(args.config, args.iou_threshold, args.blacklist)
    main(args)
