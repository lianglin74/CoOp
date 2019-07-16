import collections
import json
import logging
import math
import numpy as np
import os
import os.path as op

import matplotlib
# use a non-interactive backend to generate images without having a window appear
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from logo.classifier import CropTaggingWrapper
from logo import constants
from evaluation.utils import is_valid_bbox

from qd.qd_common import init_logging, worth_create, read_to_buffer, ensure_directory, json_dump
from qd.tsv_io import tsv_reader, tsv_writer, TSVDataset
from qd import deteval, qd_common

trained_dataset = "brand1048"
trained_dataset_version = 2
# new_dataset = "logo40"
new_dataset = "logo200"
pair_dataset = "logo40_pair"
data_split = "test"

dataset_cfgs = [
    {"data": "brand1048", "split": "test", "version": 4},
    {"data": "sports_missingSplit", "split": "test", "version": -1},
    {"data": "logo40", "split": "test", "version": -1},
    # {"data": "logo200", "split": "test", "version": -1},
    {"data": "logo40_overlap", "split": "test", "version": -1},
    {"data": "logo40_unique", "split": "test", "version": -1},
]

# rootpath = "/raid/data/brand_output/"
rootpath = "brand_output/"
iou_thres = [0.5]

det1_expid = "brand1048_darknet19_448_B_noreorg_rotate10_Init.best_model8022_extraConvKernel.1.3.1_TsvBoxSamples50ExtraConvGroups1_4_1EffectBatchSize128"
det2_expid = "TaxLogoV1_1_darknet19_448_C_Init.best_model9748_maxIter.50eEffectBatchSize128_bb_only"
det3_expid = "TaxLogoV1_7_darknet19_448_C_Init.best_model9748_maxIter.75eEffectBatchSize128_bb_only"
det4_expid = "brand1048Clean_net_RongFasterRCNN"

tag1_expid = "gt_only"
tag2_expid = "pretrained_0.1"
tag3_expid = "ccs_code_fix"
tag4_expid = "ccs_old"
tag5_expid = "pretrained_0.1_old"
tag6_expid = "logo40can2"

def main():
    records = ["Methods"] + ["mAP@{}".format(iou) for iou in iou_thres] * 4

    records.append(["1k logo detector"] + evaluate_detector(det1_expid))
    records.append(["logo/non-logo detector"] + evaluate_detector(det3_expid))

    records.append(["two-stage"] + evaluate_two_stage(det2_expid, tag3_expid))
    records.append(["two-stage-ccs"] + evaluate_two_stage(det3_expid, tag4_expid))

    fpath = os.path.join(rootpath, "table")
    tsv_writer(records, fpath)


def evaluate_two_stage(det_expid, tag_expid):
    outdir = os.path.join(rootpath, "{}/classifier/{}/eval".format(det_expid, tag_expid))
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    croptagger = CropTaggingWrapper(det_expid, tag_expid)
    eval_res = []
    # pretrained
    pred_file = croptagger.predict_on_known_class(trained_dataset, data_split)
    eval_file = evaluate_detection(trained_dataset, data_split, pred_file,
            os.path.join(outdir, "{}.{}.pretrained.region.eval".format(trained_dataset, data_split)),
            region_only=True, version=2)
    eval_res.extend(parse_eval_file(eval_file))
    fname = os.path.join(outdir, "{}.{}.pretrained.det.eval".format(trained_dataset, data_split))
    eval_file = evaluate_detection(trained_dataset, data_split, pred_file, fname, version=2)
    eval_res.extend(parse_eval_file(eval_file))

    calc_map_vs_speed(trained_dataset, data_split, pred_file, fname)

    # feature matching on unknown classes
    pred_file = croptagger.predict_on_unknown_class(new_dataset, data_split)
    eval_file = evaluate_detection(new_dataset, data_split, pred_file, os.path.join(outdir, "{}.{}.new.region.eval".format(new_dataset, data_split)), region_only=True)
    eval_res.extend(parse_eval_file(eval_file))
    fname = os.path.join(outdir, "{}.{}.new.det.eval".format(new_dataset, data_split))
    eval_file = evaluate_detection(new_dataset, data_split, pred_file, fname)
    eval_res.extend(parse_eval_file(eval_file))

    calc_map_vs_speed(new_dataset, data_split, pred_file, fname)

    # metrics for featurizer/recognition
    # ROC for pair comparision (positive pairs are from the same class, negative pairs are from differen classes)
    croptagger.compare_pairs(pair_dataset, data_split)
    # precision@k on ground truth regions (match features of canonical images and real images)
    croptagger.predict_on_unknown_class(new_dataset, data_split, region_source="gt")
    return eval_res


def evaluate_detector(det_expid):
    from qd.yolotrain import yolo_predict
    from evaluation.visual import get_wrong_pred, visualize_fp_fn_result

    outdir = os.path.join(rootpath, "{}/deteval".format(det_expid))
    ensure_directory(outdir)

    eval_res = []
    file_dict = {}

    for cfg in dataset_cfgs:
        dataset_name = cfg["data"]
        split = cfg["split"]
        version = cfg["version"]
        is_region_only = True

        pred_file, _ = yolo_predict(full_expid=det_expid, test_data=dataset_name, test_split=split)
        filter_invalid_bboxes(pred_file)
        gt_dataset = TSVDataset(dataset_name)
        gt_file = gt_dataset.get_data(split, t='label', version=version)

        eval_file = _eval_helper(pred_file, dataset_name, split, version, is_region_only, outdir)
        eval_res.extend(parse_eval_file(eval_file))
        file_dict[op.basename(eval_file)] = eval_file

        # visualize false positive, false negative
        fp_fn_fpath = op.join(outdir, "{}_fp_fn_pred_gt.tsv".format(dataset_name))
        visual_dir = op.join(outdir, "visualize")
        ensure_directory(visual_dir)
        get_wrong_pred(pred_file, gt_file, outfile=fp_fn_fpath,
                   min_conf=0.2, iou=0.5, region_only=is_region_only, num_samples=100)
        visualize_fp_fn_result(fp_fn_fpath, dataset_name, visual_dir)

    draw_pr_curve(file_dict, [0.3, 0.5], os.path.join(outdir, "pr_curve.png"))

    return eval_res

def filter_invalid_bboxes(pred_file):
    def gen_rows():
        for parts in tsv_reader(pred_file):
            bboxes = json.loads(parts[1])
            valid_bboxes = []
            for b in bboxes:
                if is_valid_bbox(b):
                    valid_bboxes.append(b)
                else:
                    logging.info("invalid bbox: {}".format(str(b)))
            yield parts[0], json_dump(valid_bboxes)
    tsv_writer(gen_rows(), pred_file)

def _eval_helper(pred_file, dataset, split, version, is_region_only, outdir):
    eval_method = "region" if is_region_only else "det"
    fname = "{}.{}.v{}.{}.eval".format(dataset, split, version, eval_method)
    eval_file = evaluate_detection(dataset, split, pred_file,
            os.path.join(outdir, fname), region_only=is_region_only, version=version)
    return eval_file

def evaluate_detection(dataset_name, split, pred_file, outfile, region_only=False, version=-1):
    """ Calculates mAP for detection results (label+bbox or bbox only)
    """
    logging.info("Evaluate detection on: {} {}".format(dataset_name, split))
    dataset = TSVDataset(dataset_name)
    truth_iter = dataset.iter_data(split, 'label', version=version)
    return deteval.deteval_iter(truth_iter=truth_iter, dets=pred_file, report_file=outfile, region_only=region_only)


def parse_eval_file(eval_file):
    eval_res= json.loads(read_to_buffer(eval_file))
    return [round(eval_res["overall"][str(iou)]["map"], 4) for iou in iou_thres]


def draw_pr_curve(eval_dict, iou_list, outfile):
    """
    eval_dict: name => eval_file
    """
    is_worth = any([worth_create(eval_dict[k], outfile) for k in eval_dict])
    if not is_worth:
        return

    fig, ax = plt.subplots()
    for name in eval_dict:
        eval_file = eval_dict[name]
        eval_res = json.loads(read_to_buffer(eval_file))
        for iou in iou_list:
            precision_list = eval_res["overall"][str(iou)]["precision"]
            recall_list = eval_res["overall"][str(iou)]["recall"]
            ax.plot(recall_list, precision_list, label=name+"@IoU={}".format(iou))
    ax.margins(0.05)
    ax.legend(loc=0, fontsize="x-small")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    ax.grid()
    fig.savefig(outfile)


def calc_map_vs_speed(dataset_name, split, pred_file, outbase,
            version=0, iou_thres=[0.5]):
    """
    pred_file: TSV file of image_key, list of bboxes. Each bbox must contain: class, rect,
            obj (proposed by Stage-1), conf (classification_conf * obj)
    """
    res_file = outbase + "_perf_res.yaml"
    figure_file = outbase + "_perf.png"
    if not worth_create(pred_file, res_file) and not worth_create(pred_file, figure_file):
        return

    multiscale=False
    num_images = 0
    num_gt = 0

    label_to_keys = None
    # load ground truth
    dataset = TSVDataset(dataset_name)
    # region + class
    truth_iter = dataset.iter_data(split, 'label', version=version)
    truths_dict = deteval.load_truths_iter(truth_iter)
    if deteval.has_negative_labels(truths_dict):
        assert label_to_keys is None
        label_to_keys = deteval.remove_negative_labels(truths_dict)
    # merge all classes, keep region only
    truth_iter = dataset.iter_data(split, 'label', version=version)
    region_truths_dict = deteval.load_truths_iter(truth_iter, region_only=True)

    for label in region_truths_dict:
        for key in region_truths_dict[label]:
            num_images = len(region_truths_dict[label])
            num_gt += len(region_truths_dict[label][key])

    # load detection
    all_det = []
    for parts in tsv_reader(pred_file):
        key = parts[0]
        bboxes = json.loads(parts[1])
        for bbox in bboxes:
            all_det.append([key, json.dumps([bbox]), bbox["obj"]])
    # sort detection by obj score
    all_det = sorted(all_det, key=lambda t: -t[2])
    num_det = len(all_det)
    step = int(num_images / 8)
    cur_det = []

    stats_names = ["map", "region_precision", "region_recall", "region_threshold", "num_proposal"]
    all_stats = {iou: {k: [] for k in stats_names} for iou in iou_thres}
    # add detection from the largest obj score
    for start in np.arange(0, num_det, step):
        end = min(start+step, num_det)
        cur_det.extend([all_det[i][:2] for i in range(start, end)])
        det_dict = deteval.load_dets_iter(cur_det)
        region_det_dict = deteval.load_dets_iter(cur_det, region_only=True)
        # evaluate mAP for class+region
        report = deteval.get_report(truths_dict, det_dict, iou_thres, multiscale, label_to_keys=label_to_keys)
        for iou in iou_thres:
            all_stats[iou]["map"].append(report["overall"][iou]["map"])
        # evaluate P/R for region only
        report = deteval.get_report(region_truths_dict, region_det_dict, iou_thres, multiscale, label_to_keys=label_to_keys)
        is_saturated = False
        for iou in iou_thres:
            all_stats[iou]["region_precision"].append(report["overall"][iou]["precision"][0])
            all_stats[iou]["region_recall"].append(report["overall"][iou]["recall"][0])
            all_stats[iou]["region_threshold"].append(all_det[end-1][2])
            all_stats[iou]["num_proposal"].append(len(cur_det))

            recalls = all_stats[iou]["region_recall"]
            if len(recalls) > 10 and abs(recalls[-1] - recalls[-5]) < 0.001:
                is_saturated = True
        if is_saturated:
            break

    qd_common.write_to_yaml_file(all_stats, res_file)

    fig, ax = plt.subplots()
    for iou in iou_thres:
        # x = [float(p)/num_images for p in all_stats[iou]["num_proposal"]]
        x = all_stats[iou]["region_threshold"]
        ax.plot(x, all_stats[iou]["region_precision"], label="region precision@{}".format(iou))
        ax.plot(x, all_stats[iou]["region_recall"], label="region recall@{}".format(iou))
        ax.plot(x, all_stats[iou]["map"], label="TwoStage mAP@{}".format(iou))

    ax.legend(loc=0, fontsize="small")
    # plt.xlabel("#region proposal per image")
    plt.xlabel("#region obj threshold")
    plt.title("{} ({})  #img:{}  #gt:{}".format(dataset_name, split, num_images, num_gt))
    ax.grid()
    fig.savefig(figure_file)

def tune_threshold(eval_file, iou, target="recall", target_num=0.75):
    eval_res = json.loads(qd_common.read_to_buffer(eval_file))
    all_recalls = eval_res["overall"][str(iou)]["recall"]
    all_precisions = eval_res["overall"][str(iou)]["precision"]
    all_thresholds = eval_res["overall"][str(iou)]["thresholds"]
    assert len(all_recalls) == len(all_precisions) and len(all_recalls) == len(all_thresholds)

    num_total = len(all_recalls)
    target_idx = None
    if target == "recall":
        for i in range(num_total):
            if all_recalls[i] < target_num:
                target_idx = max(i-1, 0)
                break
    elif target == "precision":
        for i in range(num_total):
            if all_precisions[i] > target_num:
                target_idx = max(i-1, 0)
                break
    if target_idx is None:
        target_idx = num_total - 1
    print(target_idx)
    print(all_recalls[target_idx], all_precisions[target_idx], all_thresholds[target_idx])

def filter_topN_pred(pred_file, topN, obj_thres=None):
    def gen_rows():
        for key, coded_rects in tsv_reader(pred_file):
            bboxes = json.loads(coded_rects)
            sorted_bboxes = sorted(bboxes, key=lambda b: b["obj"], reverse=True)
            # choose bbox with topN highest obj score or >obj_thres
            end_idx = topN
            if obj_thres and len(sorted_bboxes) > topN:
                while end_idx < len(sorted_bboxes) and sorted_bboxes[end_idx]["obj"] >= obj_thres:
                    end_idx += 1
            end_idx = min(end_idx, len(sorted_bboxes))
            yield key, json.dumps(sorted_bboxes[0: end_idx], separators=(',', ':'))
    new_pred_file = pred_file+'.top{}'.format(topN)
    tsv_writer(gen_rows(), new_pred_file)
    return new_pred_file

def calculate_confusion_matrix(gt_pred_file, outfile):
    gt2preds = collections.defaultdict(list)
    for parts in tsv_reader(gt_pred_file):
        gt_label = json.loads(parts[1])["class"]
        pred = parts[2].split(';')[0].split(':')[0]
        gt2preds[gt_label].append(pred)

    pred_correct_rates = []
    for gt_label in gt2preds:
        pred_labels = gt2preds[gt_label]
        pred_counts = collections.Counter(pred_labels)
        sorted_pred_counts = sorted([[pred, count] for pred, count in pred_counts.items()], key=lambda t: t[1], reverse=True)
        top_pred = sorted_pred_counts[0][0]
        if top_pred != gt_label:
            correct_rate = 0.0
        else:
            correct_rate = sorted_pred_counts[0][1] / len(pred_labels)
        pred_correct_rates.append([correct_rate, gt_label] + sorted_pred_counts)

    pred_correct_rates = sorted(pred_correct_rates, key = lambda t: t[0])
    tsv_writer(pred_correct_rates, outfile)

def eval_classifier(gt_dataset_name, split, version, det_expid, tag_expid,
            tag_snap_id, tag_model_id, labelmap=None, iou_thres=0.5, enlarge_bbox=1.0,
            topN_rp=None, obj_thres=0):
    """ Calculates:
    top1/5 acc on gt region, mAP with gt region (conf from tag),
    mAP with logo/non-logo region (conf from tag, conf from tag*obj, conf from obj)
    """
    croptagger = CropTaggingWrapper(det_expid, tag_expid,
            tag_snap_id=tag_snap_id, tag_model_id=tag_model_id, labelmap=labelmap)
    stats = []

    def cal_map(pred_file):
        eval_file = pred_file.rsplit('.', 1)[0] + "eval"
        evaluate_detection(gt_dataset_name, split, pred_file, eval_file,
                region_only=False, version=version)
        eval_res= json.loads(read_to_buffer(eval_file))
        # calculate class AP
        class_ap = [(k, v) for k, v in eval_res["overall"][str(iou_thres)]["class_ap"].items()]
        class_ap = sorted(class_ap, key=lambda p: p[1])
        tsv_writer(class_ap, pred_file.replace('.tsv', '.class.ap'))
        return eval_res["overall"][str(iou_thres)]["map"]

    # top1/5 acc on gt region
    topk = (1, 5)
    pred_file, topk_acc, tag_file = croptagger.predict_on_known_class(gt_dataset_name, split,
                version=version, region_source=constants.GT_REGION, enlarge_bbox=enlarge_bbox,
                conf_from=constants.CONF_TAG, eval_topk_acc=max(topk))

    # confusion matrix
    calculate_confusion_matrix(tag_file, tag_file + ".confusion")

    for k in topk:
        stats.append(topk_acc[k-1])
    # mAP with gt region (conf from tag)
    stats.append(cal_map(pred_file))

    # for conf_from in [constants.CONF_TAG, constants.CONF_OBJ_TAG, constants.CONF_OBJ]:
    #     pred_file, _, _ = croptagger.predict_on_known_class(gt_dataset_name, split,
    #                 version=version, region_source=constants.PRED_REGION, enlarge_bbox=enlarge_bbox,
    #                 conf_from=conf_from, eval_topk_acc=None)
    #     if conf_from == constants.CONF_TAG and topN_rp:
    #         pred_file = filter_topN_pred(pred_file, topN_rp, obj_thres=obj_thres)

    #     stats.append(cal_map(pred_file))
    return stats

def run_all_eval_classifier():
    det_expid = det3_expid

    output_root = './brand_output/'
    labelmap = None
    tag_models = [
        ("logo40syn_09v204", "snapshot", "model_best.pth.tar"),
        ("sports_logo40syn", "snapshot", "model_best.pth.tar"),
        ("brandsports_addlogo40syn", "snapshot_fixlabel", "model_best.pth.tar"),
    ]

    def gen_rows():
        for cfg in dataset_cfgs:
            gt_dataset_name, split, version = cfg["data"], cfg["split"], cfg["version"]
            for tag_expid, tag_snap_id, tag_model_id in tag_models:
                for obj_thres in [0.6]:
                    for topN in [10]:
                        for enlarge_bbox in [2]:
                            res = [gt_dataset_name, split, version, obj_thres, topN, enlarge_bbox, tag_expid, tag_snap_id, tag_model_id]
                            res.extend(eval_classifier(gt_dataset_name, split, version, det_expid, tag_expid,
                                    tag_snap_id, tag_model_id, labelmap=labelmap, iou_thres=0.5, enlarge_bbox=enlarge_bbox, topN_rp=topN, obj_thres=obj_thres))
                            print(res)
                            yield res
    tsv_writer(gen_rows(), os.path.join(output_root, "brandsports_addlogo40syn", 'eval.tsv'))


if __name__ == "__main__":
    init_logging()
    # main()
    run_all_eval_classifier()
    # tag_file = "brand_output/brandsports_addlogo40syn/snapshot/eval/logo40.test.2.label-8067034370336282024.dataset.tagging.tsv"

