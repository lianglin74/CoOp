import collections
import json
import logging
import math
import numpy as np
import os

import matplotlib
# use a non-interactive backend to generate images without having a window appear
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import _init_paths
from logo.classifier import CropTaggingWrapper
from logo import constants
from qd.qd_common import init_logging, worth_create, read_to_buffer
from qd.tsv_io import tsv_reader, tsv_writer, TSVDataset
from qd import deteval, qd_common
from scripts.yolotrain import yolo_predict
from scripts.pytablemd import write_tablemd

trained_dataset = "brand1048"
trained_dataset_version = 2
new_dataset = "logo40"
pair_dataset = "logo40_pair"
data_split = "test"
trained_gt = "/raid/data/brand1048/test.label.tsv"
new_gt = "/raid/data/logo40/test.label.tsv"

rootpath = "/raid/data/brand_output/"
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
    records = []
    headings = ["Methods"] + ["mAP@{}".format(iou) for iou in iou_thres] * 4

    records.append(["1k logo detector"] + evaluate_detector(det1_expid))
    records.append(["logo/non-logo detector"] + evaluate_detector(det3_expid))

    records.append(["two-stage"] + evaluate_two_stage(det2_expid, tag3_expid))
    records.append(["two-stage-ccs"] + evaluate_two_stage(det3_expid, tag4_expid))

    fpath = os.path.join(rootpath, "table")
    fields = range(len(records[0]))
    align = [('^', '<')] + [('^', '^')]*(len(headings) - 1)
    with open(fpath, 'w') as fp:
        write_tablemd(fp, records, fields, headings, align)


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
    outdir = os.path.join(rootpath, "{}/deteval".format(det_expid))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    eval_res = []
    file_dict = {}

    # pretrained
    pred_file, _ = yolo_predict(full_expid=det_expid, test_data=trained_dataset, test_split=data_split)
    fname = "{}.{}.pretrained.region.eval".format(trained_dataset, data_split)
    eval_file = evaluate_detection(trained_dataset, data_split, pred_file,
            os.path.join(outdir, fname), region_only=True, version=trained_dataset_version)
    eval_res.extend(parse_eval_file(eval_file))
    file_dict[fname] = eval_file
    eval_file = evaluate_detection(trained_dataset, data_split, pred_file,
            os.path.join(outdir, "{}.{}.pretrained.det.eval".format(trained_dataset, data_split)), version=trained_dataset_version)
    eval_res.extend(parse_eval_file(eval_file))

    # logo/non-logo
    pred_file, _ = yolo_predict(full_expid=det_expid, test_data=new_dataset, test_split=data_split)
    fname = "{}.{}.new.region.eval".format(new_dataset, data_split)
    eval_file = evaluate_detection(new_dataset, data_split, pred_file, os.path.join(outdir, fname), region_only=True)
    eval_res.extend(parse_eval_file(eval_file))
    file_dict[fname] = eval_file
    # detector can not detect unknown categories
    eval_res.extend(["N/A"]*len(iou_thres))

    draw_pr_curve(file_dict, [0.3, 0.5], os.path.join(outdir, "pr_curve.png"))
    return eval_res

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
    tsv_writer(gen_rows(), pred_file+'.top{}'.format(topN))

def eval_classifier(gt_dataset_name, split, version, det_expid, tag_expid,
            tag_snap_id, labelmap=None, iou_thres=0.5, enlarge_bbox=1.0):
    """ Calculates:
    top1/5 acc on gt region, mAP with gt region (conf from tag),
    mAP with logo/non-logo region (conf from tag, conf from tag*obj, conf from obj)
    """
    croptagger = CropTaggingWrapper(det_expid, tag_expid,
            tag_snap_id=tag_snap_id, labelmap=labelmap)
    stats = []

    def cal_map(pred_file):
        eval_file = pred_file.rsplit('.', 1)[0] + "eval"
        evaluate_detection(gt_dataset_name, split, pred_file, eval_file,
                region_only=False, version=version)
        eval_res= json.loads(read_to_buffer(eval_file))
        return eval_res["overall"][str(iou_thres)]["map"]

    # top1/5 acc on gt region
    topk = (1, 5)
    pred_file, topk_acc = croptagger.predict_on_known_class(gt_dataset_name, split,
                version=version, region_source=constants.GT_REGION, enlarge_bbox=enlarge_bbox,
                conf_from=constants.CONF_TAG, eval_topk_acc=max(topk))
    for k in topk:
        stats.append(topk_acc[k-1])
    # mAP with gt region (conf from tag)
    stats.append(cal_map(pred_file))

    for conf_from in [constants.CONF_TAG, constants.CONF_OBJ_TAG, constants.CONF_OBJ]:
        pred_file, _ = croptagger.predict_on_known_class(gt_dataset_name, split,
                    version=version, region_source=constants.PRED_REGION, enlarge_bbox=enlarge_bbox,
                    conf_from=conf_from, eval_topk_acc=None)
        stats.append(cal_map(pred_file))
    return stats

def test_eval_classifier():
    gt_dataset_name = "brand1048"
    split = "test"
    version = 4
    det_expid = det3_expid

    output_root = 'data/brand_output/'
    # labelmap = "data/brand_output/TaxLogoV1_7_darknet19_448_C_Init.best_model9748_maxIter.75eEffectBatchSize128_bb_only/classifier/add_sports/labelmap.txt"
    labelmap = None
    all_tag_expid = []
    # for d in os.listdir(output_root):
    #     if os.path.isdir(os.path.join(output_root, d)) and d.startswith("brand1048_resnet18") and d!="brand1048_resnet18_nobg_input112":
    #         all_tag_expid.append(d)
    # all_tag_expid.append("brandsports_resnet18_nobg")
    # tag_snap_pairs = []
    # for tag_expid in all_tag_expid:
    #     for tag_snap_id in os.listdir(os.path.join(output_root, tag_expid)):
    #         tag_snap_pairs.append((tag_expid, tag_snap_id))

    tag_snap_pairs = [("brand1048_resnet18_nobg_ccs", "snapshot_2.0"), ("brand1048_resnet18", "snapshot_addrp")]

    def gen_rows():
        for tag_expid, tag_snap_id in tag_snap_pairs:
            for enlarge_bbox in [2.0]:
                res = [enlarge_bbox, tag_expid, tag_snap_id]
                res.extend(eval_classifier(gt_dataset_name, split, version, det_expid, tag_expid,
                        tag_snap_id, labelmap=labelmap, iou_thres=0.5, enlarge_bbox=enlarge_bbox))
                # print(res)
                yield res
    tsv_writer(gen_rows(), os.path.join(output_root, 'tmp.tsv'))


if __name__ == "__main__":
    init_logging()
    # main()
    test_eval_classifier()
