import collections
import json
import logging
import numpy as np
import os

import _init_paths
from logo.classifier import CropTaggingWrapper
from scripts.qd_common import init_logging, worth_create, read_to_buffer
from scripts.tsv_io import tsv_reader, tsv_writer, TSVDataset
from scripts.deteval import deteval_iter
from scripts.yolotrain import yolo_predict
from scripts.pytablemd import write_tablemd

trained_dataset = "brand1048"
new_dataset = "logo40"
pair_dataset = "logo40_pair"
data_split = "test"
trained_gt = "/raid/data/brand1048/test.label.tsv"
new_gt = "/raid/data/logo40/test.label.tsv"

rootpath = "/raid/data/brand_output/"
iou_thres = [0.5]


def main():
    records = []
    headings = ["Methods"] + ["mAP@{}".format(iou) for iou in iou_thres] * 4

    det1_expid = "brand1048_darknet19_448_B_noreorg_rotate10_Init.best_model8022_extraConvKernel.1.3.1_TsvBoxSamples50ExtraConvGroups1_4_1EffectBatchSize128"
    tag1_expid = "gt_only"
    det2_expid = "TaxLogoV1_1_darknet19_448_C_Init.best_model9748_maxIter.50eEffectBatchSize128_bb_only"
    tag2_expid = "pretrained_0.1"
    tag3_expid = "ccs_code_fix"

    records.append(["1k logo detector"] + evaluate_detector(det1_expid))
    records.append(["logo/non-logo detector"] + evaluate_detector(det2_expid))

    records.append(["two-stage"] + evaluate_two_stage(det2_expid, tag2_expid))
    records.append(["two-stage-ccs"] + evaluate_two_stage(det2_expid, tag3_expid))

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
    eval_file = evaluate_detection(trained_dataset, data_split, pred_file, os.path.join(outdir, "{}.{}.pretrained.region.eval".format(trained_dataset, data_split)), region_only=True)
    eval_res.extend(parse_eval_file(eval_file))
    eval_file = evaluate_detection(trained_dataset, data_split, pred_file, os.path.join(outdir, "{}.{}.pretrained.det.eval".format(trained_dataset, data_split)))
    eval_res.extend(parse_eval_file(eval_file))

    # feature matching on unknown classes
    pred_file = croptagger.predict_on_unknown_class(new_dataset, data_split)
    eval_file = evaluate_detection(new_dataset, data_split, pred_file, os.path.join(outdir, "{}.{}.new.region.eval".format(new_dataset, data_split)), region_only=True)
    eval_res.extend(parse_eval_file(eval_file))
    eval_file = evaluate_detection(new_dataset, data_split, pred_file, os.path.join(outdir, "{}.{}.new.det.eval".format(new_dataset, data_split)))
    eval_res.extend(parse_eval_file(eval_file))

    # metrics for featurizer/recognition
    # ROC for pair comparision (positive pairs are from the same class, negative pairs are from differen classes)
    croptagger.compare_pairs(pair_dataset, data_split)
    # precision@k on ground truth regions (match features of canonical images and real images)
    croptagger.predict_on_unknown_class(new_dataset, data_split, region_source="gt")
    return eval_res


def evaluate_detector(det_expid):
    outdir = os.path.join(rootpath, "{}/deteval".format(det_expid))
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    eval_res = []
    # pretrained
    pred_file, _ = yolo_predict(full_expid=det_expid, test_data=trained_dataset, test_split=data_split)
    eval_file = evaluate_detection(trained_dataset, data_split, pred_file, os.path.join(outdir, "{}.{}.pretrained.region.eval".format(trained_dataset, data_split)), region_only=True)
    eval_res.extend(parse_eval_file(eval_file))
    eval_file = evaluate_detection(trained_dataset, data_split, pred_file, os.path.join(outdir, "{}.{}.pretrained.det.eval".format(trained_dataset, data_split)))
    eval_res.extend(parse_eval_file(eval_file))

    # logo/non-logo
    pred_file, _ = yolo_predict(full_expid=det_expid, test_data=new_dataset, test_split=data_split)
    eval_file = evaluate_detection(new_dataset, data_split, pred_file, os.path.join(outdir, "{}.{}.new.region.eval".format(new_dataset, data_split)), region_only=True)
    eval_res.extend(parse_eval_file(eval_file))
    # detector can not detect unknown categories
    eval_res.extend(["N/A"]*len(iou_thres))
    return eval_res


def evaluate_detection(dataset_name, split, pred_file, outfile, region_only=False):
    """ Calculates mAP for detection results (label+bbox or bbox only)
    """
    logging.info("Evaluate detection on: {} {}".format(dataset_name, split))
    dataset = TSVDataset(dataset_name)
    truth_iter = dataset.iter_data(split, 'label', version=0)
    return deteval_iter(truth_iter=truth_iter, dets=pred_file, report_file=outfile, region_only=region_only)


def parse_eval_file(eval_file):
    eval_res= read_to_buffer(eval_file)
    logging.info('json parsing...')
    eval_res = json.loads(eval_res)
    return [round(eval_res["overall"][str(iou)]["map"], 4) for iou in iou_thres]


def evaluate_region():
    """ Calculates mAP for region proposal results (only bbox)
    """
    dataset_name = "logo40"
    split = "test"
    version = -1

    region_only = True
    #det_expid = "TaxLogoV1_1_darknet19_448_C_Init.best_model9748_maxIter.50eEffectBatchSize128_bb_only"
    det_expid = "brand1048_darknet19_448_B_noreorg_rotate10_Init.best_model8022_extraConvKernel.1.3.1_TsvBoxSamples50ExtraConvGroups1_4_1EffectBatchSize128"

    eval_type = "region" if region_only else "det"
    method_type = "yolo"
    rootpath = os.path.join("/home/xiaowh/brand/output/", det_expid)
    report_file = os.path.join(rootpath, "{}.{}.{}.eval.{}".format(dataset_name, split, method_type, eval_type))

    detpred_file, deteval_file = yolo_predict(full_expid=det_expid, test_data=dataset_name, test_split=split)

    dataset = TSVDataset(dataset_name)
    truth_iter = dataset.iter_data(split, 'label', version=version)
    deteval_iter(truth_iter=truth_iter, dets=detpred_file, report_file=report_file, region_only=region_only)


if __name__ == "__main__":
    init_logging()
    main()