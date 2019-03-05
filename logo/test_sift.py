import os
import os.path as op
import time
import numpy as np
import cv2
import datetime

from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
import copy
from tqdm import tqdm
import glob
import logging

import _init_paths

from scripts.tsv_io import TSVDataset
from scripts.qd_common import init_logging, worth_create, json_dump
from scripts.yolotrain import yolo_predict
from scripts.pytablemd import write_tablemd

from logo.tagging.utils.averagemeter import AverageMeter
from logo.metric import evaluate_detection, parse_eval_file, evaluate_two_stage, evaluate_detector, draw_pr_curve
from logo.sift import sift_extract_process, sift_test_process, sift_load_process

rootpath = "/raid/data/logo_result_cs/"
iou_thres = [0.5]

def evaluation_sift(test_data, test_split, det_expid, train_data, train_split, _debug = False, is_force_extract_feature = True):
    file_dict = {}
    eval_result = []

    outdir = os.path.join(rootpath, "{}/deteval".format(det_expid))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    region_proposal_file, _ = yolo_predict(full_expid=det_expid, test_data=test_data, test_split=test_split)
    dst_file = op.join('data', test_data, "{}.{}.tsv".format(test_split, 'rp'))

    if worth_create(region_proposal_file, dst_file):
        import shutil
        shutil.copy2(region_proposal_file, dst_file)
        logging.info("{}:{}===>{}:{}".format("src", region_proposal_file, "dst", dst_file))

    # pred_file = sift_test_mp(train_feature_set, train_dataset, train_split, region_proposal_file, test_data, test_split, 'label')
    # fname = os.path.join(outdir, "{}.{}.sift.gt.det.eval".format(train_dataset, train_split))
    # eval_file = evaluate_detection(test_data, test_split, pred_file, fname)
    # eval_result.extend(parse_eval_file(eval_file))

    # change to np.array to send as param
    # np_feature_set = []
    # for train_feature in train_feature_set:
    #     nparray = decode_array(train_feature[1])
    #     kp, des = load_list_to_sift(nparray)
    #     np_feature_set.append((train_feature[0], (kp, des), train_feature[2], train_feature[3], train_feature[4]))

    train_feature_set_nparr = sift_extract_process(train_data, train_split, 224, is_force_extract_feature)
    # sift_extract_process(test_data, test_split, -1, True)

    pred_file = sift_test_process(train_feature_set_nparr, train_data, train_split, region_proposal_file, test_data, test_split, 'label', _debug, True, 1)
    logging.info("{}".format(pred_file))

    short_fname = "{}.{}.{}.{}.sift.gt.det.eval".format(train_data, train_split, test_data, test_split)
    fname = os.path.join(outdir, short_fname)
    eval_file = evaluate_detection(test_data, test_split, pred_file, fname)
    eval_result.extend(parse_eval_file(eval_file))
    file_dict[short_fname] = eval_file

    pred_file = sift_test_process(train_feature_set_nparr, train_data, train_split, region_proposal_file, test_data, test_split, 'rp', _debug, True, 3)

    logging.info("{}".format(pred_file))
    short_fname = "{}.{}.{}.{}.sift.rp.det.eval".format(train_data, train_split, test_data, test_split)
    fname = os.path.join(outdir, short_fname)
    eval_file = evaluate_detection(test_data, test_split, pred_file, fname)
    eval_result.extend(parse_eval_file(eval_file))
    file_dict[short_fname] = eval_file


    file_dict = dict(sorted(file_dict.items()))
    draw_pr_curve(file_dict, [0.5], os.path.join(outdir, "pr_curve_{}_{}_{}_{}.png".format(train_data, train_split, test_data, test_split)))

    return eval_result

def main():

    records = []

    det1_expid = "brand1048_darknet19_448_B_noreorg_rotate10_Init.best_model8022_extraConvKernel.1.3.1_TsvBoxSamples50ExtraConvGroups1_4_1EffectBatchSize128"
    det2_expid = "TaxLogoV1_1_darknet19_448_C_Init.best_model9748_maxIter.50eEffectBatchSize128_bb_only"
    det3_expid = "TaxLogoV1_7_darknet19_448_C_Init.best_model9748_maxIter.75eEffectBatchSize128_bb_only"
    det4_expid = "brand1048Clean_net_RongFasterRCNN"
    
    det5_expid = "Tax1300V14.4_0.0_0.0_darknet19_448_C_Init.best_model6933_maxIter.10eEffectBatchSize128LR7580_bb_only"

    tag1_expid = "gt_only"
    tag2_expid = "pretrained_0.1"
    tag3_expid = "ccs_code_fix"
    tag4_expid = "ccs_old"
    tag5_expid = "pretrained_0.1_old"

    # Exp 1
    train_data = "Flickr32PrototypeLogos"
    train_split = "train"

    test_data = "FlickrLogos-32"
    test_split = "test"
    records.append(["Train:{}_{}, Test:{}_{}".format(train_data, train_split, test_data, test_split)] + 
        evaluation_sift(test_data, test_split, det3_expid, train_data, train_split, True))

    # Exp 2
    train_data = "logo40"
    train_split = "train"
    
    test_data = "logo40"
    test_split = "test"

    records.append(["Train:{}_{}, Test:{}_{}".format(train_data, train_split, test_data, test_split)] + 
        evaluation_sift(test_data, test_split, det3_expid, train_data, train_split, False))

#==============================================================================#

    headings = ["Methods"] + ["mAP@{}".format(iou) for iou in iou_thres] * (len(records[0]) - 1)
    fpath = os.path.join(rootpath, "table.txt")
    fields = range(len(records[0]))
    align = [('^', '<')] + [('^', '^')]*(len(headings) - 1)

    # logging.info("records:{}".format(records))
    # logging.info("headings:{}".format(headings))
    # logging.info("fields:{}".format(fields))
    # logging.info("align:{}".format(align))

    with open(fpath, 'a+') as fp:
        timestring = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fp.write("{}\n".format(timestring))
        write_tablemd(fp, records, fields, headings, align)
    return
#==============================================================================#

if __name__ == "__main__":
    init_logging()
    # train_feature_set = sift_extract_process("logo40", "train")
    # logging.info("len(train_feature_set):{}".format(len(train_feature_set)))
    main()
