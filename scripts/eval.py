import _init_paths
import sys,os, os.path as op
import argparse
import glob
import time
import numpy as np
import base64
import re
import cv2
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import math
import json
import tsvdet,deteval
from fast_rcnn.config import cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=True, help="data folder")
    parser.add_argument("-n", "--net", required=True, help="model name, e.g., zf.")
    parser.add_argument("-e", "--expid", required=True, help="experiment id.")
    parser.add_argument("-t", "--iteration", required=True, type=int, nargs="+", help="the iteration count of the snapshot, users can pass in [0] to evaluate all snapshots.")
    parser.add_argument("-g", "--gpu", required=False, default=0, type=int, help="the gpu device ids")
    parser.add_argument("--detect_only", required=False, default=False, type=bool, help="Switch true to predict result only without print out.")
    parser.add_argument("--eval_only", required=False, default=False, type=bool, help="Switch true to output report only.")
    parser.add_argument("--precth", required=False, type=float, nargs="+", default=[0.8,0.9,0.95], help="get precision, recall, threshold above given precision threshold")
    parser.add_argument("--ovth", required=False, type=float, nargs="+", default=[0.3,0.4,0.5], help="get precision, recall, threshold above given precision threshold")
                        
    return parser.parse_args()

if __name__ == "__main__":
    cmd = parse_args()
    gpuid = int(cmd.gpu)
    caffe.set_mode_gpu()
    caffe.set_device(gpuid)
    cfg.GPU_ID = gpuid

    path_env = tsvdet.setup_paths(cmd.net, cmd.data, cmd.expid)
    modelname = cmd.net
    datafolder = path_env["data"]
    jobfolder = path_env["output"]
    testdata = op.join(datafolder, "test.tsv")
    proto = op.join(jobfolder, "test_cpp.prototxt")
    labelmap = op.join(datafolder, "labelmap.txt")

    if cmd.iteration == [0]:
        train_snapshots = glob.glob(
            op.join(jobfolder, 
                    "snapshot", 
                    "%s_faster_rcnn_iter_*.caffemodel" % modelname))
        iters = map(
            lambda x: int(
                re.findall(r"faster_rcnn_iter_(\d+).caffemodel", x)[0]),
            train_snapshots)
        iters = sorted(iters)
    else:
        iters = cmd.iteration
    print("Evaluate models on iterations: %s..." % str(iters))

    models = []
    out_tsvs = []

    for iter_idx in iters:
        models.append(
            op.join(jobfolder, "snapshot",
                    "%s_faster_rcnn_iter_%d.caffemodel" % (
                        modelname, iter_idx)))
        out_tsvs.append(op.join(jobfolder, "test_iter_%d.tsv" % iter_idx))

    # Run detection on most models.
    if not cmd.eval_only:
        for model, out_tsv in zip(models, out_tsvs):
            start = time.time()            
            nimgs = tsvdet.tsvdet(model, testdata, 0, 2, out_tsv,
                                  proto=proto, cmap=labelmap)
            time_used = time.time() - start
            print("detect %d images, used %g s (avg: %g s)" % (
                nimgs,time_used, time_used/nimgs))

    # Run evaluation on detection results.
    if not cmd.detect_only:
        for out_tsv in out_tsvs:
            current_iter = int(re.findall(
                "test_iter_(\d+).tsv", out_tsv)[0])
            # Get mAP report from outtsv over IoU threshold 0.5.
            report = deteval.eval(testdata, out_tsv, 0.5)
            # Get Precision/Recall over threhold 0.9.
            th, prec, rec = deteval.get_pr(report, 0.9)
            # TODO(zhengxu): Better visualize it.
            # Print out report.
            print(current_iter, report["map"], prec, rec)
