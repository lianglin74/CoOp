import _init_paths
import sys,os, os.path as op;
import argparse
import time
import numpy as np;
import base64;
import cv2
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import math;
import json;
import tsvdet,deteval
from fast_rcnn.config import cfg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobfolder', required=True, help='jobfolder')
    parser.add_argument('-t', '--iteration', required=True, type=int, nargs='+', help='the iteration count of the snapshot')
    parser.add_argument('-g', '--gpu', required=False, default=0, type=int, help='the gpu device ids')
    parser.add_argument('--precth', required=False, type=float, nargs='+', default=[0.8,0.9,0.95], help="get precision, recall, threshold above given precision threshold")
    parser.add_argument('--ovth', required=False, type=float, nargs='+', default=[0.3,0.4,0.5], help="get precision, recall, threshold above given precision threshold")
                        
    return parser.parse_args()

if __name__ == '__main__':
    cmd = parse_args();
    gpuid = int(cmd.gpu)
    caffe.set_mode_gpu()    
    caffe.set_device(gpuid)
    cfg.GPU_ID = gpuid
    jobname = op.split(cmd.jobfolder)[1];
    data,modelname = jobname.split('_')[:2]
    testdata = op.join('data',data,'test.tsv')
    proto = op.join('output',jobname,'test.prototxt')
    labelmap = op.join('data',data,'labelmap.txt')
    for iter in cmd.iteration:
        model = op.join('output',jobname,'snapshot','%s_faster_rcnn_iter_%d.caffemodel'%(modelname,iter))
        outtsv = op.join('output',jobname,'test_iter_%d.tsv'%iter);

        start = time.time()            
        nimgs = tsvdet.tsvdet(model, testdata, 0,2,outtsv, proto = proto, cmap = labelmap);
        time_used = time.time() - start
        print ( 'detect %d images, used %g s (avg: %g s)' % (nimgs,time_used, time_used/nimgs ) )  
        report = deteval.eval(testdata, outtsv, 0.5 );
        th,prec,rec = deteval.get_pr(report,0.9)
        print(iter,report['map'],prec,rec)
