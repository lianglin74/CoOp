#!python2
import os.path as op
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import math;
import json;
import base64;
import progressbar;

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.fromstring(jpgbytestring, np.uint8)
    try:
        return cv2.imdecode(nparr, -1);
    except:
        return None;
        
def imgdet(net, class_map, imstr, max_per_image=100, thresh=0.05, nms_thresh=0.3):
    # Load the image
    im = img_from_base64(imstr);
    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(net, im)
    class_num = scores.shape[1];        #first class is background
    
    # all detections are collected into:
    #    all_boxes[cls] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[] for _ in  xrange(class_num)]  
    # skip j = 0, because it's the background class
    for j in range(1,class_num):
        inds = np.where(scores[:, j] > thresh)[0]
        cls_scores = scores[inds, j]
        cls_boxes = boxes[inds, j*4:(j+1)*4]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(cls_dets, nms_thresh)
        all_boxes[j] = cls_dets[keep, :]
    
    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][:, -1] for j in xrange(1, class_num)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image];
            for j in xrange(1, class_num):
                keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                all_boxes[j] = all_boxes[j][keep, :]
    det_results = [];
    for j in xrange(1, class_num):
        for rect in all_boxes[j]:
            crect = dict();
            crect['rect'] = [float(x) for x in list(rect[:4])];
            crect['class'] = class_map[j];
            crect['conf'] = float(rect[4]);
            det_results += [crect];
    return json.dumps(det_results);

class FileProgressingbar:
    fileobj = None;
    pbar = None;
    def __init__(self,fileobj):
        fileobj.seek(0,os.SEEK_END);
        flen = fileobj.tell();
        fileobj.seek(0,os.SEEK_SET);
        self.fileobj = fileobj;
        widgets = ['Test: ', progressbar.AnimatedMarker(),' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(widgets=widgets, maxval=flen).start()
    def update(self):
        self.pbar.update(self.fileobj.tell());
        
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--net', dest='net', help='Network to use ' )
    parser.add_argument('--intsv', required=True,   help='input tsv file for images, col_0:key, col_1:imgbase64')
    parser.add_argument('--colkey', required=False, type=int, default=0,  help='key col index');
    parser.add_argument('--colimg', required=False, type=int, default=1,  help='imgdata col index');
    parser.add_argument('--outtsv', required=False, default="",  help='output tsv file with roi info')    
    args = parser.parse_args()
    return args

def tsvdet(caffemodel, intsv_file, key_idx,img_idx,outtsv_file):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    prototxt = op.splitext(caffemodel)[0] + '.prototxt';
    cmapfile = op.splitext(caffemodel)[0] + '.labelmap';
    if not os.path.isfile(caffemodel) :
        raise IOError(('{:s} not found.').format(caffemodel))
    if not os.path.isfile(prototxt) :
        raise IOError(('{:s} not found.').format(prototxt))    
    cmap = ['background'];
    with open(cmapfile,"r") as tsvIn:
        for line in tsvIn:
            cmap +=[line.split("\t")[0]];
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print ('\n\nLoaded network {:s}'.format(caffemodel));
    count = 0;
    with open(outtsv_file,"w") as tsv_out:
        with open(intsv_file,"r") as tsv_in :
            bar = FileProgressingbar(tsv_in);
            for line in tsv_in:
                cols = [x.strip() for x in line.split("\t")];
                if len(cols)> 1:
                    results = imgdet(net, cmap, cols[img_idx])
                    tsv_out.write(cols[key_idx] + "\t" + results+"\n")
                    count += 1;
                bar.update();
    return count;
    
if __name__ == '__main__':
    args = parse_args()
    outtsv_file = args.outtsv if args.outtsv!="" else os.path.splitext(args.intsv)[0]+".eval";    
    caffemodel = args.net;

    if args.gpu_id<0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
        
    tsvdet(caffemodel, args.intsv, args.colkey, args.colimg, outtsv_file);
