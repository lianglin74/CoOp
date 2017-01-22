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

MAX_PER_IMAGE = 20;
CONF_THRESH = 0
NMS_THRESH = 0.3

def imgdet(net, class_map, im_file, max_per_image):
    # Load the demo image
    result_img_file = op.splitext(im_file)[0] + '.result.jpg'   
    im = cv2.imread(im_file, -1)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    # Visualize detections for each class
    class_num = scores.shape[1];        #first class is background
    # all detections are collected into:
    #    all_boxes[cls] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[] for _ in  xrange(class_num)]    
    for cls_ind in range(1,class_num):
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH);
        all_boxes[cls_ind] = dets[keep, :]
    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][:, -1] for j in xrange(1, class_num)])
        if len(image_scores) > max_per_image:
            image_thresh = max(CONF_THRESH,np.sort(image_scores)[-max_per_image]);
            for j in xrange(1, class_num):
                keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                all_boxes[j] = all_boxes[j][keep, :]
    for j in xrange(1, class_num):
        for rect in all_boxes[j]:
            cv2.rectangle(im,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,0),1);
            cv2.putText(im,class_map[j], (rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),1,cv2.CV_AA);
            print(class_map[j],rect[4])
    cv2.imwrite(result_img_file, im);


# python imgdet.py --cmap image.labelmap --gpu 0 --net synlogo.caffemodel  --prototxt test.prototxt  --image test.jpg
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net', dest='net', help='Network to use ' )
    parser.add_argument('--image', required=True, help='input images')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    caffemodel = args.net;
    prototxt = op.splitext(caffemodel)[0] + '.prototxt';
    cmapfile = op.splitext(caffemodel)[0] + '.labelmap';
    cmap = ['background'];
    with open(cmapfile,"r") as tsvIn:
        for line in tsvIn:
            cmap +=[line.split("\t")[0]];
    if not os.path.isfile(caffemodel) :
        raise IOError(('{:s} not found.').format(caffemodel))
    if not os.path.isfile(prototxt) :
        raise IOError(('{:s} not found.').format(prototxt))

    if args.gpu_id<0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print ('\n\nLoaded network {:s}'.format(caffemodel));
    imgdet(net, cmap, args.image,MAX_PER_IMAGE)
    plt.show()

