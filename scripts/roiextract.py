#!python2
import os.path as op
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
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

def encode_array(nparray):
    shapestr = ",".join([str(x) for x in nparray.shape])
    array_binary = nparray.tobytes();
    b64str =  base64.b64encode(array_binary).decode()
    return ";".join([shapestr,b64str]);

def decode_array(bufferstr) :
    (shapestr,b64str) = [x.strip() for x in bufferstr.split(";")];
    arrayshape = [int(x) for x in shapestr.split(",")];
    array_binary = base64.b64decode(b64str);
    nparray = np.fromstring(array_binary, dtype=np.dtype('float32'));
    return nparray.reshape(arrayshape);

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.fromstring(jpgbytestring, np.uint8)
    try:
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR);
    except:
        return None;

def postfilter(scores, boxes, class_map, max_per_image=100, thresh=0.05, nms_thresh=0.3):
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
    parser.add_argument('--colimg', required=False, type=int, default=2,  help='imgdata col index');
    parser.add_argument('--outtsv', required=False, default="",  help='output tsv file with roi info')
    parser.add_argument('--count',  required=False, default=300, type=int, help='number of rois outputed by RPN')
    args = parser.parse_args()
    return args

def im_list_to_blob(ims):
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob
    
def _get_image_blob(im):
    #Converts an image into a network input.
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = min (float(target_size) / float(im_size_min), float(cfg.TEST.MAX_SIZE) / float(im_size_max));    
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR) if im_scale!=1.0 else np.copy(im_orig);
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
        im_scale_factors.append(1.0)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)
    
def tsvdet(caffemodel, intsv_file, key_idx,img_idx,outtsv_file, **kwargs):
    prototxt = op.splitext(caffemodel)[0] + '.prototxt' if 'proto' not in kwargs else kwargs['proto'];
    cmapfile = op.splitext(caffemodel)[0] + '.labelmap' if 'cmap' not in kwargs else kwargs['cmap'];
    if not os.path.isfile(caffemodel) :
        raise IOError(('{:s} not found.').format(caffemodel))
    if not os.path.isfile(prototxt) :
        raise IOError(('{:s} not found.').format(prototxt))
    cmap = ['background'];
    with open(cmapfile,"r") as tsvIn:
        for line in tsvIn:
            cmap +=[line.split("\t")[0].strip()];
    count = 0;
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    #print ('\n\nLoaded network {:s}'.format(caffemodel));
    with open(outtsv_file,"w") as tsv_out:
        with open(intsv_file,"r") as tsv_in :
            bar = FileProgressingbar(tsv_in);
            for line in tsv_in:
                cols = [x.strip() for x in line.split("\t")];
                if len(cols)> 1:
                    # Load the image
                    im = img_from_base64(cols[img_idx]);
                    """Convert an image and RoIs within that image into network inputs."""
                    blobs = {'data' : None, 'rois' : None}
                    blobs['data'], im_scales = _get_image_blob(im)
                    im_blob = blobs['data']
                    blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)
                    # reshape network inputs
                    net.blobs['data'].reshape(*(blobs['data'].shape))
                    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
                    # do forward
                    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False), 'im_info' : blobs['im_info'].astype(np.float32, copy=False)}
                    blobs_out = net.forward(**forward_kwargs)
                    rois = net.blobs['rois'].data.copy()
                    #rois_score = net.blobs['rois_score'].data.copy()
                    # unscale back to raw image space
                    roiboxes = rois[:, 1:5] / im_scales[0]
                    crects = [];
                    for i in range(min(roiboxes.shape[0], args.count)):
                        crects += [{'rect':[float(x) for x in list(roiboxes[i,:])] }];
                    tsv_out.write(cols[key_idx] + "\t" + json.dumps(crects)+"\n")
                    count += 1;
                bar.update();
    caffe.print_perf(count);
    
if __name__ == '__main__':
    args = parse_args()
    outtsv_file = args.outtsv if args.outtsv!="" else os.path.splitext(args.intsv)[0]+".rois";
    caffemodel = args.net;
    cfg['TEST'].RPN_POST_NMS_TOP_N = args.count;    
    if args.gpu_id<0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    tsvdet(caffemodel, args.intsv, args.colkey, args.colimg, outtsv_file);
