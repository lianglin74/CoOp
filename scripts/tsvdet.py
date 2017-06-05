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
from datetime import datetime

def createpath( pparts ):
    fpath = op.join(*pparts);
    if not os.path.exists(fpath):
        os.makedirs(fpath);
    return fpath;   
    
def setup_paths(basenet, dataset, expid):
    proj_root = op.dirname(op.dirname(op.realpath(__file__)));
    model_path = op.join (proj_root,"models");
    data_root = op.join(proj_root,"data");
    data_path = op.join(data_root,dataset);
    basemodel_file = op.join(model_path ,basenet+'.caffemodel');
    default_cfg = op.join(model_path,"faster_rcnn_end2end.yml")
    output_path = createpath([proj_root,"output","_".join([dataset,basenet,expid])]);
    solver_file = op.join(output_path,"solver.prototxt");
    snapshot_path = createpath([output_path,"snapshot"]);
    DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = op.join(output_path, '%s_%s.log' %(basenet, DATE));
    caffe_log_file = op.join(output_path, '%s_caffe_'%(basenet));
    model_pattern = "%s/%s_faster_rcnn_iter_*.caffemodel"%(snapshot_path,basenet.split('_')[0].lower());
    deploy_path = createpath([output_path,"deploy"]);
    eval_output =  op.join(output_path, '%s_%s_testeval.tsv' %(basenet, DATE));
    return { "snapshot":snapshot_path, "solver":solver_file, "log":log_file, "output":output_path, "cfg":default_cfg, 'data_root':data_root, 'data':data_path, 'basemodel':basemodel_file, 'model_pattern':model_pattern, 'deploy':deploy_path, 'caffe_log':caffe_log_file, 'eval':eval_output};

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
    # all_boxes[cls] = N x 5 array of detections in (x1, y1, x2, y2, score)
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

def tsvdet(caffemodel, intsv_file, key_idx,img_idx,outtsv_file, **kwargs):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
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
    print ('\n\nLoaded network {:s}'.format(caffemodel));
    with open(outtsv_file,"w") as tsv_out:
        with open(intsv_file,"r") as tsv_in :
            bar = FileProgressingbar(tsv_in);
            for line in tsv_in:
                cols = [x.strip() for x in line.split("\t")];
                if len(cols)> 1:
                    # Load the image
                    im = img_from_base64(cols[img_idx]);
                    # Detect all object classes and regress object bounds
                    scores, boxes = im_detect(net, im)
                    #tsv_out.write('\t'.join([cols[key_idx], encode_array(scores), encode_array(boxes)])+"\n")
                    results = postfilter(scores,boxes, cmap)
                    tsv_out.write(cols[key_idx] + "\t" + results+"\n")
                    count += 1;
                bar.update();
    caffe.print_perf(count);
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
