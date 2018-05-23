#!python2
from process_image import draw_bb, show_image
import cPickle as pkl
import os.path as op
import time
import logging
import _init_paths
# import sys
# sys.path.insert(0, r'd:\github\caffe-msrccs\pythond')
import numpy as np
import caffe, os, sys, cv2
import argparse
import numpy as np
import base64
import progressbar 
import json
import matplotlib.pyplot as plt
import fast_rcnn
from fast_rcnn.nms_wrapper import nms

from qd_common import img_from_base64, FileProgressingbar
import multiprocessing as mp
from fast_rcnn.nms_wrapper import nms

from tsv_io import tsv_writer

def parse_args(arg_list):
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='dnn finetune')
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('-g', '--gpus', help='GPU device id to use [0], e.g. -g 0 1 2 3.',
            type=int,
            nargs='+')
    parser.add_argument('--net', dest='net', help='Network to use')
    parser.add_argument('--model', required=False, default='', help='caffe model file')
    parser.add_argument('--intsv', required=True,   help='input tsv file for images, col_0:key, col_1:imgbase64')
    parser.add_argument('--colkey', required=False, type=int, default=0,  help='key col index')
    parser.add_argument('--colimg', required=False, type=int, default=1,  help='imgdata col index')
    parser.add_argument('--outtsv', required=False, default="",  help='output tsv file with roi info')
    parser.add_argument('--mean', required=False, default='104,117,123', help='pixel mean value')
    args = parser.parse_args(arg_list)

    return args

def vis_detections2(im, prob, bboxes, labelmap, thresh=0.3, save_filename=None):
    result = result2bblist2(im,  prob, bboxes, labelmap, thresh)
    draw_bb(im, [r['rect'] for r in result], [r['class'] for r in result],
            [r['conf'] for r in result])
    show_image(im)

def vis_detections(im, prob, bboxes, labelmap, thresh=0.3, save_filename=None):
    """Visual debugging of detections."""
    im = im[:, :, (2, 1, 0)]
    plt.cla()
    fig = plt.imshow(im)

    for i, box in enumerate(bboxes):
        for j in range(prob.shape[1] - 1):
            if prob[i, j] < thresh:
                continue;
            score = prob[i, j]
            cls = j
            x,y,w,h = box
        
            im_h, im_w = im.shape[0:2]
            left  = (x-w/2.)
            right = (x+w/2.)
            top   = (y-h/2.)
            bot   = (y+h/2.)

            left = max(left, 0)
            right = min(right, im_w - 1)
            top = max(top, 0)
            bot = min(bot, im_h - 1)

            plt.gca().add_patch(
                plt.Rectangle((left, top),
                                right - left,
                                bot - top, fill=False,
                                edgecolor='g', linewidth=3)
                )
            plt.text(float(left), float(top - 10), '%s: %.3f'%(labelmap[cls], score), color='darkgreen', backgroundcolor='lightgray')
            #plt.title('{}  {:.3f}'.format(class_name, score))

    if save_filename is None:
        plt.show()
    else:
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(save_filename, bbox_inches='tight', pad_inches = 0)

def load_labelmap_list(filename):
    labelmap = []
    with open(filename) as fin:
        labelmap += [line.rstrip() for line in fin]
    return labelmap

def im_rescale(im, target_size):
    im_size_min = min(im.shape[0:2])
    im_size_max = max(im.shape[0:2])
    im_scale = float(target_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return im

def load_image_data(filename):
    import struct
    with open(filename, 'rb') as f:
        w = struct.unpack('i', f.read(4))[0]
        h = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32)
        data = np.reshape(data, [3, h, w])
        data = data[::-1, ...]
    return data

def xywh_to_xyxy(bbox):
    result = np.zeros_like(bbox)
    result[:, 0] = bbox[:, 0] - bbox[:, 2] / 2.
    result[:, 2] = bbox[:, 0] + bbox[:, 2] / 2.
    result[:, 1] = bbox[:, 1] - bbox[:, 3] / 2.
    result[:, 3] = bbox[:, 1] + bbox[:, 3] / 2.
    return result

def im_classify(caffe_net, im, pixel_mean, **kwarg):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pixel_mean
    
    scale = kwarg.get('scale', 1)
    if scale != 1:
        blob *= scale
    style = kwarg.get('predict_style', None)
    if style == 'tsvdatalayer':
        scale_target = 0.875
        height, width = im_orig.shape[:2]
        crop_size = (int)(min(height, width) * scale_target)
        x_off = (width - crop_size) / 2
        y_off = (height - crop_size) / 2
        crop_h = crop_size
        crop_w = crop_size
        im_orig = im_orig[y_off: y_off + crop_size, x_off : x_off + crop_size, :]
        network_input_size = kwarg.get('network_input_size', 224)
        im_orig = cv2.resize(im_orig, (network_input_size, network_input_size))

    channel_swap = (2, 0, 1)
    blob = im_orig.transpose(channel_swap)

    caffe_net.blobs['data'].reshape(1, *blob.shape)
    caffe_net.blobs['data'].data[...] = blob.reshape(1, *blob.shape)

    caffe_net.forward()

    prob = caffe_net.blobs['prob'].data[0]

    return prob

def correct_labels_to_blob(labels, 
        orig_shape, 
        network_input_size, 
        label_map, max_truth):
    orig_min = min(orig_shape)
    orig_max = max(orig_shape)
    ratio = float(network_input_size) / orig_max
    new_h = orig_shape[0] * ratio
    new_w = orig_shape[1] * ratio
    left = (network_input_size - new_w) / 2
    top = (network_input_size - new_h) / 2
    blob = np.zeros((1, max_truth * 5))
    for i, label in enumerate(labels):
        x1, y1, x2, y2 = label['rect']
        x1 = x1 * ratio + left
        x2 = x2 * ratio + left
        y1 = y1 * ratio + top
        y2 = y2 * ratio + top
        if i >= max_truth:
            logging.info('max_truth is small')
            break
        blob[0, i * 5 + 0] = (x1 + x2) / 2.0 / float(network_input_size)
        blob[0, i * 5 + 1] = (y1 + y2) / 2.0 / float(network_input_size)
        blob[0, i * 5 + 2] = (x2 - x1) / float(network_input_size)
        blob[0, i * 5 + 3] = (y2 - y1) / float(network_input_size)
        cls = label_map.index(label['class'])
        assert cls >= 0
        blob[0, i * 5 + 4] = cls
    return blob

def prepare_net_input(im, pixel_mean, network_input_size, **kwargs):
    im_orig = im.astype(np.float32, copy=True)
    if pixel_mean and pixel_mean[0] != 0:
        im_orig -= pixel_mean
    
    im_resized = im_rescale(im_orig, network_input_size)
    new_h, new_w = im_resized.shape[0:2]
    left = (network_input_size - new_w) / 2
    right = network_input_size - new_w - left
    top = (network_input_size - new_h) / 2
    bottom = network_input_size - new_h - top

    im_squared  = cv2.copyMakeBorder(im_resized, top=top, bottom=bottom, left=left, right=right, borderType= cv2.BORDER_CONSTANT, 
            value=[0, 0, 0])

    # change blob dim order from h.w.c to c.h.w
    channel_swap = (2, 0, 1)
    blob = im_squared.transpose(channel_swap)
    if pixel_mean[0] == 0:
        blob /= 255.

    #net.blobs['data'].reshape(1, *blob.shape)
    #net.blobs['data'].data[...]=blob.reshape(1, *blob.shape)
    #net.blobs['im_info'].reshape(1,2)
    #net.blobs['im_info'].data[...] = (im_orig.shape[0:2],)

    blob_label = None

    if 'gt_labels' in kwargs:
        labels = kwargs['gt_labels']
        blob_label = correct_labels_to_blob(labels, im_orig.shape[:2], 
                network_input_size, kwargs['label_map'], kwargs['yolo_max_truth'])
        #net.blobs['label'].data[...] = blob_label

    return blob, im_orig.shape[0:2], blob_label

def im_multi_scale_detect(caffe_net, im, pixel_mean, gpu,
        test_input_sizes=[416], **kwargs):
    all_prob = []
    all_bbox = []
    for test_input_size in test_input_sizes:
        prob, bbox = im_detect(caffe_net, im, pixel_mean, test_input_size, **kwargs)
        #if len(test_input_sizes) == 1:
            #return prob, bbox
        all_prob.append(np.copy(prob))
        all_bbox.append(np.copy(bbox))
    prob = np.concatenate(all_prob)
    bbox = np.concatenate(all_bbox)
    bbox_xyxy = xywh_to_xyxy(bbox)

    if kwargs.get('class_specific_nms', True):
        for c in range(prob.shape[1] - 1):
            nms_input = np.concatenate((bbox_xyxy, prob[:, c][:, np.newaxis]), axis=1)
            keep = nms(nms_input, 0.45, False, device_id=gpu)
            removed = np.ones(prob.shape[0], dtype=np.bool)
            removed[keep] = False
            prob[removed, c] = 0
    else:
        nms_input = np.concatenate((bbox_xyxy, prob[:, -1][:, np.newaxis]), axis=1)
        keep = nms(nms_input, 0.45, False, device_id=gpu)
        removed = np.ones(prob.shape[0], dtype=np.bool)
        removed[keep] = False
        prob[removed, :] = 0
    return prob, bbox
        
def im_detect(caffe_net, im, pixel_mean, network_input_size=416, stat=None, **kwargs):
    if stat != None:
        time_start = time.time()
    im_orig = im.astype(np.float32, copy=True)
    if stat != None:
        time_curr = time.time()
        stat['convert_image'] = time_curr - time_start
        time_start = time_curr
    if pixel_mean and pixel_mean[0] != 0:
        im_orig -= pixel_mean
    if stat != None:
        time_curr = time.time()
        stat['minum_mean'] = time_curr - time_start
        time_start = time_curr

    if kwargs.get('yolo_test_maintain_ratio'):
        h, w = im_orig.shape[0:2]
        alpha = network_input_size / np.sqrt(h * w)
        height2 = int(np.round(alpha * h))
        width2 = int(np.round(alpha * w))
        if h > w:
            network_input_height = (height2 + 31) / 32 * 32
            network_input_width = ((network_input_height * w + h - 1) / h
                    + 31) / 32 * 32
        else:
            network_input_width = (width2 + 31) / 32 * 32
            network_input_height = ((network_input_width * h + w - 1) / w +
                    31) / 32 * 32
        network_input_size = max(network_input_width, 
                network_input_height)
    else:
        network_input_width = network_input_size
        network_input_height = network_input_size

    im_resized = im_rescale(im_orig, network_input_size)
    if stat != None:
        time_curr = time.time()
        stat['rescale'] = time_curr - time_start
        time_start = time_curr

    new_h, new_w = im_resized.shape[0:2]
    left = (network_input_width - new_w) / 2
    right = network_input_width - new_w - left
    top = (network_input_height - new_h) / 2
    bottom = network_input_height - new_h - top
    im_squared  = cv2.copyMakeBorder(im_resized, top=top, bottom=bottom, left=left, right=right, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    if stat != None:
        time_curr = time.time()
        stat['make_boarder'] = time_curr - time_start
        time_start = time_curr
    # change blob dim order from h.w.c to c.h.w
    channel_swap = (2, 0, 1)
    blob = im_squared.transpose(channel_swap)
    if pixel_mean[0] == 0:
        blob /= 255.
    if stat != None:
        time_curr = time.time()
        stat['transpose'] = time_curr - time_start
        time_start = time_curr

    # blob = load_image_data(r'detection-image.bin')      // for parity check

    caffe_net.blobs['data'].reshape(1, *blob.shape)
    caffe_net.blobs['data'].data[...]=blob.reshape(1, *blob.shape)
    caffe_net.blobs['im_info'].reshape(1,2)
    caffe_net.blobs['im_info'].data[...] = (im_orig.shape[0:2],)
    if 'label'  in caffe_net.blobs:
        labels = kwargs['gt_labels']
        assert network_input_width == network_input_height
        blob_label = correct_labels_to_blob(labels, im_orig.shape[:2], 
                network_input_size, kwargs['label_map'], kwargs['yolo_max_truth'])
        caffe_net.blobs['label'].data[...] = blob_label
    if stat != None:
        time_curr = time.time()
        stat['feed_data'] = time_curr - time_start
        time_start = time_curr

    caffe_net.forward()

    if stat != None:
        time_curr = time.time()
        stat['net_forward'] = time_curr - time_start
        time_start = time_curr
    result = []
    if 'prob' in caffe_net.blobs:
        prob = caffe_net.blobs['prob'].data[0]
        prob = prob.reshape(-1, prob.shape[-1])
        result.append(prob)

    if 'bbox' in caffe_net.blobs:
        bbox = caffe_net.blobs['bbox'].data[0]
        assert bbox.shape[-1] == 4
        bbox = bbox.reshape(-1, 4)
        result.append(bbox)

    if stat != None:
        time_curr = time.time()
        stat['net_return'] = time_curr - time_start
        time_start = time_curr
    return result

def postfilter(im, scores, boxes, class_map, max_per_image=1000, thresh=0.005):
    class_num = scores.shape[1] - 1;        #the last one is obj_score * max_prob
    keep = np.where(scores[:, -1] > thresh)[0]
    scores = scores[keep, :]
    boxes = boxes[keep, :]

    image_scores = scores[:,-1]
    inds = np.argsort(-image_scores)
    scores = scores[inds, :]
    boxes = boxes[inds, :]

    # Limit to max_per_image detections
    if max_per_image > 0 and boxes.shape[0] > max_per_image:
        image_thresh = image_scores[max_per_image]
        keep = np.where(scores[:,-1] >= image_thresh)[0]
        scores = scores[0:max_per_image, :]
        boxes = boxes[0:max_per_image, :]

    det_results = [];
    for i, box in enumerate(boxes):
        crect = dict();

        x,y,w,h = box
        
        im_h, im_w = im.shape[0:2]
        left  = (x-w/2.);
        right = (x+w/2.);
        top   = (y-h/2.);
        bot   = (y+h/2.);

        left = max(left, 0)
        right = min(right, im_w - 1)
        top = max(top, 0)
        bot = min(bot, im_h - 1)

        crect['rect'] = map(float, [left,top,right,bot])
        cls = scores[i, 0:-1].argmax()
        crect['class'] = class_map[cls]
        crect['conf'] = float(scores[i, -1])
        det_results += [crect]
    
    return json.dumps(det_results)

def detect_image(caffe_net, im, pixel_mean, names, stat=None, thresh=0,
        yolo_tree=False):
    '''
    this is not used in evaluation
    '''
    scores, boxes = im_detect(caffe_net, im, pixel_mean, stat=stat)
    if stat:
        time_start = time.time()
    bblist = result2bblist3(im, scores, boxes, names, thresh, yolo_tree)
    if stat != None:
        time_curr = time.time()
        stat['result2bb'] = time_curr - time_start
        time_start = time_curr

    all_bb = [bb['rect'] for bb in bblist]
    all_label = [bb['class'] for bb in bblist]
    all_conf = [bb['conf'] for bb in bblist]

    return all_bb, all_label, all_conf

def result2bblist3(im, probs, boxes, class_map, thresh=0, yolo_tree=False):
    '''
    assume each box have more than one chance to be selected. 
    result2bblist2 is not good for tree-based taxonomy, but this is
    '''
    class_num = probs.shape[1] - 1;        #the last one is obj_score * max_prob
    im_h, im_w = im.shape[0:2]

    # note: when it does the nms, the last one will not be reset
    if not yolo_tree:
        idx_bbox, idx_prob = np.where(probs[:, :-1] > thresh)
    else:
        idx_bbox = np.where(probs[:, -1] > thresh)[0]
        probs = probs[idx_bbox, :]
        boxes = boxes[idx_bbox, :]
        idx_bbox, idx_prob = np.where(probs[:, :-1] > 0.01)
    unique_idx_bb = np.lib.arraysetops.unique(idx_bbox)
    unique_selected_boxes = boxes[unique_idx_bb, :]
    boxidx_to_uniquebox = {idx: i for i, idx in enumerate(unique_idx_bb)}
    transformed_boxes = np.zeros(unique_selected_boxes.shape)
    x = unique_selected_boxes[:, 0] - unique_selected_boxes[:, 2] / 2.0
    transformed_boxes[:, 0] = np.maximum(x, 0)
    x = unique_selected_boxes[:, 1] - unique_selected_boxes[:, 3] / 2.0
    transformed_boxes[:, 1] = np.maximum(x, 0)
    x = unique_selected_boxes[:, 0] + unique_selected_boxes[:, 2] / 2.0
    transformed_boxes[:, 2] = np.minimum(x, im_w - 1)
    x = unique_selected_boxes[:, 1] + unique_selected_boxes[:, 3] / 2.0
    transformed_boxes[:, 3] = np.minimum(x, im_h - 1)

    num = len(idx_prob)
    return [{'rect': map(float, transformed_boxes[boxidx_to_uniquebox[idx_bbox[i]], :]), \
            'class': class_map[idx_prob[i]], \
            'conf': float(probs[idx_bbox[i], idx_prob[i]]) } \
            for i in xrange(num) ]

def result2bblist2(im, probs, boxes, class_map, thresh=0):
    '''
    assume each box have only one chance to be selected
    '''
    class_num = probs.shape[1] - 1;        #the last one is obj_score * max_prob
    im_h, im_w = im.shape[0:2]

    # note: when it does the nms, the last one will not be reset
    idx = np.max(probs[:, :-1], axis=1) > thresh
    selected_probs = probs[idx, -1]
    selected_label_idx = np.argmax(probs[idx, :-1], axis=1)
    selected_boxes = boxes[idx, :]
    transformed_boxes = np.zeros(selected_boxes.shape)
    x = selected_boxes[:, 0] - selected_boxes[:, 2] / 2.0
    transformed_boxes[:, 0] = np.maximum(x, 0)
    x = selected_boxes[:, 1] - selected_boxes[:, 3] / 2.0
    transformed_boxes[:, 1] = np.maximum(x, 0)
    x = selected_boxes[:, 0] + selected_boxes[:, 2] / 2.0
    transformed_boxes[:, 2] = np.minimum(x, im_w - 1)
    x = selected_boxes[:, 1] + selected_boxes[:, 3] / 2.0
    transformed_boxes[:, 3] = np.minimum(x, im_h - 1)

    num = transformed_boxes.shape[0]
    return [{'rect': map(float, transformed_boxes[i, :]), \
            'class': class_map[selected_label_idx[i]], \
            'conf': float(selected_probs[i]) } \
            for i in xrange(num) ]

def result2bblist(im, probs, boxes, class_map, thresh=0):
    class_num = probs.shape[1] - 1;        #the last one is obj_score * max_prob

    det_results = [];
    for i, box in enumerate(boxes):
        if probs[i, 0:-1].max() <= thresh:
            continue;
        for j in range(class_num):
            if probs[i,j] <= thresh:
                continue

            x,y,w,h = box
        
            im_h, im_w = im.shape[0:2]
            left  = (x-w/2.);
            right = (x+w/2.);
            top   = (y-h/2.);
            bot   = (y+h/2.);

            left = max(left, 0)
            right = min(right, im_w - 1)
            top = max(top, 0)
            bot = min(bot, im_h - 1)

            crect = dict();
            crect['rect'] = map(float, [left,top,right,bot])
            crect['class'] = class_map[j]
            crect['conf'] = float(probs[i, j])
            det_results += [crect]

    return det_results

def result2json(im, probs, boxes, class_map):
    det_results = result2bblist(im, probs, boxes, class_map)
    return json.dumps(det_results)

def detprocess(caffenet, caffemodel, pixel_mean, scale, cmap, gpu, key_idx, img_idx,
        in_queue, out_queue, **kwargs):
    if gpu >= 0:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    caffe_net = caffe.Net(str(caffenet), str(caffemodel), caffe.TEST)
    
    if kwargs.get('detmodel', 'yolo') == 'yolo':
        buffer_size = max(kwargs.get('test_input_sizes', [416])) + 100
        # buffer size should be 32x
        buffer_size = int(buffer_size) / 32 * 32
        blob = np.zeros((3, buffer_size, buffer_size), dtype=np.float32)
        caffe_net.blobs['data'].reshape(1, *blob.shape)
        caffe_net.blobs['data'].data[...]=blob.reshape(1, *blob.shape)
        caffe_net.blobs['im_info'].reshape(1,2)
        caffe_net.blobs['im_info'].data[...] = (buffer_size, buffer_size)
        # pre-allocate all memory
        caffe_net.forward()
    
    last_print_time = 0
    count = 0
    while True:
        if in_queue.qsize() == 0:
            if time.time() - last_print_time > 10:
                logging.info('processor is waiting to get the data {}'.format(count))
                last_print_time = time.time()
        cols = in_queue.get()
        if cols is None:
            out_queue.put(None)
            break
        if len(cols)> 1:
            # Load the image
            im = img_from_base64(cols[img_idx])
            if im is None:
                continue
            if kwargs.get('detmodel', 'yolo') == 'yolo':
                if 'label' in caffe_net.blobs:
                    kwargs['gt_labels'] = json.loads(cols[1])
                    kwargs['label_map'] = cmap
                x = im_multi_scale_detect(caffe_net, im, pixel_mean, gpu, **kwargs)
                if len(x) == 2:
                    scores, boxes = x[0], x[1]
                    # vis_detections(im, scores, boxes, cmap, thresh=0.5)
                    results = result2json(im, scores, boxes, cmap)
            else:
                scores = im_classify(caffe_net, im, pixel_mean, scale=scale, **kwargs)
                results = ','.join(map(str, scores))
            if out_queue.full():
                if time.time() - last_print_time > 10:
                    logging.info('processor is waiting to output the result')
                    last_print_time = time.time()
            count = count + 1
            if len(kwargs.get('extract_features', '')) > 0:
                results = [cols[key_idx]]
                for k in kwargs['extract_features'].split('.'):
                    if len(caffe_net.blobs[k].data.shape) == 0:
                        results.append(base64.b64encode(pkl.dumps(caffe_net.blobs[k].data)))
                    else:
                        results.append(base64.b64encode(pkl.dumps(caffe_net.blobs[k].data[0])))
                out_queue.put(results)
            else:
                out_queue.put((cols[key_idx], results))

    logging.info('detprocess finished')

def tsvdet_iter(caffenet, caffemodel, in_rows, key_idx,img_idx, pixel_mean,
        scale, outtsv_file, **kwargs):
    '''
    the input is a generator. This should be more general than tsvdet()
    '''
    if not caffemodel:
        caffemodel = op.splitext(caffenet)[0] + '.caffemodel'
    cmapfile = kwargs['cmapfile']
    if not os.path.isfile(caffemodel) :
        raise IOError(('{:s} not found.').format(caffemodel))
    if not os.path.isfile(caffenet) :
        raise IOError(('{:s} not found.').format(caffenet))
    cmap = load_labelmap_list(cmapfile)
    count = 0
    debug = kwargs.get('debug_detect', False)

    if debug:
        #gpus = [-1]
        gpus = [0]
    else:
        gpus = kwargs.get('gpus', [0]) * 8

    in_queue = mp.Queue(10 * len(gpus));  # thread/process safe
    if debug:
        out_queue = mp.Queue()
    else:
        out_queue = mp.Queue(10 * len(gpus));
    num_worker = len(gpus)

    def reader_process(in_rows, in_queue, num_worker, img_idx):
        last_print_time = 0
        count = 0
        for cols in in_rows:
            if len(cols) > img_idx:
                if in_queue.full():
                    if time.time() - last_print_time > 10:
                        logging.info('reader is waiting. {}'.format(count))
                        last_print_time = time.time()
                count = count + 1
                in_queue.put(cols)
        for _ in xrange(num_worker):
            in_queue.put(None)  # kill all workers
        logging.info('finished reader')
    reader = mp.Process(target=reader_process, args=(in_rows, in_queue,
        num_worker, img_idx))
    reader.daemon = True
    reader.start()

    worker_pool = [];
    if debug:
        detprocess(caffenet, caffemodel,
            pixel_mean, scale, cmap, gpus[0], key_idx, 
            img_idx, in_queue, out_queue, **kwargs)
    else:
        for gpu in gpus:
            worker = mp.Process(target=detprocess, args=(caffenet, caffemodel,
                pixel_mean, scale, cmap, gpu, key_idx, 
                img_idx, in_queue, out_queue), kwargs=kwargs);
            worker.daemon = True
            worker_pool.append(worker)
            worker.start()

    assert not kwargs.get('debug_detect', False)

    outtsv_file_tmp = outtsv_file + '.tmp'
    def writer_process(out_queue, num_worker, outtsv_file_tmp):
        def yield_output():
            num_finished = 0
            last_print_time = 0
            count = 0
            while True:
                if num_finished == num_worker:
                    break
                else:
                    if out_queue.qsize() == 0:
                        if time.time() - last_print_time > 10:
                            logging.info('writer is waiting. {}'.format(count))
                            last_print_time = time.time()
                    if out_queue.empty():
                        if time.time() - last_print_time > 10:
                            logging.info('writer waiting to get the data. {}'.format(count))
                            last_print_time = time.time()
                    x = out_queue.get()
                    if x is None:
                        num_finished = num_finished + 1
                    else:
                        count = count + 1
                        yield x
        tsv_writer(yield_output(), outtsv_file_tmp)

    writer = mp.Process(target=writer_process, args=(out_queue, num_worker,
        outtsv_file_tmp))
    writer.daemon = True
    writer.start()
    
    reader.join()
    logging.info('reader finished')
    for proc in worker_pool: #wait all process finished.
        proc.join()
    logging.info('worker finished')
    writer.join()
    logging.info('writer finished')

    os.rename(outtsv_file_tmp, outtsv_file)
    #caffe.print_perf(count)

    return count

def tsvdet(caffenet, caffemodel, intsv_file, key_idx,img_idx, pixel_mean,
        scale, outtsv_file, **kwargs):
    if not caffemodel:
        caffemodel = op.splitext(caffenet)[0] + '.caffemodel'
    if 'cmapfile' not in kwargs:
        labelmapfile = 'labelmap.txt' if 'cmap' not in kwargs else kwargs['cmap']
        cmapfile = os.path.join(op.split(caffenet)[0], labelmapfile)
    else:
        cmapfile = kwargs['cmapfile']
    if not os.path.isfile(cmapfile):
        cmapfile = os.path.join(os.path.dirname(intsv_file), 'labelmap.txt')
        assert os.path.isfile(cmapfile), cmapfile
    if not os.path.isfile(caffemodel) :
        raise IOError(('{:s} not found.').format(caffemodel))
    if not os.path.isfile(caffenet) :
        raise IOError(('{:s} not found.').format(caffenet))
    cmap = load_labelmap_list(cmapfile)
    count = 0
    debug = kwargs.get('debug_detect', False)

    gpus = kwargs.get('gpus', [0])

    in_queue = mp.Queue(10 * len(gpus));  # thread/process safe
    if debug:
        out_queue = mp.Queue()
    else:
        out_queue = mp.Queue(10 * len(gpus));
    num_worker = len(gpus)

    def reader_process(intsv_file, in_queue, num_worker, img_idx):
        logging.info('start to read {}'.format(intsv_file))
        last_print_time = 0
        count = 0
        with open(intsv_file,"r") as tsv_in :
            bar = FileProgressingbar(tsv_in)
            for i, line in enumerate(tsv_in):
                cols = [x.strip() for x in line.split("\t")]
                if len(cols) > img_idx:
                    if in_queue.full():
                        if time.time() - last_print_time > 10:
                            logging.info('reader is waiting. {}'.format(count))
                            last_print_time = time.time()
                    count = count + 1
                    in_queue.put(cols)
                    bar.update()
        for _ in xrange(num_worker):
            in_queue.put(None)  # kill all workers
        logging.info('finished reader')
    reader = mp.Process(target=reader_process, args=(intsv_file, in_queue,
        num_worker, img_idx))
    reader.daemon = True
    reader.start()

    worker_pool = [];
    if debug:
        detprocess(caffenet, caffemodel,
            pixel_mean, scale, cmap, gpus[0], key_idx, 
            img_idx, in_queue, out_queue, **kwargs)
    else:
        for gpu in gpus:
            worker = mp.Process(target=detprocess, args=(caffenet, caffemodel,
                pixel_mean, scale, cmap, gpu, key_idx, 
                img_idx, in_queue, out_queue), kwargs=kwargs);
            worker.daemon = True
            worker_pool.append(worker)
            worker.start()

    assert not kwargs.get('debug_detect', False)

    outtsv_file_tmp = outtsv_file + '.tmp'
    def writer_process(out_queue, num_worker, outtsv_file_tmp):
        def yield_output():
            num_finished = 0
            last_print_time = 0
            count = 0
            while True:
                if num_finished == num_worker:
                    break
                else:
                    if out_queue.qsize() == 0:
                        if time.time() - last_print_time > 10:
                            logging.info('writer is waiting. {}'.format(count))
                            last_print_time = time.time()
                    if out_queue.empty():
                        if time.time() - last_print_time > 10:
                            logging.info('writer waiting to get the data. {}'.format(count))
                            last_print_time = time.time()
                    x = out_queue.get()
                    if x is None:
                        num_finished = num_finished + 1
                    else:
                        count = count + 1
                        yield x
        tsv_writer(yield_output(), outtsv_file_tmp)

    writer = mp.Process(target=writer_process, args=(out_queue, num_worker,
        outtsv_file_tmp))
    writer.daemon = True
    writer.start()
    
    reader.join()
    logging.info('reader finished')
    for proc in worker_pool: #wait all process finished.
        proc.join()
    logging.info('worker finished')
    writer.join()
    logging.info('writer finished')

    os.rename(outtsv_file_tmp, outtsv_file)
    #caffe.print_perf(count)

    return count

if __name__ =="__main__":
    args = parse_args(sys.argv[1:])
    outtsv_file = args.outtsv if args.outtsv!="" else os.path.splitext(args.intsv)[0]+".eval"

    pixel_mean = [float(x) for x in args.mean.split(',')]
    
    scale = 1
    tsvdet(args.net, args.model, args.intsv, args.colkey, args.colimg,
            pixel_mean, scale, outtsv_file, gpus=args.gpus)
    
