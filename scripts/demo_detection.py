import glob
import _init_paths
from tsv_io import tsv_reader
from qd_common import init_logging, read_to_buffer
import numpy as np
from process_image import draw_bb
import caffe
import time
import cv2
import logging
import os.path as op
from pprint import pformat
import matplotlib.pyplot as plt
from yolodet import im_detect, result2bblist2, detect_image
from qd_common import img_from_base64, load_list_file
from qd_common import calculate_iou
from process_image import save_image, load_image
from qd_common import network_input_to_image
from qd_common import load_net, write_to_file


def test_demo():
    source_image_tsv = None
    waitkey = 0 if source_image_tsv else 1

    # yolo9000 from yolo
    label_names = read_to_buffer('aux_data/yolo9k/9k.names').split('\n')
    test_proto_file = 'aux_data/yolo9k/yolo9000.nobn.prototxt'
    model_param = 'aux_data/yolo9k/yolo9000.nobn.caffemodel'
    pixel_mean = [0]
    thresh = 0.24

    label_names = [l.strip() for l in label_names]
    predict_online(test_proto_file, model_param, pixel_mean, label_names,
            source_image_tsv, thresh, waitkey)

def sort_dict(stat):
    result = []
    for key in stat:
        result.append((key, stat[key]))
    result.sort(key=lambda x: x[1], reverse=True)
    return result

class ImageSource(object):
    def __init__(self, source_image_tsv):
        self._cap = None
        self._rows = None
        self._im = None
        self._all_file = None
        self._all_file_idx = 0
        if source_image_tsv is None:
            self._cap = cv2.VideoCapture(0)
        elif type(source_image_tsv) is str:
            if source_image_tsv.endswith('tsv'):
                self._rows = tsv_reader(source_image_tsv)
            elif op.isfile(source_image_tsv):
                self._im = cv2.imread(source_image_tsv, cv2.IMREAD_COLOR)
            elif type(source_image_tsv) is np.ndarray:
                self._im = source_image_tsv
            elif op.isdir(source_image_tsv):
                self._all_file = []
                for f in glob.glob(op.join(source_image_tsv, '*')):
                    self._all_file.append(f)

    def next(self):
        if self._cap:
            r, im = self._cap.read()
            im = cv2.flip(im, 1)
            return im
        elif self._rows:
            x = next(self._rows)
            if x is not None:
                return img_from_base64(x[-1])
            else:
                self._rows = tsv_reader(source_image_tsv)
                return img_from_base64(next(self._rows)[-1])
        elif self._im is not None:
            return np.copy(self._im)
        elif self._all_file is not None:
            while True:
                f = self._all_file[self._all_file_idx]
                im = cv2.imread(f, cv2.IMREAD_COLOR)
                self._all_file_idx = self._all_file_idx + 1
                if self._all_file_idx > len(self._all_file):
                    self._all_file_idx = 0
                if im is not None:
                    return im

    def close(self):
        if self._cap:
            self._cap.release()

def to_grey(w):
    nw = 255. * (w - np.min(w[:])) / (np.max(w[:]) - np.min(w[:]))
    nw = nw.astype(np.uint8)
    nw = nw[:, :, np.newaxis]
    nw = np.repeat(nw, 3, axis=2)
    return nw

def predict_one(im, test_proto_file, model_param, pixel_mean, label_names,
        source_image_tsv, thresh, gpu=0):
    if gpu >= 0:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    
    net_test_proto_file = test_proto_file
    net = caffe.Net(str(net_test_proto_file), str(model_param), caffe.TEST)
    # check if it is tree structure
    net_proto = load_net(net_test_proto_file)
    yolo_tree = False
    for l in net_proto.layer:
        if l.type == 'SoftmaxTreePrediction':
            yolo_tree = True
    all_bb, all_label, all_conf = detect_image(net, im, pixel_mean, label_names, 
            thresh=thresh, yolo_tree=yolo_tree)
    return all_bb, all_label, all_conf

def predict_online(test_proto_file, model_param, pixel_mean, label_names,
        source_image_tsv, thresh, waitkey=1, gpu=0):
    debug = False
    source = ImageSource(source_image_tsv)
    if gpu >= 0:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()
    
    net_test_proto_file = test_proto_file
    if debug:
        # enable the force backward so that we can run backward() to find out the diff
        n = load_net(test_proto_file)
        n.force_backward = True
        net_test_proto_file = test_proto_file + '.force_backward'
        write_to_file(str(n), net_test_proto_file)

    net = caffe.Net(net_test_proto_file, model_param, caffe.TEST)
    
    # check if it is tree structure
    net_proto = load_net(net_test_proto_file)
    yolo_tree = False
    for l in net_proto.layer:
        if l.type == 'SoftmaxTreePrediction':
            yolo_tree = True

    last_print_time = time.time()
    fstat = {}
    f = 0
    color_mapping = {label_name: tuple(np.random.random(3) * 255) \
            for i, label_name in enumerate(label_names)}
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while True:
        im = source.next()
        assert im is not None
        start = time.time()
        stat = {}
        all_bb, all_label, all_conf = detect_image(net, im, pixel_mean, label_names, 
                stat=stat, thresh=thresh, yolo_tree=yolo_tree)
        if stat is not None:
            time_start = time.time()
        orig_im = np.copy(im)

        if debug:
            # visualize the data
            # we plot the visualization for each bounding boxes. the layout
            # will be 
            # original im with bb; b diff; g diff; r diff; empty
            # network im with bb; b diff * b; g diff * g; r diff * r; sum all
            probs = net.blobs['prob'].data.reshape((-1, len(label_names) + 1))
            idx = np.where(np.max(probs[:, :-1], axis=1) > thresh)[0]

            feat = net.blobs['last_conv'].data
            num_anchor = 5
            feat_height = feat.shape[2]
            feat_width = feat.shape[3]
            num_bb = len(all_bb)
            stride_w = 32
            stride_h = 32
            xy_blob = net.blobs['xy'].data
            wh_blob = net.blobs['wh'].data
            obj_blob = net.blobs['obj'].data
            anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
            bbox_blob = net.blobs['bbox'].data.reshape((num_anchor, feat_height,
                feat_width, 4))
            for i in range(num_bb):
                _, ax = plt.subplots(2, 8)
                # 0, 0 -- original image + bb
                curr_bb = all_bb[i]
                curr_label = all_label[i]
                curr_conf = all_conf[i]
                im = np.copy(orig_im)
                draw_bb(im, [curr_bb], [curr_label], [curr_conf])
                ax[0, 0].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

                # 1, 0 -- get bb in the network image coordinates
                # blob of bbox is still in the original image coord. 
                target_a, target_h, target_w = np.unravel_index(idx[i], [num_anchor, feat_height, feat_width])
                # find out the x, y, w, h in the net image coord
                netim_center_x = (xy_blob[0, target_a, target_h, target_w] + target_w) * stride_w
                netim_center_y = (xy_blob[0, target_a + num_anchor, target_h, target_w] + target_h) * stride_h
                netim_w = np.exp(wh_blob[0, target_a, target_h, target_w]) * anchors[2 * target_a] * stride_w
                netim_h = np.exp(wh_blob[0, target_a + num_anchor, target_h, target_w]) * anchors[2 * target_a + 1] * stride_h
                netim_bb = [netim_center_x - netim_w/2., netim_center_y - netim_h/2., 
                        netim_center_x + netim_w/2., netim_center_y + netim_h/2.]
                net_target_obj = obj_blob[0, target_a, target_h, target_w]
                netims = network_input_to_image(net.blobs['data'].data, pixel_mean)
                assert len(netims) == 1
                netim = netims[0]
                draw_bb(netim, [netim_bb], [curr_label], [net_target_obj])
                ax[1, 0].imshow(cv2.cvtColor(netim, cv2.COLOR_BGR2RGB))

                # backward for w
                diff = net.blobs['last_conv'].diff
                diff[...] = 0
                diff[0, 2 * num_anchor + target_a, target_h, target_w] = 1
                net.backward(start='last_conv', end='input')

                # for each channel
                all_wd = 0
                for c in range(3):
                    w = net.blobs['data'].diff[0, c, :, :]
                    nw = to_grey(w)
                    ax[0, 2 * c + 1].imshow(nw)

                    d = net.blobs['data'].data[0, c, :, :]
                    wd = w * d
                    all_wd = all_wd + wd
                    nwd = to_grey(wd)
                    ax[0, 2 * c + 2].imshow(nwd)
                ax[0, 7].imshow(to_grey(all_wd))

                diff = net.blobs['last_conv'].diff
                diff[...] = 0
                diff[0, 3 * num_anchor + target_a, target_h, target_w] = 1
                net.backward(start='last_conv', end='input')
                # for each channel
                all_wd = 0
                for c in range(3):
                    w = net.blobs['data'].diff[0, c, :, :]
                    nw = to_grey(w)
                    ax[1, 2 * c + 1].imshow(nw)
                    d = net.blobs['data'].data[0, c, :, :]
                    wd = w * d
                    all_wd = all_wd + wd
                    nwd = to_grey(wd)
                    ax[1, 2 * c + 2].imshow(nwd)
                ax[1, 7].imshow(to_grey(all_wd))


                # show image
                plt.show()
                plt.close()

        draw_bb(im, all_bb, all_label, None, color=color_mapping)
        if stat != None:
            time_curr = time.time()
            stat['draw_bb'] = time_curr - time_start
            time_start = time_curr

        curr = time.time()

        if curr - last_print_time > 2:
            fstat = {key: fstat[key] / f for key in fstat }
            logging.info('result: {}; fps: {:.1f}; stat: \n{}'.format(
                zip(all_label, all_conf, all_bb), 
                1.0 / (curr - start), pformat(sort_dict(fstat))))
            last_print_time = curr
            fstat = {}
            f = 0
        for key in stat:
            if key not in fstat:
                fstat[key] = stat[key]
            else:
                fstat[key] = fstat[key] + stat[key]
        f = f + 1
        cv2.imshow('frame', im)
        k = cv2.waitKey(waitkey) & 0xFF
        if k == ord('s'):
            save_image(orig_im, 'tmp.png')
        if k == ord('q'):
            break
    source.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    init_logging()
    test_demo()

