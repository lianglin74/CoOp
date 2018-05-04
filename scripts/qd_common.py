import _init_paths
import yaml
from collections import OrderedDict
import progressbar 
import json
import sys
import os
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Event
import logging
import numpy as np
import logging
import glob
import caffe
import time
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import os.path as op
import re
from google.protobuf import text_format
import base64
import cv2
from itertools import izip

def load_class_ap(full_expid, predict_file):
    report_file = '{}.report'.format(op.splitext(predict_file)[0])
    fname = op.join('output', full_expid, 'snapshot', report_file +
            '.class_ap.json')
    if op.isfile(fname):
        return json.loads(read_to_buffer(fname))
    else:
        return None

def calculate_ap_by_true_list(corrects, total):
    precision = (1. * np.cumsum(corrects)) / np.arange(1, 1 + len(corrects))
    if np.sum(corrects) == 0:
        return 0
    return np.sum(precision * corrects) / total 

def calculate_image_ap(predicts, gts):
    '''
    a list of rects
    '''
    matched = [False] * len(gts)
    corrects = np.zeros(len(predicts))
    for j, p in enumerate(predicts):
        for i, g in enumerate(gts):
            if not matched[i]:
                iou = calculate_iou(p, g)
                if iou > 0.3:
                    matched[i] = True
                    corrects[j] = 1
    return calculate_ap_by_true_list(corrects, len(gts))

def get_parameters_by_full_expid(full_expid):
    yaml_file = op.join('output', full_expid, 'parameters.yaml')
    if not op.isfile(yaml_file):
        return None
    param = load_from_yaml_file(yaml_file)
    if 'data' not in param:
        param['data'], param['net'] = parse_data_net(full_expid,
                param['expid'])
    return param

def get_all_model_expid():
    names = os.listdir('./output')
    return names

def get_target_images(predicts, gts, cat, threshold):
    image_aps = []
    for key in predicts:
        rects = predicts[key]
        curr_gt = [g for g in gts[key] if cat == 'any' or g['class'] == cat]
        curr_pred = [p for p in predicts[key] if cat == 'any' or (p['class'] == cat and
                p['conf'] > threshold)]
        if len(curr_gt) == 0 and len(curr_pred) == 0:
            continue
        curr_pred = sorted(curr_pred, key=lambda x: -x['conf'])
        ap = calculate_image_ap([p['rect'] for p in curr_pred],
                [g['rect'] for g in curr_gt])
        image_aps.append((key, ap))
    image_aps = sorted(image_aps, key=lambda x: x[1])
    #image_aps = sorted(image_aps, key=lambda x: -x[1])
    target_images = [key for key, ap in image_aps]
    return target_images, image_aps

def readable_confusion_entry(entry):
    '''
    entry: dictionary, key: label, value: count
    '''
    label_count = [(label, entry[label]) for label in entry]
    label_count.sort(key=lambda x: -x[1])
    total = sum([count for label, count in label_count])
    percent = [1. * count / total for label, count in label_count]
    cum_percent = np.cumsum(percent)
    items = []
    for i, (label, count) in enumerate(label_count):
        if i >= 5:
            continue
        items.append((label, '{}'.format(count), '{:.1f}'.format(100. *
            percent[i]),
            '{:.1f}'.format(100. * cum_percent[i])))
    return items

def get_all_tree_data():
    names = sorted(os.listdir('./data'))
    return [name for name in names 
        if op.isfile(op.join('data', name, 'root.yaml'))]

def parse_test_data(predict_file):
    parts = predict_file.split('.')
    idx_caffemodel = [i for i, p in enumerate(parts) if 'caffemodel' in p]
    if len(idx_caffemodel) == 1:
        idx_caffemodel = idx_caffemodel[0]
        test_data = parts[idx_caffemodel + 1]
        test_data_split = parts[idx_caffemodel + 2]
        if test_data_split in ['train', 'trainval', 'test']:
            return test_data, test_data_split
    all_data = os.listdir('data/')
    candidates = [data for data in all_data if '.caffemodel.' + data in predict_file]
    assert len(candidates) > 0
    max_length = max([len(c) for c in candidates])
    test_data = [c for c in candidates if len(c) == max_length][0]
    test_data_split = 'test' if 'testOnTrain' not in predict_file else 'train'
    return test_data, test_data_split

def parse_data(full_expid):
    all_data = os.listdir('data/')
    candidates = [data for data in all_data if full_expid.startswith(data)]
    max_length = max([len(c) for c in candidates])
    return [c for c in candidates if len(c) == max_length][0]

def parse_iteration(predict_file):
    begin_key = 'model_iter_'
    end_key = '.caffemodel'
    begin_idx = predict_file.find(begin_key)
    end_idx = predict_file.find(end_key)
    return float(predict_file[begin_idx + len(begin_key) : end_idx])

def get_all_predict_files(full_expid):
    model_folder = op.join('output', full_expid, 'snapshot')
    found = glob.glob(op.join(model_folder, '*.predict'))
    predict_files = [op.basename(f) for f in found]
    iterations = [(parse_iteration(p), p) for p in predict_files]
    iterations.sort(key=lambda x: -x[0])
    return [p for i, p in iterations]

def dict_to_list(d, idx):
    result = []
    for k in d:
        vs = d[k]
        for v in vs:
            try:
                r = []
                # if v is a list or tuple
                r.extend(v[:idx])
                r.append(k)
                r.extend(v[idx: ])
            except TypeError:
                r = []
                if idx == 0:
                    r.append(k)
                    r.append(v)
                else:
                    assert idx == 1
                    r.append(v)
                    r.append(k)
            result.append(r)
    return result

def list_to_dict(l, idx):
    result = {}
    for x in l:
        if x[idx] not in result:
            result[x[idx]] = []
        y = x[:idx] + x[idx + 1:]
        if len(y) == 1:
            y = y[0]
        result[x[idx]].append(y)
    return result

def generate_lineidx(filein, idxout):
    assert not os.path.isfile(idxout)
    with open(filein,'r') as tsvin, open(idxout,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0;
        while fpos!=fsize:
    	    tsvout.write(str(fpos)+"\n");
            tsvin.readline()
            fpos = tsvin.tell();

def fix_net_bn_layers(net, num_bn_fix):
    for l in net.layer:
        if l.type == 'BatchNorm':
            if num_bn_fix > 0:
                l.batch_norm_param.use_global_stats = True
                num_bn_fix = num_bn_fix - 1
            else:
                break

def yolo_old_to_new(old_proto, old_model, new_model):
    '''
    input: old_proto, old_model
    old_proto: train or test proto
    output: new_model
    '''
    assert op.isfile(old_proto)
    assert op.isfile(old_model)

    ensure_directory(op.dirname(new_model))

    # infer the number of anchors and the number of classes
    proto = load_net(old_proto)
    last_layer = proto.layer[-1]
    if last_layer.type == 'RegionLoss':
        num_classes = last_layer.region_loss_param.classes
        biases_length = len(last_layer.region_loss_param.biases)
    else:
        assert last_layer.type == 'RegionOutput'
        num_classes = last_layer.region_output_param.classes
        biases_length = len(last_layer.region_output_param.biases)
    num_anchor = biases_length / 2
    assert num_anchor * 2 == biases_length

    target_layer = proto.layer[-2]
    assert target_layer.type == 'Convolution'
    assert target_layer.convolution_param.num_output == ((5 + num_classes) *
        num_anchor)
    net = caffe.Net(old_proto, old_model, caffe.TRAIN)
    target_param = net.params[target_layer.name]
    old_weight = target_param[0].data
    new_weight = np.zeros_like(old_weight)
    has_bias = len(target_param) > 0
    all_old = [old_weight]
    all_new = [new_weight]
    if has_bias:
        old_bias = target_param[1].data
        new_bias = np.zeros_like(old_bias)
        all_old.append(old_bias[:, np.newaxis])
        all_new.append(new_bias[:, np.newaxis])

    for i in xrange(num_anchor):
        for old_p, new_p in izip(all_old, all_new):
            x = old_p[0 + i * (5 + num_classes), :]
            y = old_p[1 + i * (5 + num_classes), :]
            w = old_p[2 + i * (5 + num_classes), :]
            h = old_p[3 + i * (5 + num_classes), :]
            o = old_p[4 + i * (5 + num_classes), :]
            old_cls_start = 5 + i * (5 + num_classes)
            cls = old_p[old_cls_start: (old_cls_start + num_classes),
                    :]
            new_p[i + 0 * num_anchor, :] = x
            new_p[i + 1 * num_anchor, :] = y
            new_p[i + 2 * num_anchor, :] = w
            new_p[i + 3 * num_anchor, :] = h
            new_p[i + 4 * num_anchor, :] = o
            new_cls_start = i * num_classes + 5 * num_anchor
            new_p[new_cls_start : (new_cls_start + num_classes), :] = cls
    
    for old_p, new_p in izip(all_old, all_new):
        old_p[...] = new_p

    net.save(new_model)

def is_cluster(ssh_info):
    return '-p' in ssh_info and '-i' not in ssh_info

def visualize_train(solver):
    plt.figure()
    features = []
    for i in range(100):
        solver.step(10)
        #logging.info(np.sum(solver.net.blobs['label'].data))
        delta = 0.000001
        data_values = []
        for key in solver.net.blobs:
            data_value = np.mean(np.abs(solver.net.blobs[key].data))
            data_values.append(data_value + delta)
        diff_values = []
        for key in solver.net.blobs:
            diff_values.append(np.mean(np.abs(solver.net.blobs[key].diff))
                + delta)
        param_keys = []
        param_data = []
        for key in solver.net.params:
            for i, b in enumerate(solver.net.params[key]):
                param_keys.append('{}_{}'.format(key, i))
                param_data.append(np.mean(np.abs(b.data)) + delta)
        param_diff = []
        for key in solver.net.params:
            for i, b in enumerate(solver.net.params[key]):
                param_diff.append(np.mean(np.abs(b.diff)) + delta)
    
        xs = range(len(solver.net.blobs))
        plt.gcf().clear()
        plt.subplot(2, 1, 1)
    
        plt.semilogy(xs, data_values, 'r-o')
        plt.semilogy(xs, diff_values, 'b-*')
        plt.xticks(xs, solver.net.blobs.keys(), rotation='vertical')
        plt.grid()
    
        plt.subplot(2, 1, 2)
        xs = range(len(param_keys))
        plt.semilogy(xs, param_data, 'r-o')
        plt.semilogy(xs, param_diff, 'b-*')
        plt.xticks(xs, param_keys, rotation='vertical')
        plt.grid()
        plt.draw()
        plt.pause(0.001)

def network_input_to_image(data, mean_value):
    all_im = []
    for d in data:
        im = (d.transpose((1, 2, 0)) + np.asarray(mean_value).reshape(1, 1,
            3)).astype(np.uint8).copy()
        all_im.append(im)
    return all_im

def remove_data_augmentation(data_layer):
    assert data_layer.type == 'TsvBoxData'
    data_layer.box_data_param.jitter = 0
    data_layer.box_data_param.hue = 0
    data_layer.box_data_param.exposure = 1
    data_layer.box_data_param.random_scale_min = 1
    data_layer.box_data_param.random_scale_max = 1
    data_layer.box_data_param.fix_offset = True
    data_layer.box_data_param.saturation = True

def calculate_macc(prototxt):
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffe.TEST)
    net_proto = load_net(prototxt)
    macc = []
    ignore_layers = ['BatchNorm', 'Scale', 'ReLU', 'Softmax', 
            'Pooling', 'Eltwise', 'Shift', 'Concat']
    for layer in net_proto.layer:
        if layer.type == 'Convolution':
            assert len(layer.bottom) == 1
            input_shape = net.blobs[layer.bottom[0]].data.shape
            assert len(layer.top) == 1
            output_shape = net.blobs[layer.top[0]].data.shape
            assert len(input_shape) == 4
            assert len(output_shape) == 4
            assert input_shape[0] == 1
            assert output_shape[0] == 1
            m = output_shape[3] * output_shape[1] * output_shape[2]
            assert layer.convolution_param.kernel_h == 0
            assert layer.convolution_param.kernel_w == 0
            assert len(layer.convolution_param.kernel_size) == 1
            m = m * layer.convolution_param.kernel_size[0] * \
                    layer.convolution_param.kernel_size[0]
            m = m * input_shape[1]
            m = m / layer.convolution_param.group
            macc.append((layer.name, m/1000000.))

        elif layer.type == 'InnerProduct':
            assert len(layer.bottom) == 1
            assert len(layer.top) == 1
            input_shape = net.blobs[layer.bottom[0]].data.shape
            output_shape = net.blobs[layer.top[0]].data.shape
            assert input_shape[0] == 1
            assert output_shape[0] == 1
            m = reduce(lambda x,y:x*y, input_shape)
            m = m * reduce(lambda x,y:x*y, output_shape)
            macc.append((layer.name, m/1000000.))
        #elif layer.type == 'Scale':
            #assert len(layer.bottom) == 1
            #input_shape = net.blobs[layer.bottom[0]].data.shape
            #m = reduce(lambda x,y:x*y, input_shape)
            #macc = macc + m
        else:
            #assert layer.type in ignore_layers, layer.type
            pass

    return macc

def check_best_iou(biases, gt_w, gt_h, n):
    def iou(gt_w, gt_h, w, h):
        inter = min(gt_w, w) * min(gt_h, h)
        return inter / (gt_w * gt_h + w * h - inter)

    best_iou = -1
    best_n = -1
    for i in range(len(biases) / 2):
        u = iou(gt_w, gt_h, biases[2 * i], biases[2 * i + 1])
        if u > best_iou:
            best_iou = u
            best_n = i
    assert best_n == n

def calculate_iou(rect0, rect1):
    '''
    x0, y1, x2, y3
    '''
    w = min(rect0[2], rect1[2]) - max(rect0[0], rect1[0])
    if w < 0:
        return 0
    h = min(rect0[3], rect1[3]) - max(rect0[1], rect1[1])
    if h < 0:
        return 0
    i = w * h
    a1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    a0 = (rect0[2] - rect0[0]) * (rect0[3] - rect0[1])
    return 1. * i / (a0 + a1 - i) 

def process_run(func, *args):
    def internal_func(queue):
        result = func(*args)
        queue.put(result)
    queue = mp.Queue()
    p = Process(target=internal_func, args=(queue,))
    p.start()
    p.join()
    assert p.exitcode == 0
    return queue.get()

def setup_yaml():
    """ https://stackoverflow.com/a/8661021 """
    represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)    

def init_logging():
    np.seterr(divide = "raise", over="warn", under="warn",  invalid="raise")
    logging.basicConfig(level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s',
    datefmt='%m-%d %H:%M:%S',
    )
    setup_yaml()

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        if not os.path.exists(path) and not op.islink(path):
            os.makedirs(path)

def parse_pattern(pattern, s):
    result = re.search(pattern, s)
    if result is None:
        return result
    return [float(g) for g in result.groups()]

def parse_yolo_log(log_file):
    pattern = 'loss_xy: ([0-9, .]*); loss_wh: ([0-9, .]*); '
    pattern = pattern + 'loss_objness: ([0-9, .]*); loss_class: ([0-9, .]*)' 

    base_log_lines = read_lines(log_file)
    xys = []
    whs = []
    loss_objnesses = []
    loss_classes = []
    for line in base_log_lines:
        gs = parse_pattern(pattern, line)
        if gs is None:
            continue
        idx = 0
        xys.append(float(gs[idx]))
        idx = idx + 1
        whs.append(float(gs[idx]))
        idx = idx + 1
        loss_objnesses.append(float(gs[idx]))
        idx = idx + 1
        loss_classes.append(float(gs[idx]))

    return xys, whs, loss_objnesses, loss_classes

def parse_nums(p, log_file):
    result = []
    for line in read_lines(log_file):
        gs = parse_pattern(p, line)
        if gs is None:
            continue
        result.append(gs)
    return result

def parse_yolo_log_st(log_file):
    p = 'region_loss_layer\.cpp:1138] ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*)'
    ss = parse_nums(p, log_file)
    p = 'region_loss_layer\.cpp:1140] ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*), ([0-9]*)'
    tt = parse_nums(p, log_file)
    return ss, tt

def parse_yolo_log_acc(log_file):
    p = 'Region Avg IOU: ([0-9, .]*), Class: ([0-9, .]*), '
    p = p + 'Obj: ([0-9, .]*), No Obj: ([0-9, .]*), Avg Recall: ([0-9, .]*),  count: ([0-9]*)'
    all_ious = []
    all_probs = []
    all_obj = []
    all_noobj = []
    all_recall = []
    all_count = []
    for line in read_lines(log_file):
        gs = parse_pattern(p, line)
        if gs is None:
            continue
        all_ious.append(gs[0])
        all_probs.append(gs[1])
        all_obj.append(gs[2])
        all_noobj.append(gs[3])
        all_recall.append(gs[4])
        all_count.append(gs[5])
    return all_ious, all_probs, all_obj, all_noobj, all_recall, all_count


def read_lines(file_name):
    with open(file_name, 'r') as fp:
        for line in fp:
            yield line

def read_to_buffer(file_name):
    with open(file_name, 'r') as fp:
        all_line = fp.read()
    return all_line

def load_solver(file_name):
    with open(file_name, 'r') as fp:
        all_line = fp.read()
    solver_param = caffe.proto.caffe_pb2.SolverParameter()
    text_format.Merge(all_line, solver_param)
    return solver_param

class Model(object):
    def __init__(self, test_proto_file, train_proto_file, model_param, mean_value, scale, model_iter):
        self.test_proto_file = test_proto_file
        self.model_param = model_param
        self.mean_value = mean_value
        self.model_iter = model_iter
        self.scale = scale
        self.train_proto_file = train_proto_file

def construct_model(solver, test_proto_file, is_last=True, iteration=None):
    solver_param = load_solver(solver)
    train_net_param = load_net(solver_param.train_net)
    data_layer = train_net_param.layer[0]
    # if we don't convert it to list, the type is repeated field, which is not
    # pickable, and thus cannot be paralled by mp.Pool()
    mean_value = list(train_net_param.layer[0].transform_param.mean_value)
    scale = train_net_param.layer[0].transform_param.scale

    if is_last:
        last_model = '{0}_iter_{1}.caffemodel'.format(
                solver_param.snapshot_prefix, solver_param.max_iter)
        return Model(test_proto_file, solver_param.train_net, 
                last_model, mean_value, scale,
                solver_param.max_iter)
    elif iteration:
        last_model = '{0}_iter_{1}.caffemodel'.format(
                solver_param.snapshot_prefix, iteration)
        return Model(test_proto_file, solver_param.train_net,
                last_model, mean_value, scale,
                solver_param.max_iter)
    else:
        total = (solver_param.max_iter + solver_param.snapshot - 1) / solver_param.snapshot
        all_model = []
        for i in xrange(total + 1, 0, -1):
            if i == 0:
                continue
            j = i * solver_param.snapshot
            j = min(solver_param.max_iter, j)
            if j == solver_param.max_iter:
                continue
            last_model = '{0}_iter_{1}.caffemodel'.format(
                    solver_param.snapshot_prefix, j)
            all_model.append(Model(test_proto_file, solver_param.train_net,
                last_model, mean_value,
                scale, j))
        return all_model

def load_net_from_str(all_line):
    net_param = caffe.proto.caffe_pb2.NetParameter()
    text_format.Merge(all_line, net_param)
    return net_param

def load_net(file_name):
    with open(file_name, 'r') as fp:
        all_line = fp.read()
    return load_net_from_str(all_line)

def remove_nms(n):
    for l in n.layer:
        if l.type == 'RegionOutput':
            l.region_output_param.nms = -1
        if l.type == 'RegionPrediction':
            l.region_prediction_param.nms = -1

def load_binary_net(file_name):
    with open(file_name, 'r') as fp:
        c = fp.read()
    param = caffe.proto.caffe_pb2.NetParameter()
    param.ParseFromString(c)
    return param

def exist_bn(net, conv_top):
    for l in net.layer:
        if l.type == 'BatchNorm':
            assert len(l.bottom) == 1
            if l.bottom[0] == conv_top:
                return True
    return False

def update_bn(net):
    layers = []
    for l in net.layer:
        layers.append(l)
        if l.type == 'Convolution':
            assert len(l.top) == 1
            conv_top = l.top[0]
            if not exist_bn(net, conv_top):
                bn = caffe.proto.caffe_pb2.LayerParameter()
                bn.name = 'bn_{}'.format(conv_top)
                bn.bottom.append(conv_top)
                bn.type = 'BatchNorm'
                bn.top.append(conv_top)
                for i in range(3):
                    p = bn.param.add()
                    p.lr_mult = 0
                    p.decay_mult = 0
                layers.append(bn)
                scale = caffe.proto.caffe_pb2.LayerParameter()
                scale.name = 'scale_{}'.format(conv_top)
                scale.bottom.append(conv_top)
                scale.top.append(conv_top)
                scale.type = 'Scale'
                scale.scale_param.bias_term = True
                for i in range(2):
                    p = scale.param.add()
                    p.lr_mult = 1
                    p.decay_mult = 1 - i
                layers.append(scale)

    net.ClearField('layer')
    net.layer.extend(layers)

def update_conv_channels(net, factor, skip):
    c = 0
    s = 0
    for l in net.layer:
        if l.type == 'Convolution':
            if s < skip:
                s = s + 1
                continue
            o = l.convolution_param.num_output
            l.convolution_param.num_output = int(o * factor)
            c = c + 1
    logging.info('updated {} layers for channel factor'.format(c))

def update_crop_type(net, crop_type, inception_crop_kl=None):
    c = 0
    for l in net.layer:
        if l.type == 'TsvData':
            l.tsv_data_param.crop_type = crop_type
            if crop_type == caffe.params.TsvData.InceptionStyle:
                l.tsv_data_param.color_kl_file = inception_crop_kl
            c = c + 1
    logging.info('updated {} layers for crop type'.format(c))

def update_kernel_active2(net, **kwargs):
    kernel_active = kwargs['kernel_active']
    kernel_active_skip = kwargs.get('kernel_active_skip', 0)
    kernel_active_type = kwargs.get('kernel_active_type', 'SEQ')
    shrink_group_if_group_e_out = kwargs.get('shrink_group_if_group_e_out', 
            False)
    logging.info('type: {}'.format(kernel_active_type))
    c = 0
    skipped = 0
    logging.info('{}-{}'.format(kernel_active, kernel_active_skip));
    layers = [] 
    bottom_map = {}
    for l in net.layer:
        if l.type == 'Convolution':
            assert l.convolution_param.kernel_h == 0
            assert l.convolution_param.kernel_w == 0
            assert len(l.convolution_param.kernel_size) == 1
            if l.convolution_param.kernel_size[0] > 1:
                if skipped < kernel_active_skip:
                    skipped = skipped + 1
                    logging.info('skiping to update active kernel')
                else:
                    assert len(l.bottom) == 1
                    ks = l.convolution_param.kernel_size[0]
                    bottom = l.bottom[0]
                    if bottom not in bottom_map:
                        shift_layer = caffe.proto.caffe_pb2.LayerParameter()
                        shift_name = 'shift_{}'.format(bottom)
                        shift_layer.name = shift_name
                        shift_layer.bottom.append(bottom)
                        shift_layer.type = 'Shift'
                        shift_layer.top.append(shift_name)
                        sp = shift_layer.shift_param
                        sp.sparsity = kernel_active
                        sp.kernel_s = ks
                        if kernel_active_type != 'SEQ':
                            if kernel_active_type == 'SEQ_1x1':
                                sp.type = caffe.params.Shift.SEQ_1x1
                            elif kernel_active_type == 'UNIFORM_1x1':
                                sp.type = caffe.params.Shift.UNIFORM_1x1
                            else:
                                assert False
                        layers.append(shift_layer)
                        bottom_map[bottom] = shift_name
                    else:
                        shift_name = bottom_map[bottom]
                    l.ClearField('bottom')
                    l.bottom.append(shift_name)
                    assert len(l.convolution_param.pad) == 1
                    assert  l.convolution_param.pad[0] == \
                            l.convolution_param.kernel_size[0] / 2
                    l.convolution_param.ClearField('pad')
                    l.convolution_param.kernel_size[0] = 1
                    num_output = l.convolution_param.num_output
                    if l.convolution_param.group == num_output:
                        if shrink_group_if_group_e_out:
                            assert num_output / ks / ks > 0
                            l.convolution_param.group = num_output / (ks * ks)
                    else:
                        assert l.convolution_param.group == 1
                    c = c + 1
        layers.append(l)
    logging.info('update {} layers'.format(c))
    net.ClearField('layer')
    net.layer.extend(layers)

def get_channel(net, blob_name):
    for l in net.layer:
        if l.type == 'Convolution':
            assert len(l.top) == 1
            if l.top[0] == blob_name:
                return l.convolution_param.num_output
    assert False, 'not found'

def fix_net_parameters(net, **kwargs):
    last_fixed_param = kwargs['last_fixed_param']
    found = False
    for l in net.layer:
        if l.type == 'Convolution':
            for p in l.param:
                p.lr_mult = 0
                p.decay_mult = 0
        elif l.type == 'BatchNorm':
            l.batch_norm_param.use_global_stats = True
        if l.name == last_fixed_param:
            for b in l.bottom:
                l.propagate_down.append(False)
            found = True
            break
    assert found

def set_no_bias(net, layer_name):
    for l in net.layer:
        if l.name == layer_name:
            assert l.type == 'Convolution'
            l.convolution_param.bias_term = False
            if len(l.param) == 2:
                del l.param[1]
            else:
                assert len(l.param) == 0
            return
    assert False

def add_yolo_angular_loss_regularizer(net, **kwargs):
    for l in net.layer:
        if l.name == 'angular_loss':
            logging.info('angular loss exists')
            return
    conf_layer = None
    for l in net.layer:
        if l.name == 'conf':
            conf_layer = l
            assert 'conf' in l.top
    found_t_label = False
    for l in net.layer:
        if 't_label' in l.top:
            found_t_label = True
            break
    assert conf_layer and found_t_label

    conf_layer.param[0].name = 'conf_w'
    CA = conf_layer.convolution_param.num_output
    assert len(conf_layer.bottom) == 1
    num_feature = get_channel(net, conf_layer.bottom[0])

    param_layer = net.layer.add()
    param_layer.name = 'param_conf_w'
    param_layer.type = 'Parameter'
    param_layer.parameter_param.shape.dim.append(CA)
    param_layer.parameter_param.shape.dim.append(num_feature)
    param_layer.parameter_param.shape.dim.append(1)
    param_layer.parameter_param.shape.dim.append(1)
    param_layer.top.append('conf_w')
    p = param_layer.param.add()
    p.name = 'conf_w'

    layer = net.layer.add()
    layer.name = 'angular_loss'
    layer.type = 'Python'
    layer.bottom.append(conf_layer.bottom[0])
    layer.bottom.append('t_label')
    layer.bottom.append('conf_w')
    layer.python_param.module = 'kcenter_exp'
    layer.python_param.layer = 'YoloAngularLossLayer'
    layer.propagate_down.append(True)
    layer.propagate_down.append(False)
    layer.propagate_down.append(False)
    layer.top.append('angular_loss')
    weight = kwargs.get('yolo_angular_loss_weight', 1)
    layer.loss_weight.append(weight)

def add_yolo_low_shot_regularizer(net, low_shot_label_idx):
    assert net.layer[-1].type == 'RegionLoss'
    assert net.layer[-2].type == 'Convolution'
    assert net.layer[-1].bottom[0] == net.layer[-2].top[0]
    assert net.layer[-2].convolution_param.kernel_size[0] == 1
    assert net.layer[-2].convolution_param.kernel_h == 0
    assert net.layer[-2].convolution_param.kernel_w == 0

    num_classes = net.layer[-1].region_loss_param.classes
    num_anchor = len(net.layer[-1].region_loss_param.biases) / 2

    param_dim1 = net.layer[-2].convolution_param.num_output
    param_dim2 = get_channel(net, net.layer[-2].bottom[0])

    # add the parameter name into the convolutional layer
    last_conv_param_name = 'last_conv_param_low_shot'
    net.layer[-2].param[0].name = last_conv_param_name

    # add the parameter layer to expose the parameter
    param_layer = net.layer.add()
    param_layer.type = 'Parameter'
    param_layer.name = 'param_last_conv'
    param_layer.top.append(last_conv_param_name)
    p = param_layer.param.add()
    p.name = last_conv_param_name
    p.lr_mult = 1
    p.decay_mult = 1
    param_layer.parameter_param.shape.dim.append(param_dim1)
    param_layer.parameter_param.shape.dim.append(param_dim2)
    
    # add the regularizer layer
    reg_layer = net.layer.add()
    reg_layer.type = 'Python'
    reg_layer.name = 'equal_norm'
    reg_layer.bottom.append(last_conv_param_name)
    reg_layer.top.append('equal_norm')
    reg_layer.loss_weight.append(1)
    reg_layer.python_param.module = 'equal_norm_loss'
    reg_layer.python_param.layer = 'YoloAlignNormToBaseLossLayer'
    reg_param = {'num_classes': num_classes, 
            'low_shot_label_idx': low_shot_label_idx,
            'num_anchor': num_anchor}
    reg_layer.python_param.param_str = json.dumps(reg_param)

def update_kernel_active(net, kernel_active, kernel_active_skip):
    assert False, 'use update_kernel_active2'
    c = 0
    skipped = 0
    logging.info('{}-{}'.format(kernel_active, kernel_active_skip));
    for l in net.layer:
        if l.type == 'Convolution':
            if skipped < kernel_active_skip:
                skipped = skipped + 1
                logging.info('skiping to update active kernel')
                continue
            l.convolution_param.kernel_active = kernel_active
            c = c + 1

    logging.info('update {} layers'.format(c))


def plot_to_file(xs, ys, file_name, **kwargs):
    fig = plt.figure()
    if all(isinstance(x, str) or isinstance(x, unicode) for x in xs):
        xs2 = range(len(xs))
        #plt.xticks(xs2, xs, rotation=15, ha='right')
        plt.xticks(xs2, xs, rotation='vertical')
        xs = xs2
    if type(ys) is dict:
        for key in ys:
            plt.plot(xs, ys[key], '-o')
    else:
        plt.plot(xs, ys, '-o')
    plt.grid()
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    ensure_directory(op.dirname(file_name))
    plt.tight_layout()
    # explicitly remove the file because philly does not support overwrite
    if op.isfile(file_name):
        os.remove(file_name)
    fig.savefig(file_name)
    plt.close(fig)

def parse_training_time(log_file):
    log = read_to_buffer(log_file)
    all_time_cost = []
    all_iters = []
    for line in log.split('\n'):
        m = re.match('.*Iteration.*iter\/s, ([0-9\.]*)s\/([0-9]*) iters.*', line)
        if m:
            r = m.groups() 
            time_cost = float(r[0])
            iters = float(r[1])
            all_iters.append(iters)
            all_time_cost.append(time_cost)
    return all_iters, all_time_cost

def encode_expid(prefix, *args):
    parts = [prefix]
    for (t, a) in args:
        p = ''
        if a != None:
            if type(a) == str:
                a = a.replace(':', '_')
            if t != None and len(t) > 0:
                p = p + '_{}'.format(t)
            p = p + '_{}'.format(a)
        parts.append(p)
    return ''.join(parts)

def caffemodel_num_param(model_file):
    param = caffe.proto.caffe_pb2.NetParameter()
    with open(model_file, 'r') as fp:
        model_context = fp.read()
    param.ParseFromString(model_context)
    result = 0
    for l in param.layer:
        for b in l.blobs:
            if len(b.double_data) > 0:
                result += len(b.double_data)
            elif len(b.data) > 0:
                result += len(b.data)
    return result

def write_to_yaml_file(context, file_name):
    ensure_directory(op.dirname(file_name))
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, default_flow_style=False)

def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)

def write_to_file(contxt, file_name):
    p = os.path.dirname(file_name)
    ensure_directory(p)
    with open(file_name, 'w') as fp:
        fp.write(contxt)

def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result

def correct_caffemodel_file_path(solverstate_file, out_file):
    solverstate = caffe.proto.caffe_pb2.SolverState()
    with open(solverstate_file, 'r') as fp:
        contents = fp.read()
    solverstate.ParseFromString(contents)
    changed = False
    if not os.path.exists(solverstate.learned_net):
        basename = os.path.basename(contents.learned_net)
        directory = os.path.dirname(solverstate)
        caffemodel = os.path.join(directory, basename)
        if os.path.exists(caffemodel):
            solverstate.learned_net = caffemodel
            changed = True
        else:
            assert False
    if changed or True:
        with open(out_file, 'w') as fp:
            fp.write(solverstate.SerializeToString())

class LoopProcess(Process):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        '''
        same signiture with Process.__init__
        The process will keep running the function of target and will wait for
        several seconds in between. This is useful to run some monitoring job
        or regular job
        '''
        super(LoopProcess, self).__init__(group, target, name, args, kwargs)
        self._exit = Event()

    def run(self):
        sleep_time = 5
        while not self._exit.is_set():
            if self._target:
                self._target(*self._args, **self._kwargs)
            time.sleep(sleep_time)

    def init_shutdown(self):
        self._exit.set()

class PyTee(object):
    def __init__(self, logstream, stream_name):
        valid_streams = ['stderr','stdout'];
        if  stream_name not in valid_streams:
            raise IOError("valid stream names are %s" % ', '.join(valid_streams))
        self.logstream =  logstream
        self.stream_name = stream_name;
    def __del__(self):
        pass;
    def write(self, data):  #tee stdout
        self.logstream.write(data);
        self.fstream.write(data);
        self.logstream.flush();
        self.fstream.flush();

    def flush(self):
        self.logstream.flush();
        self.fstream.flush();

    def __enter__(self):
        if self.stream_name=='stdout' :
            self.fstream   =  sys.stdout
            sys.stdout = self;
        else:
            self.fstream   =  sys.stderr
            sys.stderr = self;
        self.fstream.flush();
    def __exit__(self, _type, _value, _traceback):
        if self.stream_name=='stdout' :
            sys.stdout = self.fstream;
        else:
            sys.stderr = self.fstream;

def parse_basemodel_with_depth(net):
    '''
    darknet19->darknet19
    darknet19_abc->darknet19
    '''
    if '_' not in net:
        return net
    else:
        i = net.index('_')
        return net[: i]

def parallel_train(
        solver,  # solver proto definition
        snapshot,  # solver snapshot to restore
        weights,
        gpus,  # list of device ids
        timing=False,  # show timing info for compute and communications
        extract_blob=None
):
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()

    caffe.log('Using devices %s' % str(gpus))

    procs = []
    blob_queue = mp.Queue()
    logging.info('train on {}'.format(map(str, gpus)))
    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver, snapshot, weights, gpus, timing, uid, rank,
                        extract_blob, blob_queue))
        p.daemon = True
        p.start()
        procs.append(p)

    r = None
    if extract_blob:
        for rank in range(len(gpus)):
            if r is None:
                r = blob_queue.get()
            else:
                r = r + blob_queue.get()

    for p in procs:
        p.join()

    return r

def caffe_param_check(caffenet, caffemodel):
    if not os.path.exists(caffenet) or not os.path.exists(caffemodel):
        return {}

    caffe.set_mode_cpu()
    net = caffe.Net(str(caffenet), str(caffemodel), caffe.TEST)

    return caffe_net_check(net)

def get_solver(solver_prototxt, restore_snapshot=None):
    solver = caffe.SGDSolver(solver_prototxt)
    if restore_snapshot:
        solver.restore(restore_snapshot)
    return solver

def caffe_net_check(net):
    result = {}
    result['param'] = {}
    for key in net.params:
        value = net.params[key]
        result['param'][key] = []
        for i, v in enumerate(value):
            result['param'][key].append((float(np.mean(v.data)),
                float(np.std(v.data))))

    result['blob'] = {}
    for key in net.blobs:
        v = net.blobs[key]
        result['blob'][key] = (float(np.mean(v.data)), float(np.std(v.data)))

    return result

def caffe_train(solver_prototxt, gpu=-1, pretrained_model=None, restore_snapshot=None):
    if gpu >= 0:
        caffe.set_device(gpu)
        caffe.set_mode_gpu()
    #caffe.set_random_seed(777)
    solver = caffe.SGDSolver(solver_prototxt)
    if pretrained_model:
        solver.net.copy_from(pretrained_model, ignore_shape_mismatch=True)
    if restore_snapshot:
        solver.restore(restore_snapshot)
    solver.solve()

def register_timing(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            caffe.log(s)

    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))

def solve(proto, snapshot, weights, gpus, timing, uid, rank, extract_blob,
        blob_queue):
    caffe.init_glog(str(os.path.join(os.path.dirname(proto),
        'log_rank_{}_'.format(str(rank)))))
    caffe.set_device(gpus[rank])
    caffe.set_mode_gpu()
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(str(proto))
    logging.info('solve: {}'.format(str(proto)))
    assert solver is not None
    has_snapshot = snapshot and len(snapshot) != 0
    has_weight = weights and len(weights) != 0
    if has_snapshot:
        solver.restore(str(snapshot))
    elif has_weight:
        solver.net.copy_from(str(weights), ignore_shape_mismatch=True)

    if extract_blob is not None:
        logging.info('extract_blob = {}'.format(extract_blob))
        assert extract_blob in solver.net.blobs

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    if timing and rank == 0:
        register_timing(solver, nccl)
    else:
        solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)

    iters = solver.param.max_iter - solver.iter
    solver.step(iters)
    if rank == 0:
        solver.snapshot()

    if extract_blob is not None:
        blob_queue.put(solver.net.blobs[extract_blob].data)

def worth_create(base_file_name, derived_file_name):
    if os.path.isfile(derived_file_name) and \
            os.path.getmtime(derived_file_name) > os.path.getmtime(base_file_name):
        return False
    else:
        return True

def basename_no_ext(file_name):
    return op.splitext(op.basename(file_name))[0]

def default_data_path(dataset):
    '''
    use TSVDataset instead
    '''
    proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)));
    result = {}
    data_root = os.path.join(proj_root, 'data', dataset)
    result['data_root'] = data_root 
    result['source'] = os.path.join(data_root, 'train.tsv')
    result['trainval'] = op.join(data_root, 'trainval.tsv')
    result['test_source'] = os.path.join(data_root, 'test.tsv')
    result['labelmap'] = os.path.join(data_root, 'labelmap.txt')
    result['source_idx'] = os.path.join(data_root, 'train.lineidx')
    result['test_source_idx'] = os.path.join(data_root, 'test.lineidx')
    return result

class FileProgressingbar:
    fileobj = None
    pbar = None
    def __init__(self, fileobj, keyword='Test'):
        fileobj.seek(0,os.SEEK_END)
        flen = fileobj.tell()
        fileobj.seek(0,os.SEEK_SET)
        self.fileobj = fileobj
        widgets = ['{}: '.format(keyword), progressbar.AnimatedMarker(),' ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
        self.pbar = progressbar.ProgressBar(widgets=widgets, maxval=flen).start()
    def update(self):
        self.pbar.update(self.fileobj.tell())

def encoded_from_img(im):
    x = cv2.imencode('.jpg', im)[1]
    return base64.b64encode(x)

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.frombuffer(jpgbytestring, np.uint8)
    try:
        r = cv2.imdecode(nparr, cv2.IMREAD_COLOR);
        return r
    except:
        return None;

def test_correct_caffe_file_path():
    solverstate = \
    './output/cifar100_vggstyle_0.1A_bn_16_1_1_1_1/snapshot/model_iter_1000.solverstate'
    out_file = solverstate + '_2'
    correct_caffemodel_file_path(solverstate, out_file)
    s1 = caffe.proto.caffe_pb2.SolverState()
    s2 = caffe.proto.caffe_pb2.SolverState()
    with open(solverstate, 'r') as fp:
        s1.ParseFromString(fp.read())
    with open(out_file, 'r') as fp:
        s2.ParseFromString(fp.read())
    assert s1.learned_net == s2.learned_net
    assert s1.iter == s2.iter
    assert s1.current_step == s2.current_step
    assert len(s1.history) == len(s2.history)
    for s1_h, s2_h in zip(s1.history, s2.history):
        assert len(s1_h.data) == len(s2_h.data)
        for s1_d, s2_d in zip(s1_h.data, s2_h.data):
            assert s1_d == s2_d

