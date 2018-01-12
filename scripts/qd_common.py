import _init_paths
import yaml
from collections import OrderedDict
import sys
import os
from multiprocessing import Process
from multiprocessing import Event
import numpy as np
import logging
import os.path as op
import caffe
import time
from google.protobuf import text_format
import base64
import cv2
from itertools import izip


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
    mean_value = train_net_param.layer[0].transform_param.mean_value
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

def setup_yaml():
    """ https://stackoverflow.com/a/8661021 """
    represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)    

def init_logging():
    np.seterr(all='raise')
    logging.basicConfig(level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s',
    datefmt='%m-%d %H:%M:%S',
    )
    setup_yaml()

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        if not os.path.exists(path):
            os.makedirs(path)

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    nparr = np.fromstring(jpgbytestring, np.uint8)
    try:
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR);
    except:
        return None;

def read_to_buffer(file_name):
    with open(file_name, 'r') as fp:
        all_line = fp.read()
    return all_line

def write_to_yaml_file(context, file_name):
    ensure_directory(op.dirname(file_name))
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, default_flow_style=False)

def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        return yaml.safe_load(fp)

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
):
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()

    caffe.log('Using devices %s' % str(gpus))

    procs = []
    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver, snapshot, weights, gpus, timing, uid, rank))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

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

def solve(proto, snapshot, weights, gpus, timing, uid, rank):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(str(proto))
    if snapshot and len(snapshot) != 0:
        solver.restore(str(snapshot))

    if weights and len(weights) != 0:
        solver.net.copy_from(str(weights),ignore_shape_mismatch=True)

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    if timing and rank == 0:
        register_timing(solver, nccl)
    else:
        solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)

    solver.step(solver.param.max_iter)
    if rank == 0:
        solver.snapshot()

def default_data_path(dataset):
    proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)));
    result = {}
    data_root = os.path.join(proj_root, 'data', dataset)
    result['source'] = os.path.join(data_root, 'train.tsv')
    result['test_source'] = os.path.join(data_root, 'test.tsv')
    result['labelmap'] = os.path.join(data_root, 'labelmap.txt')
    result['source_idx'] = os.path.join(data_root, 'train.lineidx')
    result['test_source_idx'] = os.path.join(data_root, 'test.lineidx')
    return result
