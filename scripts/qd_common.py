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

def setup_yaml():
    """ https://stackoverflow.com/a/8661021 """
    represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
    yaml.add_representer(OrderedDict, represent_dict_order)    
    setup_yaml()

def init_logging():
    np.seterr(all='raise')
    logging.basicConfig(level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s: %(message)s',
    datefmt='%m-%d %H:%M:%S',
    )

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        if not os.path.exists(path):
            os.makedirs(path)

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

    solver = caffe.SGDSolver(proto)
    if snapshot and len(snapshot) != 0:
        solver.restore(snapshot)

    if weights and len(weights) != 0:
        solver.net.copy_from(weights,ignore_shape_mismatch=True)

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
