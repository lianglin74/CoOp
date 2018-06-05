import matplotlib
matplotlib.use('Agg')
from shutil import copyfile
import cPickle as pkl
import base64
import argparse
import cv2
from pprint import pformat
from datetime import datetime
import _init_paths
import os
import gen_prototxt
import caffe
import re
import quickcaffe.modelzoo as mzoo
from caffe.proto import caffe_pb2
import numpy as np
from google.protobuf import text_format
from deteval import deteval_iter
from yolodet import tsvdet_iter
from yolodet import tsvdet
import time
from pprint import pprint
from qd_common import init_logging
from qd_common import calculate_macc

from taxonomy import load_label_parent, get_nick_name, noffset_to_synset
from qd_common import write_to_file, read_to_buffer, ensure_directory
from qd_common import parallel_train, LoopProcess, plot_to_file
from qd_common import default_data_path, caffe_train
from qd_common import caffe_param_check
from qd_common import worth_create
from qd_common import write_to_yaml_file
#from qd_common import add_layer_for_extract
from qd_common import parse_training_time
from qd_common import caffemodel_num_param
from qd_common import remove_nms, adjust_tree_prediction_threshold
from qd_common import process_run
from qd_common import load_solver, load_net, load_net_from_str
from qd_common import Model
from qd_common import construct_model
from qd_common import check_best_iou, update_kernel_active, update_crop_type
from qd_common import update_kernel_active2
from qd_common import update_conv_channels, update_bn
from qd_common import add_yolo_low_shot_regularizer
from qd_common import remove_data_augmentation
from qd_common import load_list_file
from qd_common import visualize_train
from qd_common import add_yolo_angular_loss_regularizer
from process_image import show_net_input
from taxonomy import load_label_parent
from tsv_io import TSVDataset, tsv_reader, tsv_writer
from process_tsv import TSVFile
from process_image import draw_bb, show_image
from process_dataset import dynamic_process_tsv
import os.path as op
from qd_common import parse_basemodel_with_depth
from taxonomy import LabelTree
from yolodet import im_detect
from qd_common import img_from_base64
from qd_common import load_from_yaml_file
from yolodet import im_classify
from itertools import izip

from process_tsv import update_yolo_test_proto
from process_tsv import build_taxonomy_impl
from process_tsv import populate_dataset_details
import copy
import random

import matplotlib.pyplot as plt
import simplejson as json
import yaml
from process_tsv import TSVTransformer
from vis.eval import parse_loss
import glob
import math
import logging
import yoloinit
from multiprocessing import Process, Queue
from qd_common import get_mpi_rank, get_mpi_size
from qd_common import get_mpi_local_rank, get_mpi_local_size
from qd_common import concat_files
from yoloinit import data_dependent_init2

def num_non_empty_lines(file_name):
    with open(file_name) as fp:
        context = fp.read()
    lines = context.split('\n')
    return sum([1 if len(line.strip()) > 0 else 0 for line in lines])

def create_shuffle_for_init(data):
    dataset = TSVDataset(data)
    out_file = op.join(dataset._data_root, 'train.init.shuffle.txt')
    if op.isfile(out_file):
        return
    label_to_idxes = {}
    shuffles = [row for row in tsv_reader(dataset.get_shuffle_file('train'))]
    for row in dataset.iter_data('train', 'inverted.label'):
        label = row[0]
        str_idxes = row[1].split(' ')
        if len(str_idxes) == 0:
            idxes = []
        else:
            idxes = map(int, str_idxes)
        assert len(idxes) > 0
        random.shuffle(idxes)
        label_to_idxes[label] = idxes
    logging.info(len(label_to_idxes))
    result = []
    all_idx = []
    for i in range(200):
        for label in label_to_idxes:
            idxes = label_to_idxes[label]
            all_idx.append(idxes[i % len(idxes)])
            result.append(shuffles[idxes[i % len(idxes)]])
    tsv_writer(result, out_file)
    
    label_to_num = {}
    for i, row in enumerate(dataset.iter_composite('train', 'label', 
            version=-1,
            filter_idx=all_idx)):
        if (i % 1000) == 0:
            logging.info('checking {}'.format(i))
        rects = json.loads(row[1])
        for r in rects:
            if r['class'] not in label_to_num:
                label_to_num[r['class']] = 0
            label_to_num[r['class']] = label_to_num[r['class']] + 1
    label_num = [(l, label_to_num[l]) for l in label_to_num]
    label_num = sorted(label_num, key=lambda x: x[1])
    logging.info(pformat(label_num))
            

def data_dependent_init_tree2(pre_trained_full_expid, 
        target_full_expid):
    target_exp = CaffeWrapper(full_expid=target_full_expid,
            load_parameter=True)
    pre_exp = CaffeWrapper(full_expid=pre_trained_full_expid, 
            load_parameter=True)

    pretrained_proto = pre_exp._path_env['train_proto_file']
    pretrained_weight = pre_exp.best_model().model_param
    target_proto = target_exp._path_env['train_proto_file']
    init_target_proto = target_proto + '_init.prototxt'

    net = load_net(target_proto)
    num_class = 1000
    found = False
    for l in net.layer:
        if l.type == 'TsvBoxData':
            assert not found
            found = True
            p = l.tsv_data_param
            p.batch_size = 8
            origin_shuffle = p.source_shuffle
            data = op.basename(op.dirname(origin_shuffle))
            populate_dataset_details(data, check_image_details=False)
            #create_shuffle_for_init2(data)
            create_shuffle_for_init(data)
            p.source_shuffle = op.join(op.dirname(origin_shuffle),
                'train.init.shuffle.txt')
            assert op.isfile(p.source_shuffle)
        elif l.type == 'SoftmaxTreeWithLoss':
            tree_file = l.softmaxtree_param.tree
            num_class = len(load_list_file(tree_file))
    assert found
    write_to_file(str(net), init_target_proto)
    
    out_fname = op.join(op.dirname(target_proto), 'snapshot',
        'model_iter_-1.caffemodel')
    if not op.isfile(out_fname) or True:
        process_run(data_dependent_init2, pretrained_weight,
            pretrained_proto, init_target_proto, out_fname,
            tr_cnt=20,
            max_iters=4 * num_class)
    return out_fname

class ProtoGenerator(object):
    def __init__(self, base_model_with_depth, num_classes, **kwargs):
        self._base_model_with_depth = base_model_with_depth
        self._num_classes = num_classes
        self._kwargs = kwargs

    def generate_prototxt(self, train_net_file, test_net_file, solver_file):
        base_model_with_depth, num_classes = self._base_model_with_depth, self._num_classes
    
        net = self._gen_net(deploy=False)
        write_to_file(net, train_net_file)
    
        net = self._gen_net(deploy=True)
        write_to_file(net, test_net_file)
    
        solver = self._gen_solver(train_net_file)
        write_to_file(solver, solver_file)

    def _gen_net(self, deploy=False):
        base_model_with_depth, num_classes = self._base_model_with_depth, self._num_classes
        kwargs = self._kwargs

        cpp_version = kwargs.get('cpp_version', False)
        detmodel = kwargs.get('detmodel', 'fasterrcnn')

        model_parts = re.findall(r'\d+|\D+', base_model_with_depth)
        model_name = model_parts[0].lower()
        model_depth = -1 if len(model_parts) == 1 else int(model_parts[1])
        
        det = self._create_model(detmodel)
        model = self._create_model(model_name)
        
        n = caffe.NetSpec()
        if not deploy:
            det.add_input_data(n, num_classes, **kwargs)
        else:
            n.data = caffe.layers.Layer()
            if detmodel == 'yolo' or detmodel == 'fasterrcnn':
                # create a placeholder, and replace later
                n.im_info = caffe.layers.Layer()
            elif detmodel == 'classification':
                n.label = caffe.layers.Layer()
            else:
                assert False, '{} is not supported'.format(detmodel)

        model.add_body(n, depth=model_depth, lr=1, deploy=deploy, **kwargs)
        det.add_body(n, lr=1, num_classes=num_classes, cnnmodel=model,
                deploy=deploy, cpp_version=cpp_version, **kwargs)

        layers = str(n.to_proto()).split('layer {')[1:]
        layers = ['layer {' + x for x in layers]
        if detmodel == 'fasterrcnn':
            im_info2 = 3
            image_dim = 224
        elif detmodel == 'yolo':
            im_info2 = 2
            image_dim = 416
        elif detmodel == 'classification':
            image_dim = 224
        else:
            assert False

        if deploy:
            layers[0] = 'input: {}\ninput_shape {{\n  dim: {}\n  dim: {}\n  dim: {}\n  dim: {}\n}}\n'.format(
                    '"data"', 1, 3, image_dim, image_dim)
            if detmodel == 'classification':
                layers[1] = ''
            else:
                layers[1] = 'input: {}\ninput_shape {{\n  dim: {}\n  dim: {}\n}}\n'.format(
                        '"im_info"', 1, im_info2)
        proto_str = ''.join(layers)
        proto_str = proto_str.replace("\\'", "'")
        
        prefix = detmodel

        proto_str = 'name: "{}-{}"\n{}'.format(prefix, base_model_with_depth, proto_str)

        net = load_net_from_str(proto_str)
        if self._kwargs.get('kernel_active', -1) > 0:
            if self._kwargs.get('use_shift', False):
                update_kernel_active2(net, **self._kwargs)
            else:
                assert False
                update_kernel_active(net, self._kwargs['kernel_active'],
                        self._kwargs.get('kernel_active_skip', 0))

        if self._kwargs.get('crop_type', None):
            update_crop_type(net, self._kwargs['crop_type'], 
                    self._kwargs.get('inception_crop_kl', None))
        if self._kwargs.get('channels_factor', 1) != 1:
            update_conv_channels(net, self._kwargs['channels_factor'],
                    self._kwargs.get('channels_factor_skip', 0))
        if self._kwargs.get('net_bn', False):
            update_bn(net)
        if not deploy and self._kwargs.get('yolo_low_shot_regularizer', False):
            add_yolo_low_shot_regularizer(net,
                    kwargs['dataset_ops'][0]['labels_idx'])
        if not deploy and self._kwargs.get('yolo_angular_loss', False):
            add_yolo_angular_loss_regularizer(net, **self._kwargs)
            self._kwargs['layer_wise_reduce'] = False
        if self._kwargs.get('no_bias', False):
            from qd_common import set_no_bias 
            set_no_bias(net, self._kwargs['no_bias'])
        if not deploy and len(self._kwargs.get('last_fixed_param', '')) > 0:
            from qd_common import fix_net_parameters
            fix_net_parameters(net, **self._kwargs)
        if not deploy and self._kwargs.get('num_bn_fix', 0) > 0:
            from qd_common import fix_net_bn_layers
            fix_net_bn_layers(net, self._kwargs['num_bn_fix'])

        proto_str = str(net)
        
        return proto_str
    
    def _gen_solver(self, train_net_file):
        kwargs = self._kwargs
        train_net_dir = os.path.dirname(train_net_file)
        snapshot_prefix = os.path.join(train_net_dir, 'snapshot', 'model')
        ensure_directory(os.path.dirname(snapshot_prefix))

        max_iters = kwargs.get('max_iters', None)

        num_train_images = kwargs.get('num_train_images', 5011)
        def to_iter(e):
            if type(e) is str and e.endswith('e'):
                effective_batch_size = kwargs.get('effective_batch_size', 64)
                iter_each_epoch = 1. * num_train_images / effective_batch_size
                return int(float(e[:-1]) * iter_each_epoch)
            else:
                return int(e)

        if type(max_iters) is str:
            max_iters = to_iter(max_iters)
        elif max_iters == None:
            max_iters = 10000 # default values

        logging.info('max iter: {}'.format(max_iters))

        if 'stageiter' in kwargs and kwargs['stageiter']:
            stageiter = [to_iter(si) for si in kwargs['stageiter']]
        else:
            stageiter_dist = kwargs.get('stageiter_dist', 'origin')
            if stageiter_dist == 'origin':
                stageiter = map(lambda x:int(x*max_iters/10000), 
                        [100,5000,9000,10000])
            else:
                assert stageiter_dist == 'compact'
                stageiter = map(lambda x:int(x*max_iters/7000), 
                        [100,5000,6000,7000])


        extra_param = {}
        if 'burn_in' in kwargs:
            extra_param['burn_in'] = to_iter(kwargs['burn_in'])
        if 'burn_in_power' in kwargs:
            extra_param['burn_in_power'] = kwargs['burn_in_power']

        lr_policy = kwargs.get('lr_policy', 'multifixed')
        
        if 'snapshot' in kwargs:
            snapshot = kwargs['snapshot']
        else:
            snapshot = max(int(num_train_images / 100), 500)
        solver_param = {
                'train_net': op.relpath(train_net_file), 
                'lr_policy': lr_policy,
                'gamma': 0.1,
                'display': kwargs.get('display', 100),
                'momentum': 0.9,
                'weight_decay': kwargs.get('weight_decay', 0.0005),
                'snapshot': snapshot,
                'snapshot_prefix': op.relpath(snapshot_prefix),
                'max_iter': max_iters,
                }

        if kwargs.get('solver_debug_info', False):
            solver_param['debug_info'] = kwargs['solver_debug_info']
        if kwargs.get('iter_size', 1) != 1:
            solver_param['iter_size'] = kwargs['iter_size']

        if lr_policy == 'multifixed':
            if kwargs.get('stagelr', None):
                solver_param['stagelr'] = kwargs['stagelr']
            else:
                solver_param['stagelr'] = [0.0001,0.001,0.0001,0.00001]
            solver_param['stageiter'] = stageiter
        else:
            solver_param['base_lr'] = kwargs['base_lr']
            solver_param['stepsize'] = kwargs['stepsize']

        if 'layer_wise_reduce' in kwargs:
            solver_param['layer_wise_reduce'] = kwargs['layer_wise_reduce']

        for key in extra_param:
            solver_param[key] = extra_param[key]

        if 'base_lr' in kwargs:
            solver_param['base_lr'] = kwargs['base_lr']

        solver = caffe_pb2.SolverParameter(**solver_param)

        return str(solver)

    def _create_model(self, model_name):
        if model_name == 'zf':
            return mzoo.ZFNet(add_last_pooling_layer=False)
        elif model_name == 'zfb':
            return mzoo.ZFBNet(add_last_pooling_layer=False)
        elif model_name == 'vgg': 
            return mzoo.VGG(add_last_pooling_layer=False)
        elif model_name == 'resnet':
            return mzoo.ResNet(add_last_pooling_layer=False)
        elif model_name == 'darknet': 
            return mzoo.DarkNet(add_last_pooling_layer=False)
        elif model_name == 'squeezenet': 
            return mzoo.SqueezeNet(add_last_pooling_layer=False)
        elif model_name == 'fasterrcnn':
            return mzoo.FasterRCNN()
        elif model_name == 'yolo':
            return mzoo.Yolo()
        elif model_name == 'vggstyle':
            return mzoo.VGGStyle()
        elif model_name == 'classification':
            return mzoo.Classification()
        elif model_name == 'sebninception':
            return mzoo.SEBNInception()
        elif model_name == 'seresnet':
            return mzoo.SEResnet()
        else:
            assert False

class CaffeWrapper(object):
    def __init__(self, load_parameter=False, **kwargs):
        self._kwargs = {} 
        if load_parameter:
            full_expid = kwargs['full_expid']
            self._output = op.join('output', full_expid)
            yaml_pattern = op.join(self._output,
                    'parameters_*.yaml')
            yaml_files = glob.glob(yaml_pattern)
            if len(yaml_files) > 0:
                def parse_time(f):
                    m = re.search('.*parameters_(.*)\.yaml', f) 
                    t = datetime.strptime(m.group(1), '%Y_%m_%d_%H_%M_%S')
                    return t
                times = [parse_time(f) for f in yaml_files]
                fts = [(f, t) for f, t in izip(yaml_files, times)]
                fts.sort(key=lambda x: x[1], reverse=True)
                yaml_file = fts[0][0]
            else: 
                yaml_file = op.join(self._output, 'parameters.yaml')
            logging.info('using {}'.format(yaml_file))
            param = load_from_yaml_file(yaml_file)
            self._kwargs = param
            if 'debug_detect' in param and 'debug_detect' not in kwargs:
                del self._kwargs['debug_detect']
            if 'force_predict' in param and 'force_predict' not in kwargs:
                del self._kwargs['force_predict']
        # note if load_parameter is true, the self._kwargs has been initialized
        # by some parameters. Thus, don't overwrite it simply
        for k in kwargs: 
            self._kwargs[k] = copy.deepcopy(kwargs[k])

        self._data = self._kwargs['data']
        self._net = self._kwargs['net']
        self._expid = self._kwargs['expid']
        self._path_env = default_paths(self._net, self._data, self._expid)
        self._output_root = self._path_env['output_root']
        self._output = self._path_env['output']
        self._full_expid = op.basename(self._output)


        if 'detmodel' not in self._kwargs:
            self._kwargs['detmodel'] = 'yolo'
        self._detmodel = self._kwargs['detmodel']
        if 'yolo_max_truth' not in self._kwargs:
            self._kwargs['yolo_max_truth'] = 30

        self._tree = None
        if self._kwargs.get('yolo_tree', False):
            source_dataset = TSVDataset(self._data)
            self._kwargs['target_synset_tree'] = source_dataset.get_tree_file()

        self._test_data = self._kwargs.get('test_data', self._data)
        self._test_split = self._kwargs.get('test_split', 'test')
        self._test_dataset = TSVDataset(self._test_data)
        self._test_source = self._test_dataset.get_data(self._test_split)
        source_dataset = TSVDataset(self._data)
        if self._test_data == self._data:
            source_dataset.dynamic_update(self._kwargs.get('dataset_ops', []))
            if self._kwargs.get('test_on_train', False):
                self._test_source = source_dataset.get_train_tsv()
            else:
                self._test_source = source_dataset.get_test_tsv_file()
        self._labelmap = source_dataset.get_labelmap_file()

        if 'yolo_extract_target_prediction' in self._kwargs:
            assert 'extract_features' not in self._kwargs
            self._kwargs['extract_features'] = 'target_prediction'

    def demo(self, source_image_tsv=None, m=None):
        labels = load_list_file(self._labelmap)
        if m is None:
            all_model = [construct_model(self._path_env['solver'],
                    self._path_env['test_proto_file'],
                    is_last=True)]
            all_model.extend(construct_model(self._path_env['solver'],
                    self._path_env['test_proto_file'],
                    is_last=False))
            best_avail_model = [m for m in all_model if op.isfile(m.model_param)][0]
        else:
            best_avail_model = m
        pixel_mean = best_avail_model.mean_value
        model_param = best_avail_model.model_param
        logging.info('param: {}'.format(model_param))
        test_proto_file = self._get_test_proto_file(best_avail_model)
        waitkey = 0 if source_image_tsv else 1
        thresh = 0.24
        gpu = 0
        from demo_detection import predict_online
        predict_online(test_proto_file, model_param, pixel_mean, labels,
                source_image_tsv, thresh, waitkey, gpu)
    
    def _get_num_classes(self):
        if 'num_classes' not in self._kwargs:
            if 'target_synset_tree' in self._kwargs:
                self._tree = LabelTree(self._kwargs['target_synset_tree'])
                num_classes = len(self._tree.noffsets)
            else:
                num_classes = num_non_empty_lines(self._labelmap)
        else:
            assert False
            num_classes = kwargs['num_classes']

        return num_classes

    def run_in_process(self, func, *args):
        def run_in_queue(que, *args):
            que.put(func(*args))

        que = Queue()
        process_args = [que]
        process_args.extend(args)
        p = Process(target=run_in_queue, args=tuple(process_args))
        p.daemon = False
        p.start()
        p.join()
        return que.get()

    def initialize_linear_layer_with_data(self, path_env, gpus):
        if len(gpus) > 0:
            caffe.set_device(gpus[0])
            caffe.set_mode_gpu()

        base_model_proto_test_file = path_env['basemodel'][:-11]+"_test.prototxt"
        new_base_net = yoloinit.data_dependent_init(path_env['basemodel'], base_model_proto_test_file, path_env['train_proto_file'])
        new_base_net_filepath = os.path.join(path_env['output'], path_env['basemodel'][:-11]+"_dpi.caffemodel")
        new_base_net.save(new_base_net_filepath)
        return new_base_net_filepath

    def monitor_train(self):
        init_logging()
        while True:
            logging.info('monitoring')
            finished_time = False
            if self._is_train_finished():
                self.cpu_test_time()
                self.gpu_test_time()
                finished_time = True
            is_unfinished = self.tracking()
            self.plot_loss()
            self.param_distribution()
            s = self.get_iter_acc()
            self._display(s)
            if not is_unfinished and finished_time:
                break 
            logging.info('sleeping')
            time.sleep(5)

    def cpu_test_time(self):
        str_s = '_'.join(map(str, self._kwargs.get('test_input_sizes', [416])))
        result_file = op.join(self._path_env['output'],
                'cpu_test_time_{}.yaml'.format(str_s))

        if not op.isfile(result_file):
            process_run(self._test_time, -1, result_file)

        r = load_from_yaml_file(result_file)
        return r
        
    def gpu_test_time(self):
        str_s = '_'.join(map(str, self._kwargs.get('test_input_sizes', [416])))
        result_file = op.join(self._path_env['output'], 
                'gpu_test_time_{}.yaml'.format(str_s))
        if not op.isfile(result_file):
            gpus = self._gpus()
            tested = False
            for g in gpus:
                if g >= 0:
                    process_run(self._test_time, g,result_file)
                    tested = True
                    break
            if not tested:
                return 0;
        r = read_to_buffer(result_file)
        return r

    def _test_time(self, gpu, result_file):
        if gpu >= 0:
            caffe.set_device(gpu)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        model = construct_model(self._path_env['solver'],
                self._path_env['test_proto_file'],
                is_last=True)
        
        if op.isfile(model.model_param):
            net = caffe.Net(str(model.test_proto_file), 
                    str(model.model_param), caffe.TEST)
        else:
            net = caffe.Net(str(model.test_proto_file), caffe.TEST)

        logging.info('gpu:{}->result_file:{}'.format(gpu, result_file))

        cols = self._test_dataset.iter_data(self._test_split) 

        ims = []
        for i, col in enumerate(cols):
            im = img_from_base64(col[2])
            ims.append(im)
            if i > 10:
                break

        start_time = time.time()
        all_stat = []
        for im in ims:
            if self._kwargs['detmodel'] == 'yolo':
                stat = {}
                scores, boxes = im_detect(net, im, model.mean_value,
                        network_input_size=self._kwargs.get('test_input_sizes',
                            [416])[0],
                        stat=stat,
                        **self._kwargs)
                all_stat.append(stat)
            else:
                scores = im_classify(net, im, model.mean_value,
                        scale=model.scale,
                        **self._kwargs)
        end_time = time.time()

        avg_time = (end_time - start_time) / len(ims)
        import socket
        write_to_yaml_file({'all_stat': all_stat, 
            'host_name': socket.gethostname(),
            'avg_time': avg_time},
                result_file)

    def _gpus(self):
        return self._kwargs.get('gpus', [0])

    def _is_train_finished(self):
        if not op.isfile(self._path_env['solver']) or \
                not op.isfile(self._path_env['test_proto_file']):
            return False

        model = construct_model(self._path_env['solver'], 
                self._path_env['test_proto_file'])
        train_proto = self._path_env['train_proto_file']
        finished = os.path.isfile(model.test_proto_file) \
                and os.path.isfile(model.model_param)

        return finished

    def train(self):
        data, net, kwargs = self._data, self._net, self._kwargs
        path_env = self._path_env

        # save teh following two fields for saving only
        self._kwargs['data']  = data
        self._kwargs['net'] = net
        write_to_yaml_file(self._kwargs, op.join(path_env['output'], 
            'parameters_{}.yaml'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))))

        source_dataset = TSVDataset(self._data)

        x = dynamic_process_tsv(source_dataset, path_env['output_data_root'], 
                self._tree,
                **kwargs)

        num_classes = self._get_num_classes()

        sources, source_labels, source_shuffles, data_batch_weights, labelmap = x

        if not kwargs.get('skip_genprototxt', False):
            if len(source_shuffles) > 0:
                num_train_images = sum([TSVFile(s).num_rows() for s in
                        source_shuffles])
            else:
                num_train_images = sum([TSVFile(s).num_rows() 
                    for ss in sources
                    for s in ss])

            p = ProtoGenerator(parse_basemodel_with_depth(net), num_classes, sources=sources, 
                    labelmap=labelmap,
                    source_labels=source_labels, 
                    source_shuffles=source_shuffles,
                    data_batch_weights=data_batch_weights,
                    num_train_images=num_train_images,
                    **kwargs)
            p.generate_prototxt(path_env['train_proto_file'], 
                    path_env['test_proto_file'], path_env['solver'])

        #self._ensure_macc_calculated()

        if len(self._kwargs.get('init_from', {})) > 0:
            self._path_env['basemodel'] = self._create_init_model(
                    self._kwargs['init_from'])
        else:
            pretrained_model_file_path = self.run_in_process(
                    self.initialize_linear_layer_with_data, path_env,
                    self._gpus()) if kwargs.get('data_dependent_init', False) else path_env['basemodel']
            self._path_env['basemodel'] = pretrained_model_file_path

        model = construct_model(path_env['solver'], path_env['test_proto_file'])
        if not kwargs.get('skip_train', False) and (not self._is_train_finished() or kwargs.get('force_train', False)):
            with open(path_env['log'], 'w') as fp:
                self._train()

        return model

    def _create_init_model(self, init_from):
        if init_from['type'] == 'ncc2':
            old_full_expid = init_from['full_expid']
            new_full_expid = self._full_expid
            assert self._kwargs.get('yolo_tree'), 'only tree-based is implemented'
            base_model = data_dependent_init_tree2(old_full_expid, 
                new_full_expid)
            # evaluate the accuracy
            m = self.best_model()
            m.model_param = base_model
            p = self.predict(m)
            self.evaluate(m, p)
            return base_model
        raise ValueError()
        if init_from['type'] == 'min_l2':
            init_model_path = op.join(self._output, 'init.caffemodel')
            base_c = CaffeWrapper(net=init_from['net'], 
                    data=init_from['data'], expid=init_from['expid'])
            base_best_model = base_c.best_model()
            from model_initialization import min_l2_init
            old_dataset = TSVDataset(init_from['data'])
            old_tsv = old_dataset.get_train_tsv()
            old_labelmap = old_dataset.load_labelmap()
            new_dataset = TSVDataset(init_from['new_data'])
            new_tsv = new_dataset.get_train_tsv()
            new_labelmap = new_dataset.load_labelmap()
            target_dataset = TSVDataset(self._data)
            target_labelmap = target_dataset.load_labelmap()
            process_run(min_l2_init, base_best_model.train_proto_file,
                    base_best_model.model_param,
                    self._path_env['train_proto_file'],
                    new_dataset.get_train_tsv(),
                    self._get_num_classes() - len(old_dataset.load_labelmap()),
                    0.1,
                    init_model_path)
            dest = 'snapshot/model_iter_0.caffemodel'
            ensure_directory(op.dirname(dest))
            copyfile(init_model_path, dest)
            model = construct_model(self._path_env['solver'], 
                    self._path_env['test_proto_file'], is_last=True)
            model.model_param = dest 
            predict_result = self.predict(model)
            self.evaluate(model, predict_result)
        return init_model_path

    def _ensure_macc_calculated(self):
        model = construct_model(self._path_env['solver'], 
                self._path_env['test_proto_file'], is_last=True)
        test_proto = self._get_test_proto_file(model)
        network_input_size = self._kwargs.get('test_input_sizes', [416])[0]
        net = load_net(test_proto)
        net.input_shape[0].dim.pop()
        net.input_shape[0].dim.pop()
        net.input_shape[0].dim.append(network_input_size)
        net.input_shape[0].dim.append(network_input_size)
        test_proto = '{1}_{0}{2}'.format(network_input_size,
                *op.splitext(test_proto))
        write_to_file(str(net), test_proto)
        macc_file = test_proto + '.macc'
        macc_png = macc_file + '.png'
        if not op.isfile(macc_file) or not op.isfile(macc_png) or True:
            logging.info('calculating macc...')
            macc = process_run(calculate_macc, test_proto)
            total = sum(m[1] for m in macc)
            write_to_file('{}\n{}'.format(total, str(pformat(macc))), macc_file)
            plot_to_file([m[0] for m in macc], [m[1] for m in macc], macc_png)

    def _get_test_proto_file(self, model):
        surgery = False
        test_proto_file = model.test_proto_file
        yolo_tree = self._kwargs.get('yolo_tree', False)
        if yolo_tree:
            if self._kwargs.get('softmax_tree_prediction_threshold', 0.5) != 0.5:
                surgery=True
            else:
                return test_proto_file
        blame = self._kwargs.get('yolo_blame', '')
        extract_target = self._kwargs.get('yolo_extract_target_prediction', False)
        need_label = len(blame) != 0 or extract_target
        fix_xy = self._kwargs.get('yolo_test_fix_xy', False)
        fix_wh = self._kwargs.get('yolo_test_fix_wh', False)
        extract_features = self._kwargs.get('extract_features', '')
        need_label = need_label or 'label' in extract_features
        if len(extract_features) > 0:
            assert not extract_target

        test_input_sizes = self._kwargs.get('test_input_sizes', [416])
        if len(test_input_sizes) > 1:
            surgery = True

        surgery = surgery or len(blame) != 0 or extract_target or fix_xy or fix_wh \
                or 'conf_debug' in extract_features or need_label
        
        #nms_type = self._kwargs.get('nms_type',
                #caffe.proto.caffe_pb2.RegionPredictionParameter.Standard)
        if 'class_specific_nms' in self._kwargs and not yolo_tree:
            surgery = True
        if 'yolo_test_thresh' in self._kwargs:
            surgery = True
        if 'test_tree_cls_specific_th_by_average' in self._kwargs:
            surgery = True
        if 'o_obj_loss' in extract_features:
            surgery = True
            need_label = True

        #surgery = surgery or nms_type != caffe.proto.caffe_pb2.RegionPredictionParameter.Standard

        if surgery:
            n = load_net(test_proto_file)
            l = n.layer[-1]
            out_file = test_proto_file
            #if nms_type != caffe.proto.caffe_pb2.RegionPredictionParameter.Standard:
                #assert l.type == 'RegionPrediction'
                #l.region_prediction_param.nms_type = nms_type
                #out_file = '{}.nms{}'.format(out_file, nms_type)
                #if 'gaussian_nms_sigma' in self._kwargs:
                    #l.region_prediction_param.gaussian_nms_sigma = \
                        #self._kwargs['gaussian_nms_sigma']
                    #out_file = '{}.gnms{}'.format(out_file,
                            #self._kwargs['gaussian_nms_sigma'])
            if 'class_specific_nms' in self._kwargs and not yolo_tree:
                l.region_output_param.class_specific_nms = self._kwargs['class_specific_nms']
                out_file = '{}_clasSpecificNMS{}'.format(out_file, 
                        self._kwargs['class_specific_nms'])
            if 'yolo_test_thresh' in self._kwargs:
                if l.type == 'RegionOutput':
                    l.region_output_param.thresh = \
                        self._kwargs['yolo_test_thresh']
                else:
                    assert l.type == 'RegionPrediction'
                    l.region_prediction_param.thresh = \
                            self._kwargs['yolo_test_thresh']
                out_file = '{}_th{}'.format(out_file,
                        self._kwargs['yolo_test_thresh'])
            if need_label:
                n.input.append('label')
                s = n.input_shape.add()
                s.dim.append(1)
                s.dim.append(self._kwargs.get('yolo_max_truth', 300) * 5)
            if len(blame) > 0:
                l.region_output_param.blame = blame
            if need_label:
                if l.type != 'RegionPrediction':
                    l.bottom.append('label')
            if extract_target or 'conf_debug' in extract_features:
                l.top.append('conf_debug')
            if extract_target:
                l.top.append('target_prediction')
            if fix_xy:
                l.region_output_param.fix_xy_output = True
            if fix_wh:
                l.region_output_param.fix_wh_output = True
            if len(blame) > 0:
                out_file = out_file + '.blame{}'.format(blame)
            if extract_target: 
                out_file = out_file + '.extracttarget'
            if fix_xy:
                out_file = out_file + '.fixXY'
            if fix_wh:
                out_file = out_file + '.fixWH'
            if 'test_tree_cls_specific_th_by_average' in self._kwargs:
                th = self._kwargs['test_tree_cls_specific_th_by_average']
                out_file = out_file + '.clsSpecificTreeAvg{}'.format(
                        th)
                from taxonomy import gen_cls_specific_th
                th_file = self._output + '.tree_thAvg{}.txt'.format(th)
                gen_cls_specific_th(l.region_output_param.tree, th, th_file)
                l.region_output_param.cls_specific_hier_thresh = th_file
            if 'o_obj_loss' in extract_features:
                out_file = '{}_extract{}'.format(out_file, extract_features)
                train_net = load_net(model.train_proto_file)
                blob_names = extract_features.split('.')
                del train_net.layer[0]
                train_net.input.extend(n.input)
                train_net.input_shape.extend(n.input_shape)
                n = train_net
                #add_layer_for_extract(n, train_net,
                        #blob_names)
                if 'angular_loss' in blob_names:
                    add_yolo_angular_loss_regularizer(n, **self._kwargs)
            if len(test_input_sizes) > 1:
                out_file = '{}.noNms'.format(out_file)
                remove_nms(n)
            if self._kwargs.get('yolo_tree', False) and \
                    self._kwargs.get('softmax_tree_prediction_threshold', 0.5) != 0.5:
                tree_th = self._kwargs['softmax_tree_prediction_threshold']
                out_file = '{}.TreePredTh{}'.format(out_file, tree_th)
                adjust_tree_prediction_threshold(n, tree_th)
                
            write_to_file(str(n), out_file)
            test_proto_file = out_file
        
        logging.info('test proto file: {}'.format(test_proto_file))
        return test_proto_file
    
    def _evaluate_loss_per_cls(self, model, predict_result):
        loss_layer = load_net(model.train_proto_file).layer[-1]
        assert loss_layer.type == 'RegionLoss'
        tree_file = loss_layer.region_loss_param.tree
        noffset_idx, noffset_parentidx, noffsets = load_label_parent(tree_file)
        loss = pkl.loads(read_to_buffer(predict_result))
        ave_prob = loss[0, 1, :, 0] / (loss[0, 0, :, 0] + 0.001)
        ave_loss = loss[0, 2, :, 0] / (loss[0, 0, :, 0] + 0.001)
        nick_names = [get_nick_name(noffset_to_synset(no)) for no in noffsets]
        tree_loss_file = predict_result + '.tree'
        write_to_file('\n'.join(['\t'.join(map(str, x)) for x in zip(noffsets, nick_names, ave_prob, ave_loss,
            loss[0, 0, :, 0])]), tree_loss_file)

        idx = sorted(range(loss.shape[2]), key=lambda x: ave_prob[x])
        fig, ax = plt.subplots(2, 2)
        ax[0, 0].plot(ave_prob[idx])
        ax[0, 0].grid()
        ax[0, 1].plot(ave_loss[idx])
        ax[0, 1].grid()
        ax[1, 0].plot(loss[0, 0, idx, 0])
        ax[1, 0].grid()
        plt.tight_layout()
        fig.savefig(tree_loss_file + '.png')
        plt.close(fig)
    
    def extract_loss(self, model):
        pass

    def predict_loss_per_cls(self, model):
        out_file = self._predict_file(model)
        if not worth_create(model.model_param, out_file):
            logging.info('skip to predict {}'.format(out_file))
            return out_file

        test_source = TSVFile(self._test_source)
        num_images = test_source.num_rows()

        # create the hacked train proto
        test_net = load_net(model.test_proto_file)
        train_net = load_net(model.train_proto_file)
        net = caffe.proto.caffe_pb2.NetParameter()
        data_layer = train_net.layer[1]
        remove_data_augmentation(data_layer)
        data_layer.tsv_data_param.batch_size = 32
        batch_size = data_layer.tsv_data_param.batch_size
        num_iter = num_images / (len(self._gpus()) * batch_size)
        logging.info('num iter: {}'.format(num_iter))
        assert len(data_layer.top) == 2
        data_layer.top[0] = 'data' 
        data_layer.top[1] = 'label' 
        net.layer.extend([data_layer])
        net.layer.extend(test_net.layer[: (len(test_net.layer) - 1)])
        loss_layer = train_net.layer[-1]
        loss_layer.top.append('x')
        loss_layer.top.append('loss_per_cls')
        loss_layer.propagate_down.append(False)
        loss_layer.propagate_down.append(False)
        loss_layer.loss_weight.append(1)
        loss_layer.loss_weight.append(0)
        loss_layer.loss_weight.append(0)
        net.layer.extend([loss_layer])
        for layer in net.layer:
            if layer.type == 'BatchNorm':
                layer.batch_norm_param.use_global_stats = True
        net_file = op.join(self._output, 'evaluate_loss_per_cls_net.prototxt')
        write_to_file(str(net), net_file)

        solver_param = {
                'train_net': net_file, 
                'lr_policy': 'step',
                'gamma': 0.1,
                'display': 100, 
                'momentum': 0,
                'weight_decay': 1, 
                'snapshot': 1000000, 
                'snapshot_prefix': op.join(self._output, 'snapshot', 'debug'),
                'base_lr': 0,
                'stepsize': 10000,
                'max_iter': num_iter }
        solver = caffe.proto.caffe_pb2.SolverParameter(**solver_param)
        solver_file = op.join(self._output,
                'evaluate_loss_per_cls_solver.prototxt')
        write_to_file(str(solver), solver_file)
        
        # debug to display the network input image
        #caffe.set_mode_gpu()
        #solver = caffe.SGDSolver(solver_file)
        #solver.net.copy_from(model.model_param)
        #solver.solve()
        #net = solver.net
        #i = 2
        #im = (net.blobs['data'].data[i].transpose((1, 2, 0)) + np.asarray(model.mean_value).reshape(1, 1, 3)).astype(np.uint8)
        #show_image(im)

        loss_per_cls = parallel_train(solver_file,
                None,
                model.model_param,
                self._gpus(),
                timing=False,
                extract_blob='loss_per_cls')

        write_to_file(pkl.dumps(loss_per_cls), out_file)
        return out_file

    def predict(self, model):
        if self._kwargs.get('predict_evaluate_loss_per_cls', False):
            return self.predict_loss_per_cls(model)
        test_proto_file = self._get_test_proto_file(model)

        model_param = model.model_param
        mean_value = model.mean_value
        scale = model.scale

        if not os.path.isfile(test_proto_file) or \
                not os.path.isfile(model_param):
            return None

        colkey, colimg = 0, 2

        outtsv_file = self._predict_file(model)
        logging.info(outtsv_file)

        if os.path.isfile(outtsv_file) and not self._kwargs.get('force_predict',
                False) and os.path.getmtime(outtsv_file) > os.path.getmtime(model_param):
            logging.info('skip to predict (exist): {}'.format(outtsv_file))
            return outtsv_file 
        
        mpi_rank = get_mpi_rank()
        mpi_size = get_mpi_size()
        if mpi_rank == 0 and mpi_size == 1:
            tsvdet_iter(test_proto_file, 
                    model_param, 
                    self._test_dataset.iter_data(self._test_split, unique=True,
                        progress=True),
                    colkey, 
                    colimg, 
                    mean_value, 
                    scale,
                    outtsv_file,
                    cmapfile=self._labelmap,
                    **self._kwargs)
        else:
            assert get_mpi_local_rank() == 0
            assert get_mpi_local_size() == 1
            logging.info('mpi_rank = {}; mpi_size = {}'.format(mpi_rank,
                mpi_size))
            num_test_images = self._test_dataset.num_rows(self._test_split)
            logging.info('num_test_images = {}'.format(num_test_images))
            assert num_test_images > 0
            # ceil of num_test_images/mpi_size
            num_image_per_process = (num_test_images + mpi_size - 1) / mpi_size
            logging.info('num_image_per_process = {}'.format(
                num_image_per_process))
            start_idx = num_image_per_process * mpi_rank
            end_idx = start_idx + num_image_per_process
            end_idx = min(num_test_images, end_idx)
            if end_idx <= start_idx:
                return
            logging.info('start_idx = {}; end_idx = {}'.format(start_idx, 
                end_idx))
            filter_idx = range(start_idx, end_idx)
            curr_outtsv_file = outtsv_file + '_{}_{}.tsv'.format(mpi_rank,
                    mpi_size)
            if not os.path.isfile(curr_outtsv_file) or self._kwargs.get('force_predict',
                    False) or os.path.getmtime(curr_outtsv_file) < os.path.getmtime(model_param):
                tsvdet_iter(test_proto_file, 
                        model_param, 
                        self._test_dataset.iter_data(self._test_split, unique=True,
                            progress=True,
                            filter_idx=filter_idx),
                        colkey, 
                        colimg, 
                        mean_value, 
                        scale,
                        curr_outtsv_file,
                        cmapfile=self._labelmap,
                        **self._kwargs)
            if mpi_rank == 0:
                # need to wait and merge all the results
                all_output =[outtsv_file + '_{}_{}.tsv'.format(mpi_rank,
                        mpi_size) for i in range(mpi_size)]
                ready = False
                while not ready:
                    ready = all(op.isfile(f) for f in all_output)
                    # even ready is true, let's sleep a while
                    time.sleep(10)
                    logging.info('ready = {}'.format(ready))
                logging.info('begin to merge the files')
                concat_files(all_output, outtsv_file)
                logging.info('finished to merge the files')

        logging.info('finished predict')

        return outtsv_file

    def _evaluate_yolo_target_prediction(self, model, predict_result):
        t_target_xs = {'s': [], 'm': [], 'l': []}
        t_target_ys = {'s': [], 'm': [], 'l': []}
        t_pred_xs = {'s': [], 'm': [], 'l': []}
        t_pred_ys = {'s': [], 'm': [], 'l': []}
        region_output_param = load_net(model.test_proto_file).layer[-1].region_output_param
        biases = region_output_param.biases
        multibin_xy = region_output_param.multibin_xy
        multibin_xy_high = region_output_param.multibin_xy_high
        multibin_xy_low = region_output_param.multibin_xy_low
        multibin_xy_count = region_output_param.multibin_xy_count
        if multibin_xy:
            multibin_xy_step = (multibin_xy_high - multibin_xy_low) / (multibin_xy_count - 1)
        rows = tsv_reader(predict_result)
        for row in rows:
            target = pkl.loads(base64.b64decode(row[1]))
            for r in range(target.shape[0]):
                if target[r, 1, 0] == 0 and target[r, 0, 0] == 0:
                    continue
                gt_x, gt_y, gt_w, gt_h = target[r, 0:4, 0]
                target_i, target_j = target[r, 4:6, 0]
                curr_target_x = gt_x * 13 - target_i
                curr_target_y = gt_y * 13 - target_j

                if not multibin_xy:
                    curr_pred_x = target[r, 7, 0]
                    curr_pred_y = target[r, 8, 0]
                else:
                    pred_x_hist = target[r, 7 : 7 + multibin_xy_count, 0]
                    assert np.abs(np.sum(pred_x_hist) - 1) < 0.001
                    idx_x = np.argmax(pred_x_hist)
                    curr_pred_x = idx_x * multibin_xy_step + multibin_xy_low
                    pred_y_hist = target[r, 7 + multibin_xy_count : 7 + 2 *
                        multibin_xy_count, 0]
                    assert np.abs(np.sum(pred_y_hist) - 1) < 0.001
                    idx_y = np.argmax(pred_y_hist)
                    curr_pred_y = idx_y * multibin_xy_step + multibin_xy_low

                if gt_w * gt_h < 1. / 13 / 13.:
                    t = 's'
                elif gt_w * gt_h < 9. / 13 / 13:
                    t = 'm'
                else:
                    t = 'l'
                t_target_xs[t].append(float(curr_target_x))
                t_target_ys[t].append(float(curr_target_y))
                t_pred_xs[t].append(float(curr_pred_x))
                t_pred_ys[t].append(float(curr_pred_y))
                
                n = int(target[r, 6, 0])
                check_best_iou(biases, gt_w * 13, gt_h * 13, n)

        evaluate_result = predict_result + '.eval'
        x = {'t_target_xs': t_target_xs, 
            't_target_ys': t_target_ys, 
            't_pred_xs': t_pred_xs, 
            't_pred_ys': t_pred_ys}
        data = json.dumps(x)
        write_to_file(data, evaluate_result)

        evaluate_result = evaluate_result + '.simple' 
        result = {}
        for key in t_target_xs.keys():
            result[key] = {}
            d = np.asarray(t_target_xs[key]) - np.asarray(t_pred_xs[key])
            result[key]['x'] = np.mean(np.abs(d))
            d = np.asarray(t_target_ys[key]) - np.asarray(t_pred_ys[key])
            result[key]['y'] = np.mean(np.abs(d))
        data = json.dumps(result)
        write_to_file(data, evaluate_result)

    def _evaluate_conf_debug(self, model, predict_result):
        net = load_net(model.test_proto_file)
        if net.layer[-1].type != 'RegionOutput':
            logging.info('last layer is not region output')
            return
        region_output_layer = net.layer[-1]
        if len(region_output_layer.region_output_param.tree) == 0:
            logging.info('softmax tree is not enabled')
            return
        tree_file = region_output_layer.region_output_param.tree

        label_to_idx, label_to_parientIdx, labels = load_label_parent(tree_file)
        rows = tsv_reader(predict_result)
        for i, row in enumerate(rows):
            assert len(row) == 3
            label = pkl.loads(base64.b64decode(row[1]))
            conf_debug = pkl.loads(base64.b64decode(row[2]))
    
    def _evaluate_features(self, model, predict_result):
        if self._detmodel != 'yolo':
            return
        out_file = predict_result + '.evaluate'
        rows = tsv_reader(predict_result)
        features = self._kwargs['extract_features'].split('.')
        feat_sum = {feat: 0 for feat in features}
        num_rows = 0
        for row in rows:
            num_rows = num_rows + 1
            for i, feat in enumerate(features):
                s = pkl.loads(base64.b64decode(row[i + 1]))
                feat_sum[feat] += s

        for feat in feat_sum:
            feat_sum[feat] /= num_rows
        logging.info(pformat(feat_sum))
        write_to_file(pformat(feat_sum), out_file)

    def evaluate(self, model, predict_result):
        if 'target_prediction' in self._kwargs.get('extract_features', ''):
            self._evaluate_yolo_target_prediction(model, predict_result)
            return
        elif 'conf_debug' in self._kwargs.get('extract_features', ''):
            self._evaluate_conf_debug(model, predict_result)
            return
        elif self._kwargs.get('predict_evaluate_loss_per_cls', False):
            self._evaluate_loss_per_cls(model, predict_result)
            return
        elif self._kwargs.get('extract_features', '') != '':
            self._evaluate_features(model, predict_result)
            return

        model_param = model.model_param
        data = self._data
        kwargs = self._kwargs

        if not model or not os.path.isfile(model.model_param):
            if model:
                logging.info('skip evaluation because model does not exist: {}'.format(model.model_param))
            return None

        if self._detmodel != 'classification':
            eval_file = deteval_iter(truth_iter=self._test_dataset.iter_data(
                self._test_split, 'label'), dets=predict_result, **kwargs)
            # create the index of the eval file so that others can load the
            # information fast
            result = None
            simple_file = eval_file + '.map.json'
            if worth_create(eval_file, simple_file):
                if result is None:
                    logging.info('data reading...')
                    eval_result= read_to_buffer(eval_file)
                    logging.info('json parsing...')
                    result = json.loads(eval_result)
                s = {}
                for size_type in result:
                    if size_type not in s:
                        s[size_type] = {}
                    for thresh in result[size_type]:
                        if thresh not in s[size_type]:
                            s[size_type][thresh] = {}
                        s[size_type][thresh]['map'] = \
                                result[size_type][thresh]['map']
                write_to_file(json.dumps(s, indent=4, sort_keys=True), simple_file)
            simple_file = eval_file + '.class_ap.json'
            if worth_create(eval_file, simple_file):
                if result is None:
                    eval_result= read_to_buffer(eval_file)
                    result = json.loads(eval_result)
                s = {}
                for size_type in result:
                    if size_type not in s:
                        s[size_type] = {}
                    for thresh in result[size_type]:
                        if thresh not in s[size_type]:
                            s[size_type][thresh] = {}
                        s[size_type][thresh]['class_ap'] = \
                                result[size_type][thresh]['class_ap']
                write_to_file(json.dumps(s, indent=4, sort_keys=True), simple_file)

        else:
            eval_file = self._perf_file(model)
            if os.path.isfile(eval_file) and not worth_create(model_param,
                    eval_file) and not self._kwargs.get('force_evaluate',
                            False):
                logging.info('skip since {} exists'.format(eval_file))
                return eval_file
            truth = {}
            predict = {}
            collector = TSVTransformer()
            def collect_label(row, result):
                assert row[0] not in result
                result[row[0]] = row[1]
            collector.ReadProcess(self._test_source, lambda row:
                    collect_label(row, truth))
            collector.ReadProcess(predict_result, lambda row:
                    collect_label(row, predict))
            predict_seq = []
            truth_seq = []
            predict_label = {}
            for key in predict:
                idx = np.argmax(map(float, predict[key].split(',')))
                predict_seq.append(idx)
                truth_seq.append(int(float(truth[key])))
            predict_seq = np.asarray(predict_seq)
            truth_seq = np.asarray(truth_seq)
            correct = sum(predict_seq == truth_seq)
            acc = float(correct) / len(predict_seq)
            result = {}
            result['acc'] = acc
            with open(eval_file, 'w') as fp:
                fp.write(json.dumps(result))
            logging.info('acc = {}'.format(acc))
        return eval_file

    def _predict_file(self, model):
        cc = [model.model_param, self._test_data, self._test_split]
        if len(self._kwargs.get('yolo_blame', '')) > 0:
            cc.append('blame_' + self._kwargs['yolo_blame'])
        if len(self._kwargs.get('extract_features', '')) > 0:
            cc.append('extract_{}'.format(self._kwargs['extract_features']))
        if self._kwargs.get('yolo_test_fix_xy', False):
            cc.append('fixXY')
        if self._kwargs.get('yolo_test_fix_wh', False):
            cc.append('fixWH')
        if self._kwargs.get('predict_style', None):
            cc.append('predictAs{}'.format(self._kwargs['predict_style']))
        if self._kwargs.get('yolo_test_maintain_ratio', False):
            cc.append('maintainRatio')
        if self._kwargs.get('test_on_train', False):
            cc.append('testOnTrain')
        if self._kwargs.get('predict_evaluate_loss_per_cls', False):
            cc.append('loss_per_cat')
        #if self._kwargs.get('nms_type',
                #caffe_pb2.RegionPredictionParameter.Standard) != \
                    #caffe_pb2.RegionPredictionParameter.Standard:
            #cc.append('nms{}'.format(self._kwargs['nms_type']))
            #if self._kwargs.get('gaussian_nms_sigma', 0.5) != 0.5:
                #cc.append('gnms{}'.format(self._kwargs['gaussian_nms_sigma']))
        test_input_sizes = self._kwargs.get('test_input_sizes', [416])
        if len(test_input_sizes) != 1 or test_input_sizes[0] != 416:
            cc.append('testInput{}'.format('.'.join(map(str,
                test_input_sizes))))
        if 'yolo_nms' in self._kwargs:
            cc.append('NMS{}'.format(self._kwargs['yolo_nms']))
        if self._kwargs.get('output_tree_path', False):
            cc.append('OutTreePath')
        if self._kwargs.get('softmax_tree_prediction_threshold', 0.5) != 0.5:
            cc.append('TreeThreshold{}'.format(self._kwargs['softmax_tree_prediction_threshold']))
        if not self._kwargs.get('class_specific_nms', True) :
            cc.append('ClsIndependentNMS')
        if self._kwargs.get('detmodel', 'yolo') == 'classification' and \
                self._kwargs.get('network_input_size', 224) != 224:
            cc.append('testInput{}'.format(self._kwargs['network_input_size']))
        if self._kwargs.get('predict_thresh', 0) != 0:
            cc.append('th{}'.format(self._kwargs['predict_thresh']))

        cc.append('predict')
        return '.'.join(cc)

    def _param_dist_file(self, model):
        return model.model_param + '.param'

    def _perf_file(self, model):
        if self._detmodel != 'classification':
            return op.splitext(self._predict_file(model))[0] + '.report'
        else:
            return self._predict_file(model) + '.report'
    
    def best_model_perf(self):
        solver = self._path_env['solver']
        test_proto_file = self._path_env['test_proto_file']
        best_model = construct_model(solver, test_proto_file,
                is_last=True)
        eval_result = self._perf_file(best_model)
        with open(eval_result, 'r') as fp:
            perf = json.loads(fp.read())

        return perf 

    def best_model(self, is_last=True):
        solver = self._path_env['solver']
        test_proto_file = self._path_env['test_proto_file']
        best_model = construct_model(solver, test_proto_file,
                is_last=is_last)
        return best_model

    def training_time(self):
        log_files = self._get_log_files()
        time_info = [parse_training_time(log) for log in log_files]
        display = None
        cost = []
        for one_info in time_info:
            for i in one_info[0]:
                if display == None:
                    display = i
                assert i == display
            cost += one_info[1]
        assert len(cost) > 0
        # remove the 5% smallest and 5% largest
        cost = sorted(cost)
        n = int(0.05 * len(cost))
        cost = cost[n : len(cost) - n]
        cost = np.asarray(cost)
        return np.mean(cost), np.sqrt(np.var(cost)), display
    
    def get_iter_acc(self):
        solver, test_proto_file = self._path_env['solver'], self._path_env['test_proto_file']
        if not os.path.exists(solver) or \
                not os.path.exists(test_proto_file):
            return [0], [0]
        all_model = []
        all_model.extend(construct_model(solver, test_proto_file,
                is_last=False))
        all_model.append(construct_model(solver, test_proto_file,
                is_last=True))
        all_ready_model = [m for m in all_model if os.path.isfile(m.model_param)]
        valid = [(m, self._perf_file(m)) for m in all_ready_model if
                        os.path.isfile(self._perf_file(m))] 
        
        ious = None
        xs, ys = [], []
        for v in valid:
            model, eval_result = v
            if self._detmodel == 'classification':
                xs.append(model.model_iter)
                ys.append(load_from_yaml_file(eval_result)['acc'])
            else:
                eval_result_simple = eval_result + '.map.json'
                if not op.isfile(eval_result_simple):
                    continue
                model_iter = model.model_iter
                xs.append(model_iter)
                logging.info('loading eval result: {}'.format(eval_result_simple))
                with open(eval_result_simple, 'r') as fp:
                    perf = json.loads(fp.read())
                if 'overall' in perf:
                    if ious == None:
                        ious = perf['overall'].keys()
                        ys = {}
                        for key in ious:
                            ys[key] = []
                    for key in ious:
                        if key not in perf['overall']:
                            ys[key].append(0)
                        else:
                            ys[key].append(perf['overall'][key]['map'])
                else:
                    ys.append(perf['acc'])
        return xs, ys

    def plot_acc(self):
        xs, ys = self.get_iter_acc()
        self._display((xs, ys))

    def tracking(self):
        data, net, kwargs = self._data, self._net, self._kwargs
        expid = kwargs.get('expid', '777')
        solver = self._path_env['solver']
        test_proto_file = self._path_env['test_proto_file']
        if not os.path.isfile(solver) or not os.path.isfile(test_proto_file):
            logging.info('proto file does not exist')
            return True
        all_model = construct_model(solver, test_proto_file,
                is_last=False)
        logging.info('there are {} models in total'.format(len(all_model)))
        all_ready_model = [m for m in all_model if os.path.isfile(m.model_param)]
        need_predict_model = any(not op.isfile(self._predict_file(m)) for m in
                all_ready_model)
        is_unfinished = len(all_model) > len(all_ready_model) or \
                need_predict_model

        sub_tasks_param1 = []
        sub_tasks_param2 = []
        start_time = time.time()
        random.shuffle(all_ready_model)
        for m in all_ready_model:
            predict_result = self.predict(m)
            if predict_result:
                sub_tasks_param1.append(m)
                sub_tasks_param2.append(predict_result)
                elapsed_time = time.time() - start_time
                if elapsed_time > 60 * 60 * 4:
                    logging.info('finished the {} predicts; {} left'.format(
                        len(sub_tasks_param1), 
                        len(all_ready_model) - len(sub_tasks_param1)))
                    break
        
        # the standard multiprocessing does not support the member function of
        # a class, but pathos' supports
        from pathos.multiprocessing import ProcessingPool as Pool
        pool = Pool()
        pool.map(self.evaluate, sub_tasks_param1, sub_tasks_param2)
        self.plot_loss()
        self.plot_acc()

        return is_unfinished

    def _tracking_one(self, iteration):
        solver = self._path_env['solver']
        test_proto_file = self._path_env['test_proto_file']
        m = construct_model(solver, test_proto_file,
                is_last=False, iteration=iteration)
        predict_result = self.predict(m)
        if predict_result:
            self.evaluate(m, predict_result)

    def param_num(self):
        all_model = construct_model(self._path_env['solver'],
                self._path_env['test_proto_file'],
                is_last=False)
        for m in all_model:
            if op.isfile(m.model_param):
                return caffemodel_num_param(m.model_param)

        raise ValueError('no model file exists')

    def param_distribution(self):
        if not os.path.exists(self._path_env['solver']) or \
                not os.path.exists(self._path_env['test_proto_file']):
                    return
        all_model = construct_model(self._path_env['solver'],
                self._path_env['test_proto_file'],
                is_last=False)
        all_ready_model = [m for m in all_model if os.path.isfile(m.model_param)]
        need_check_model = [m for m in all_ready_model if not
                os.path.isfile(self._param_dist_file(m))]
        for m in need_check_model:
            p = pcaffe_param_check(m.test_proto_file, m.model_param)
            with open(self._param_dist_file(m), 'w') as fp:
                fp.write(json.dumps(p, indent=4))

    def _display(self, s):
        xs, ys = s

        if len(xs) > 0:
            out_file = os.path.join(
                self._path_env['output'], 
                'map_{}.png'.format(self._test_data))
            logging.info('create {}'.format(out_file))
            if op.isfile(out_file):
                os.remove(out_file)
            plot_to_file(xs, ys, out_file)
        else:
            logging.info('nothing plotted')

    def _get_log_files(self):
        return [file_name for file_name in glob.glob(os.path.join(self._path_env['output'],
            '*_*')) if not file_name.endswith('png')]
    
    def plot_loss(self):
        xy = {}
        log_files = self._get_log_files() 
        worth_run = False
        for file_name in log_files:
            png_file = os.path.splitext(file_name)[0] + '.png'
            if worth_create(file_name, png_file):
                worth_run = True
                break
        if not worth_run:
            return
        for file_name in log_files:
            png_file = os.path.splitext(file_name)[0] + '.png'
            xs, ys = parse_loss(file_name)
            for x, y in zip(xs, ys):
                xy[x] = y
            if len(xs) > 0 and worth_create(file_name, png_file):
                plot_to_file(xs, ys, png_file)
        png_file = os.path.join(self._path_env['output'], 'loss.png')
        xys = sorted([(x, xy[x]) for x in xy], key=lambda i: i[0])
        xs = [xy[0] for xy in xys]
        ys = [xy[1] for xy in xys]
        plot_to_file(xs, ys, png_file)

    def _load_net(self, file_name):
        with open(file_name, 'r') as fp:
            all_line = fp.read()
        net_param = caffe.proto.caffe_pb2.NetParameter()
        text_format.Merge(all_line, net_param)
        return net_param

    def _load_solver(self, file_name):
        with open(file_name, 'r') as fp:
            all_line = fp.read()
        solver_param = caffe.proto.caffe_pb2.SolverParameter()
        text_format.Merge(all_line, solver_param)
        return solver_param

    def _train(self):
        solver_prototxt = self._path_env['solver']
        if self._kwargs.get('use_pretrained', True):
            pretrained_model = self._kwargs.get('basemodel', self._path_env['basemodel'])
        else:
            pretrained_model = None

        gpus = self._gpus() 
        restore_snapshot_iter = self._kwargs.get('restore_snapshot_iter', None)
        restore_snapshot = None
        if restore_snapshot_iter != None:
            if restore_snapshot_iter < 0:
                pattern = os.path.join(self._path_env['output'],
                        'snapshot',
                        'model_iter_*.solverstate')
                all_solverstate = glob.glob(pattern)
                if len(all_solverstate) > 0:
                    def parse_iter(p):
                        return int(p[p.rfind('_') + 1 : p.rfind('.')])
                    restore_snapshot_iter = max([parse_iter(p) for p in all_solverstate])
            if restore_snapshot_iter >= 0:
                restore_snapshot = os.path.join(self._path_env['output'],
                        'snapshot',
                        'model_iter_{}.solverstate'.format(restore_snapshot_iter))
            else:
               restore_snapshot_iter = None

        if len(gpus) == 1 and self._kwargs.get('debug_train', False):
            gpu = gpus[0]
            if gpu >= 0:
                caffe.set_device(gpu)
                caffe.set_mode_gpu()
            caffe.set_random_seed(777)
            solver = caffe.SGDSolver(str(solver_prototxt))
            if restore_snapshot_iter:
                solver.restore(restore_snapshot)
            elif pretrained_model:
                solver.net.copy_from(str(pretrained_model),
                        ignore_shape_mismatch=True)
            
            # visualize how the parameters are changed during the trainining. 
            #visualize_train(solver)
            while True:
                solver.step(50)
                # the commented shows how to save some data blob and how to
                # visualize the network input blob
                #np.save('data.npy', solver.net.blobs['data'].data)
                #show_net_input(solver.net.blobs['data'].data,
                        #solver.net.blobs['label'].data)

        else:
            logging.info('solver proto: {}'.format(solver_prototxt))
            logging.info('weights: {}'.format(pretrained_model))
            parallel_train(solver_prototxt, restore_snapshot,
                pretrained_model, gpus, timing=False)

def pcaffe_param_check(caffenet, caffemodel):
    def wcaffe_param_check(caffenet, caffemodel, result):
        result.put(caffe_param_check(caffenet, caffemodel))
    result = Queue()
    p = Process(target=wcaffe_param_check, args=(caffenet, caffemodel, result))
    p.start()
    p.join()
    return result.get()

def setup_paths(basenet, dataset, expid):
    proj_root = op.dirname(op.dirname(op.realpath(__file__)));
    model_path = op.join (proj_root,"models");
    data_root = op.join(proj_root,"data");
    data_path = op.join(data_root,dataset);
    basemodel_file = op.join(model_path ,basenet+'.caffemodel');
    output_path = op.join(proj_root,"output", "_".join([dataset,basenet,expid]));
    solver_file = op.join(output_path,"solver.prototxt");
    snapshot_path = op.join([output_path,"snapshot"]);
    DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = op.join(output_path, '%s_%s.log' %(basenet, DATE));
    caffe_log_file = op.join(output_path, '%s_caffe_'%(basenet));
    model_pattern = "%s/%s_faster_rcnn_iter_*.caffemodel"%(snapshot_path,basenet.split('_')[0].lower());
    deploy_path = op.join(output_path, "deploy");
    eval_output =  op.join(output_path, '%s_%s_testeval.tsv' %(basenet, DATE));
    return { "snapshot":snapshot_path, "solver":solver_file, "log":log_file, "output":output_path, 'data_root':data_root, 'data':data_path, 'basemodel':basemodel_file, 'model_pattern':model_pattern, 'deploy':deploy_path, 'caffe_log':caffe_log_file, 'eval':eval_output};

def default_paths(net, data, expid):
    path_env = setup_paths(net, data, expid)

    output_path = path_env['output'] 

    train_file_base = 'train.prototxt'
    test_file_base = 'test.prototxt'
    solver_file_base = 'solver.prototxt'
    
    data_path = default_data_path(data)
    for key in data_path:
        path_env[key] = data_path[key]

    path_env['train_proto_file'] = os.path.join(output_path, train_file_base)
    path_env['test_proto_file'] = os.path.join(output_path, test_file_base)
    path_env['solver'] = os.path.join(output_path, solver_file_base)
    path_env['output_root'] = os.path.dirname(path_env['output'])
    path_env['output_data_root'] = op.join(path_env['output_root'], 'data')
    path_env['output_data'] = os.path.join(path_env['output_root'], 'data', 
            data)

    path_env['labelmap'] = os.path.join(path_env['data'], 'labelmap.txt')

    return path_env

def yolotrain_main(**kwargs):
    has_tax_folder = 'taxonomy_folder' in kwargs
    if not has_tax_folder:
        yolotrain(**kwargs)
    else:
        yolo_tree_train(**kwargs)

def yolo_tree_train(**ikwargs):
    assert 'data' in ikwargs
    assert 'taxonomy_folder' in ikwargs
    kwargs = copy.deepcopy(ikwargs)
    build_taxonomy_impl(**kwargs)
    
    assert 'yolo_tree' not in kwargs or kwargs['yolo_tree']
    kwargs['yolo_tree'] = True
    kwargs['class_specific_nms'] = False
    kwargs['output_tree_path'] = True
    kwargs['yolo_tree_eval_label_lift'] = not kwargs['output_tree_path']
    kwargs['yolo_tree_eval_gt_lift'] = False
    kwargs['stageiter_dist'] = 'compact'

    # train only on the data with bbox
    bb_data = '{}_with_bb'.format(kwargs['data'])
    if len(TSVDataset(bb_data).get_train_tsvs()) == 0:
        logging.info('there is no image with bounding box labels')
        return
    dataset_ops = [{'op':'remove'},
            {'op':'add',
             'name': bb_data,
             'source':'train',
             'weight': 1},
            ]
    assert 'dataset_ops' not in kwargs or \
            len(kwargs['dataset_ops']) == 0
    kwargs['dataset_ops'] = dataset_ops
    expid = kwargs['expid']
    kwargs['expid'] = expid + '_bb_only'
    if 'test_data' not in kwargs:
        kwargs['test_data'] = bb_data
    c = CaffeWrapper(**kwargs)

    monitor_train_only = kwargs.get('monitor_train_only', False)
    if not monitor_train_only:
        model = c.train()
        p = c.predict(model)
        c.evaluate(model, p)
    else:
        c.monitor_train()
        assert c._is_train_finished()
        model = c.train()
    
    no_bb_data = '{}_no_bb'.format(kwargs['data'])
    # train it with no_bb as well
    if TSVDataset(no_bb_data).get_num_train_image() == 0:
        logging.info('there is no training image for image-level label')
        return
    
    if 'test_data' not in ikwargs:
        curr_task = copy.deepcopy(kwargs)
        curr_task['ovthresh'] = [-1]
        curr_task['test_data'] = no_bb_data
        c = CaffeWrapper(**curr_task)
        model = c.train()
        p = c.predict(model)
        c.evaluate(model, p)

    dataset_ops = [{'op':'remove'},
            {'op':'add',
             'name':bb_data,
             'source':'train',
             'weight': 1},
            {'op': 'add',
             'name': no_bb_data,
             'source': 'train',
             'weight': 3},
            ]
    kwargs['basemodel'] = model.model_param
    kwargs['expid'] = expid + '_bb_nobb'
    kwargs['dataset_ops'] = dataset_ops
    if 'tree_max_iters2' in kwargs:
        kwargs['max_iters'] = kwargs['tree_max_iters2']
    else:
        kwargs['max_iters'] = '30e'
    c = CaffeWrapper(**kwargs)
    if not monitor_train_only:
        model = c.train()
        p = c.predict(model)
        c.evaluate(model, p)
    else:
        c.monitor_train()
        assert c._is_train_finished()
    
    kwargs['ovthresh'] = [-1]
    if 'test_data' not in ikwargs:
        kwargs['test_data'] = no_bb_data
    c = CaffeWrapper(**kwargs)
    if not monitor_train_only:
        model = c.train()
        p = c.predict(model)
        c.evaluate(model, p)
    else:
        assert c._is_train_finished()
        c.monitor_train()

def yolotrain(data, net, **kwargs):
    init_logging()
    c = CaffeWrapper(data=data, net=net, **kwargs)

    monitor_train_only = kwargs.get('monitor_train_only', False)

    if not monitor_train_only:
        model = c.train()
        if not kwargs.get('debug_train', False) and model:
            predict_result = c.predict(model)
            if predict_result and not kwargs.get('skip_evaluate', False):
                c.evaluate(model, predict_result)
    else:
        c.monitor_train()

def get_confusion_matrix(data, net, test_data, expid, threshold=0.2, **kwargs):
    logging.info('deprecated: use get_confusion_matrix_by_predict_file')
    c = CaffeWrapper(data=data, net=net, 
            test_data=test_data,
            yolo_test_maintain_ratio = True,
            expid=expid,
            **kwargs)
    
    # load predicted results
    model = c.best_model()
    predict_file = c._predict_file(model)

    predicts, _ = load_labels(predict_file)

    # load the gt
    test_dataset = TSVDataset(test_data)
    test_label_file = test_dataset.get_data('test', 'label')
    gts, label_to_idx = load_labels(test_label_file)

    # calculate the confusion matrix
    confusion_pred_gt = {}
    confusion_gt_pred = {}
    update_confusion_matrix(predicts, gts, threshold, 
            confusion_pred_gt, 
            confusion_gt_pred)

    return {'predicts': predicts, 
            'gts': gts, 
            'confusion_pred_gt': confusion_pred_gt, 
            'confusion_gt_pred': confusion_gt_pred,
            'label_to_idx': label_to_idx}

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Yolo network')
    parser.add_argument('-g', '--gpus', help='GPU device id to use [0], e.g. -g 0 1 2 3.',
            type=int,
            nargs='+')
    parser.add_argument('-n', '--net', required=False, type=str.lower,
            help='only darknet19 is not supported', default='darknet19')
    parser.add_argument('-dm', '--detmodel', required=False, type=str,
            help='detection model', default='yolo')
    parser.add_argument('-t', '--iters', dest='max_iters',  help='number of iterations to train', 
            default=10000, required=False, type=str)
    parser.add_argument('-d', '--data', help='the name of the dataset', required=True)
    parser.add_argument('-e', '--expid', help='the experiment id', required=True)
    parser.add_argument('-ft', '--force_train', 
            default=False,
            action='store_true', 
            help='train the model even if the model is there', 
            required=False)
    parser.add_argument('-no-ar', '--no-add_reorg', dest='add_reorg',
            action='store_false',
            default=True,
            required=False,
            help='if the reorg layer should be added')
    parser.add_argument('-nc', '--num_extra_convs', 
            default=3,
            type=int,
            help='the number of extra conv layers')
    parser.add_argument('-eb', '--effective_batch_size', 
            default=64,
            type=int,
            help='effective batch size')
    parser.add_argument('-sg', '--skip_genprototxt', default=False,
            action='store_true', 
            help='skip the proto file generation')
    parser.add_argument('-st', '--skip_train',
            default=False,
            action='store_true',
            help='skip the training phase (only generate prototxt, taxonomy, ...)')
    parser.add_argument('-s', '--snapshot',
            help='the number of iterations to snaphot', required=False,
            type=int,
            default=500)
    parser.add_argument('-dpi', '--data_dependent_init',
            default=False,
            action='store_true',
            help='initialize the last linear layer of the model with data',
            required=False)
    parser.add_argument('-si', '--stageiter',
            default=None,
            help='e.g. 30e,30e,30e',
            type=lambda s: s.split(','),
            required=False)
    parser.add_argument('-sl', '--stagelr',
            default=None,
            help='e.g. 0.002,0.01',
            type=lambda s: [float(x) for x in s.split(',')],
            required=False)
    parser.add_argument('-fg', '--yolo_full_gpu', default=False,
            action='store_true', 
            help='full gpu')
    parser.add_argument('-ma', '--yolo_test_maintain_ratio', default=False,
            action='store_true', 
            help='maintain the aspect ratio')
    parser.add_argument('-fp', '--force_predict', default=False,
            action='store_true', 
            help='force to predict even if the predict file exists')
    parser.add_argument('-ti', '--test_input_sizes', default=[416],
            nargs='+', type=int, 
            help='test input sizes')
    parser.add_argument('-tf', '--taxonomy_folder', 
            default=argparse.SUPPRESS,
            type=str,
            help='taxonomy folder when -yt is specified')
    parser.add_argument('-snbv', '--yolo_softmax_norm_by_valid', 
            default=False,
            action='store_true', 
            help='normalize the softmax loss by VALID')
    parser.add_argument('-monitor', '--monitor_train_only',
            default=False,
            action='store_true',
            help='track the intermediate results')
    # 1. if the restore_snapshot_iter is specified, we will use that snapshot to
    # continue the training, and ignore the basemodel
    # 2. if the restore_snapshot_iter is None (not specified) or it is specified
    # as -1, but there is no snapshot in the output folder, we will try to use
    # the --basemodel as the initial weights. If the --basemodel is not
    # specified, we will try to use the model in models/${net}.caffemodel as
    # the initial model. 
    # Thus, it is ok to copy the pre-trained model to the
    # folder of models and rename it properly, and then specify the net
    # accordingly. However, this approach couples together how the network is
    # constructed and how the parameter is initialized, and may be deprecated
    # in the future.
    # Suggestions: always specify the restore_snapshot_iter
    # as -1 so that it can be trained continuously. Specify the basemodel only
    # if you know what is happening. A typical scenario is that we changed the
    # data, and the output will be in different folders, and we want to use
    # that model to initialize the current model.
    parser.add_argument('-rsi', '--restore_snapshot_iter',
            type=int,
            help='The iteration of the model used to restore the training. -1 for the last one')
    parser.add_argument('-bm', '--basemodel', 
            default=argparse.SUPPRESS,
            type=str,
            help='The pre-trained (weight) model path')
    return parser.parse_args()

if __name__ == '__main__':
    '''
    e.g. python scripts/yolotrain.py --data voc20 --iters 10000 --expid 789 \
            --net darknet19 \
            --gpus 4 5 6 7
    '''
    init_logging()
    args = parse_args()
    yolotrain_main(**vars(args))

