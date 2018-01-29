import matplotlib
matplotlib.use('Agg')
import argparse
import _init_paths
import os
import gen_prototxt
import caffe
import re
import quickcaffe.modelzoo as mzoo
from caffe.proto import caffe_pb2
import numpy as np
from google.protobuf import text_format
from qd_common import PyTee

from deteval import deteval
from yolodet import tsvdet
import time
from multiprocessing import Process, Queue

from qd_common import init_logging
from qd_common import write_to_file, read_to_buffer, ensure_directory
from qd_common import default_data_path
from qd_common import parse_basemodel_with_depth
from qd_common import parallel_train, LoopProcess
from qd_common import remove_nms
from qd_common import load_net
from qd_common import construct_model
import os.path as op
from datetime import datetime
from taxonomy import LabelTree

import matplotlib.pyplot as plt
import json
import logging
import yoloinit
from tsv_io import TSVDataset
from pprint import pformat
from process_tsv import build_taxonomy_impl
from process_dataset import dynamic_process_tsv
from process_tsv import TSVFile

def num_non_empty_lines(file_name):
    with open(file_name) as fp:
        context = fp.read()
    lines = context.split('\n')
    return sum([1 if len(line.strip()) > 0 else 0 for line in lines])

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
        
        return proto_str

    def _gen_solver(self, train_net_file):
        kwargs = self._kwargs
        train_net_dir = os.path.dirname(train_net_file)
        snapshot_prefix = os.path.join(train_net_dir, 'snapshot', 'model')
        ensure_directory(os.path.dirname(snapshot_prefix))

        max_iters = kwargs.get('max_iters', None)

        def to_iter(e):
            if type(e) is str and e.endswith('e'):
                num_train_images = kwargs.get('num_train_images', 5011)
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
            stageiter = map(lambda x:int(x*max_iters/10000), 
                    [100,5000,9000,10000000])

        lr_policy = kwargs.get('lr_policy', 'multifixed')

        solver_param = {
                'train_net': train_net_file, 
                'lr_policy': lr_policy,
                'gamma': 0.1,
                'display': 100,
                'momentum': 0.9,
                'weight_decay': 0.0005,
                'snapshot': kwargs.get('snapshot', 2000),
                'snapshot_prefix': snapshot_prefix,
                'iter_size': 1,
                'max_iter': max_iters
                }

        if lr_policy == 'multifixed':
            if kwargs.get('stagelr', None):
                solver_param['stagelr'] = kwargs['stagelr']
            else:
                solver_param['stagelr'] = [0.0001,0.001,0.0001,0.00001]
            solver_param['stageiter'] = stageiter

        if 'base_lr' in kwargs:
            solver_param['base_lr'] = kwargs['base_lr']

        solver = caffe_pb2.SolverParameter(**solver_param)

        return str(solver)

    def _list_models(self):
        return ['zf', 'zfb', 'vgg16', 'vgg19', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'squeezenet', 'darknet19']

    def _list_detmodel(self):
        return ['fasterrcnn', 'yolo']

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
        else:
            assert False

class CaffeWrapper(object):
    def __init__(self, data, net, **kwargs):
        self._data = data
        self._net = net
        self._kwargs = kwargs

        self._expid = kwargs['expid']
        self._path_env = default_paths(self._net, self._data, self._expid)

        source_dataset = TSVDataset(self._data)
        self._labelmap = source_dataset.get_labelmap_file()

        self._tree = None
        self._test_data = kwargs.get('test_data', data)
        self._test_source = self._path_env['test_source']
        if self._test_data != self._data:
            test_data_path = default_data_path(self._test_data)
            self._test_source = test_data_path['test_source']
            source_dataset = TSVDataset(self._test_data)
            if kwargs.get('test_on_train', False):
                self._test_source = source_dataset.get_train_tsv()
            else:
                self._test_source = source_dataset.get_test_tsv_file()
        else:
            source_dataset = TSVDataset(self._data)
            if kwargs.get('test_on_train', False):
                self._test_source = source_dataset.get_train_tsv()
            else:
                self._test_source = source_dataset.get_test_tsv_file()
        self._labelmap = source_dataset.get_labelmap_file()

        if 'yolo_extract_target_prediction' in self._kwargs:
            assert 'extract_features' not in self._kwargs
            self._kwargs['extract_features'] = 'target_prediction'

    def demo(self, source_image_tsv=None):
        labels = load_list_file(self._labelmap)
        all_model = [construct_model(self._path_env['solver'],
                self._path_env['test_proto_file'],
                is_last=True)]
        all_model.extend(construct_model(self._path_env['solver'],
                self._path_env['test_proto_file'],
                is_last=False))
        best_avail_model = [m for m in all_model if op.isfile(m.model_param)][0]
        pixel_mean = best_avail_model.mean_value
        model_param = best_avail_model.model_param
        logging.info('param: {}'.format(model_param))
        test_proto_file = self._get_test_proto_file(best_avail_model)
        waitkey = 0 if source_image_tsv else 1
        thresh = 0.24
        from demo_detection import predict_online
        predict_online(test_proto_file, model_param, pixel_mean, labels,
                source_image_tsv, thresh, waitkey)
    
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

        return num_classes

    def monitor_train(self, data, net, **kwargs):
        expid = kwargs.get('expid', '777')
        path_env = default_paths(net, data, expid)
        while True:
            if not self._train_monitor_once(data, path_env['solver'],
                    path_env['test_proto_file'], **kwargs):
                break
            time.sleep(5)

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

        write_to_file(pformat(self._kwargs), op.join(path_env['output'],
            'parameters.txt'))

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
            if kwargs.get('yolo_tree', False):
                kwargs['target_synset_tree'] = source_dataset.get_tree_file()
            p = ProtoGenerator(parse_basemodel_with_depth(net), num_classes, sources=sources, 
                    labelmap=labelmap,
                    source_labels=source_labels, 
                    source_shuffles=source_shuffles,
                    data_batch_weights=data_batch_weights,
                    num_train_images=num_train_images,
                    **kwargs)
            p.generate_prototxt(path_env['train_proto_file'], 
                    path_env['test_proto_file'], path_env['solver'])
        
        pretrained_model_file_path = self.run_in_process(
                self.initialize_linear_layer_with_data, path_env,
                self._gpus()) if kwargs.get('data_dependent_init', False) else path_env['basemodel']
        
        # one process can only call init_glog once. For this script, the glog
        # is not required to initialize and the log file will be in the output
        # folder
        #caffe.init_glog(str(path_env['log']))

        model = construct_model(path_env['solver'], path_env['test_proto_file'])
        if not self._is_train_finished() or kwargs.get('force_train', False):
            with open(path_env['log'], 'w') as fp:
                self._train()

        return model

    def predict(self, model):
        data = self._data
        kwargs = self._kwargs
        model_param = model.model_param
        mean_value = model.mean_value

        test_proto_file = self._get_test_proto_file(model)

        if not os.path.isfile(test_proto_file) or \
                not os.path.isfile(model_param):
            return None

        path_env = default_data_path(data)
        colkey = 0
        colimg = 2

        outtsv_file = self._predict_file(model)

        if os.path.isfile(outtsv_file) and not kwargs.get('force_predict',
                False):
            logging.info('skip to predict (exist): {}'.format(outtsv_file))
            return outtsv_file 

        tsvdet(test_proto_file, 
                model_param, 
                self._test_source, 
                colkey, 
                colimg, 
                mean_value, 
                outtsv_file,
                **kwargs)

        return outtsv_file

    def evaluate(self, model, predict_result):
        data = self._data
        kwargs = self._kwargs
        path_env = default_data_path(data)

        eval_file = deteval(truth=self._test_source,
                dets=predict_result, **self._kwargs)

        return eval_file

    def _get_test_proto_file(self, model):
        surgery = False
        test_proto_file = model.test_proto_file

        test_input_sizes = self._kwargs.get('test_input_sizes', [416])
        if len(test_input_sizes) > 1:
            surgery = True

        if surgery:
            n = load_net(test_proto_file)
            out_file = test_proto_file
            if len(test_input_sizes) > 1:
                out_file = '{}.noNms'.format(out_file)
                remove_nms(n)
            write_to_file(str(n), out_file)
            test_proto_file = out_file

        return test_proto_file

    def _predict_file(self, model):
        cc = [model.model_param, self._test_data]
        test_input_sizes = self._kwargs.get('test_input_sizes', [416])
        if self._kwargs.get('yolo_test_maintain_ratio', False):
            cc.append('maintainRatio')
        if len(test_input_sizes) != 1 or test_input_sizes[0] != 416:
            cc.append('testInput{}'.format('.'.join(map(str,
                test_input_sizes))))
        cc.append('predict')
        return '.'.join(cc)

    def _perf_file(self, model):
        test_proto_file, model_param, mean_value, model_iter = model
        return model_param + '.report'

    def _construct_model(self, solver, test_proto_file, is_last=True):
        logging.info('deprecating... use construct_model instead')
        solver_param = self._load_solver(solver)
        train_net_param = self._load_net(solver_param.train_net)
        data_layer = train_net_param.layer[0]
        mean_value = train_net_param.layer[0].transform_param.mean_value

        if is_last:
            last_model = '{0}_iter_{1}.caffemodel'.format(
                    solver_param.snapshot_prefix, solver_param.max_iter)
            return (test_proto_file, last_model, mean_value,
                    solver_param.max_iter)
        else:
            total = (solver_param.max_iter + solver_param.snapshot - 1) / solver_param.snapshot
            all_model = []
            for i in xrange(total + 1):
                if i == 0:
                    continue
                j = i * solver_param.snapshot
                j = min(solver_param.max_iter, j)
                last_model = '{0}_iter_{1}.caffemodel'.format(
                        solver_param.snapshot_prefix, j)
                all_model.append((test_proto_file, last_model, mean_value, j))
            return all_model

    def _train_monitor_once(self, data, solver_file, test_net_file, **kwargs):
        print 'monitoring'
        all_model = self._construct_model(solver_file, test_net_file, False) 
        all_ready_model = [m for m in all_model if os.path.isfile(m[1])]
        need_predict_model = [m for m in all_ready_model if not os.path.isfile(self._predict_file(m))]
        is_monitoring = len(all_model) > len(all_ready_model) or  len(need_predict_model) > 1 
        for m in need_predict_model:
            predict_result = self.predict(m)
            self.evaluate(m, predict_result)

        self._display([(m, self._perf_file(m)) for m in all_ready_model if
                os.path.isfile(self._perf_file(m))])

        return is_monitoring 

    def _display(self, valid):
        xs = []
        ys = []
        best_model_iter = -1
        best_model_class_ap = None
        for v in valid:
            model, eval_result = v
            test_proto_file, model_param, mean_value, model_iter = model
            xs.append(model_iter)
            with open(eval_result, 'r') as fp:
                perf = json.loads(fp.read())
            if best_model_iter < model_iter:
                best_model_iter = model_iter
                best_model_class_ap = perf['overall']['0.5']['class_ap']
            ys.append(perf['overall']['0.5']['map'])

        if best_model_class_ap:
            pprint((best_model_iter, best_model_class_ap))
        
        if len(xs) > 0:
            fig = plt.figure()
            plt.plot(xs, ys, '-o')
            plt.grid()
            fig.savefig(os.path.join(os.path.dirname(model_param), 'map.png'))
            plt.close(fig)
    
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
            pretrained_model = kwargs.get('pretrained_model', None)
            if pretrained_model:
                solver.net.copy_from(str(pretrained_model), 
                        ignore_shape_mismatch=True)
            solver.solve()
        else:
            parallel_train(solver_prototxt, restore_snapshot,
                pretrained_model, gpus, timing=False)


def Analyser(object):
    def AnalyzeSolver(self, solver):
        self.AnalyzeNet(solver.net)

    def AnalyzeNet(self, net):
        print 'parameter mean:'
        for key in net.params:
            value = solver.net.params[key]
            for v in value:
                print key, np.mean(v.data)
        print 'bottom/top mean:'
        for key in net.blobs:
            v = solver.net.blobs[key]
            print key, np.mean(v.data)

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

def yolo_tree_train(**kwargs):
    assert 'data' in kwargs
    assert 'taxonomy_folder' in kwargs
    build_taxonomy_impl(**kwargs)
    
    assert 'yolo_tree' not in kwargs or kwargs['yolo_tree']
    kwargs['yolo_tree'] = True
    kwargs['yolo_tree_eval_label_lift'] = True

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
    kwargs['test_data'] = bb_data
    c = CaffeWrapper(**kwargs)
    model = c.train()

    p = c.predict(model)
    c.evaluate(model, p)
    
    # train it with no_bb as well
    no_bb_data = '{}_no_bb'.format(kwargs['data'])
    if len(TSVDataset(no_bb_data).get_train_tsvs()) == 0:
        logging.info('there is no training image for image-level label')
        return

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
    c = CaffeWrapper(**kwargs)
    model = c.train()
    p = c.predict(model)
    c.evaluate(model, p)

    kwargs['ovthresh'] = [-1]
    kwargs['test_data'] = no_bb_data
    c = CaffeWrapper(**kwargs)
    model = c.train()
    p = c.predict(model)
    c.evaluate(model, p)


def yolotrain(data, net, **kwargs):
    init_logging()
    c = CaffeWrapper(data, net, **kwargs)

    monitor_train_only = kwargs.get('monitor_train_only', False)

    if not monitor_train_only:
        model = c.train()
        if not kwargs.get('debug_train', False) and model and \
                not kwargs.get('yolo_tree', False):
            predict_result = c.predict(model)
            c.evaluate(model, predict_result)
    else:
        c.monitor_train(data, net, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Yolo network')
    parser.add_argument('-g', '--gpus', help='GPU device id to use [0].',
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
    return parser.parse_args()

if __name__ == '__main__':
    '''
    e.g. python scripts/yolotrain.py --data voc20 --iters 10000 --expid 789 \
            --net darknet19 \
            --gpus 4,5,6,7
    '''
    init_logging()
    args = parse_args()
    yolotrain_main(**vars(args))

