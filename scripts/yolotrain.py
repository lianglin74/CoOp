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
import os.path as op
from datetime import datetime
import matplotlib.pyplot as plt
import json
import logging
import yoloinit

def num_non_empty_lines(file_name):
    with open(file_name) as fp:
        context = fp.read()
    lines = context.split('\n')
    return sum([1 if len(line.strip()) > 0 else 0 for line in lines])

class ProtoGenerator:
    def generate_prototxt(self, base_model_with_depth, num_classes, 
            train_net_file, test_net_file, solver_file, **kwargs):
    
        net = self.gen_net(base_model_with_depth, num_classes, 
                deploy=False, **kwargs)
        write_to_file(net, train_net_file)
    
        net = self.gen_net(base_model_with_depth, num_classes, 
                deploy=True, **kwargs)
        write_to_file(net, test_net_file)
    
        solver = self.gen_solver(train_net_file, **kwargs)
        write_to_file(solver, solver_file)

    def gen_net(self, base_model_with_depth, num_classes, **kwargs):
        deploy = kwargs.get('deploy', False)
        cpp_version = kwargs.get('cpp_version', False)
        detmodel = kwargs.get('detmodel', 'FasterRCNN')

        assert base_model_with_depth.lower() in self._list_models(), 'Unsupported basemodel: %s' %base_model_with_depth 
        assert detmodel in self._list_detmodel(), 'Unsupported detmodel: %s' % detmodel

        model_parts = re.findall(r'\d+|\D+', base_model_with_depth)
        model_name = model_parts[0].lower()
        model_depth = -1 if len(model_parts) == 1 else int(model_parts[1])
        
        det = self._create_model(detmodel)
        model = self._create_model(model_name)
        
        n = caffe.NetSpec()
        if not deploy:
            det.add_input_data(n, num_classes, **kwargs)
        else:
            # create a placeholder, and replace later
            n.data = caffe.layers.Layer()
            n.im_info = caffe.layers.Layer()

        model.add_body(n, depth=model_depth, lr=1, deploy=deploy)
        det.add_body(n, lr=1, num_classes=num_classes, cnnmodel=model,
                cpp_version=cpp_version, **kwargs)

        layers = str(n.to_proto()).split('layer {')[1:]
        layers = ['layer {' + x for x in layers]
        im_info2 = 3 if detmodel == 'FasterRCNN' else 2
        image_dim = 224 if detmodel == 'FasterRCNN' else 416
        if deploy:
            layers[0] = 'input: {}\ninput_shape {{\n  dim: {}\n  dim: {}\n  dim: {}\n  dim: {}\n}}\n'.format(
                    '"data"', 1, 3, image_dim, image_dim)
            layers[1] = 'input: {}\ninput_shape {{\n  dim: {}\n  dim: {}\n}}\n'.format(
                    '"im_info"', 1, im_info2)
        proto_str = ''.join(layers)
        proto_str = proto_str.replace("\\'", "'")
        
        prefix = 'Faster-RCNN' if detmodel == 'FasterRCNN' else 'Yolo'
        
        return 'name: "{}-{}"\n{}'.format(prefix, base_model_with_depth, proto_str)

    def gen_solver(self, train_net_file, **kwargs):
        train_net_dir = os.path.dirname(train_net_file)
        snapshot_prefix = os.path.join(train_net_dir, 'snapshot', 'model')
        ensure_directory(os.path.dirname(snapshot_prefix))

        max_iters = kwargs.get('max_iters', None)

        def to_iter(e):
            if type(e) is str and e.endswith('e'):
                num_train_images = kwargs.get('num_train_images', 5011)
                effective_batch_size = kwargs.get('effective_batch_size', 64)
                iter_each_epoch = num_train_images / effective_batch_size
                return int(e[:-1]) * iter_each_epoch
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
        return ['FasterRCNN', 'Yolo']

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
        elif model_name == 'FasterRCNN':
            return mzoo.FasterRCNN()
        elif model_name == 'Yolo':
            return mzoo.Yolo()

class CaffeWrapper(object):
    def __init__(self, data, net, **kwargs):
        self._data = data
        self._net = net
        self._kwargs = kwargs

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

    def train(self):
        data, net, kwargs = self._data, self._net, self._kwargs
        expid = kwargs.get('expid', '777')
        path_env = default_paths(net, data, expid)

        if 'num_classes' not in kwargs:
            num_classes = num_non_empty_lines(path_env['labelmap'])
        else:
            num_classes = kwargs['num_classes']
        
        source = path_env['source']
        labelmap = path_env['labelmap'] 

        if not kwargs.get('skip_genprototxt', False):
            with open(path_env['source_idx'], 'r') as fp:
                num_train_images = len(fp.readlines())
            p = ProtoGenerator()
            p.generate_prototxt(parse_basemodel_with_depth(net), num_classes, 
                     path_env['train_proto_file'], path_env['test_proto_file'],
                     path_env['solver'], detmodel='Yolo', 
                     source=source, labelmap=labelmap, 
                     num_train_images=num_train_images,
                     **kwargs)
        
        gpus = [int(float(s)) for s in kwargs.get('gpus', '-1').split(',')]
        pretrained_model_file_path = self.run_in_process(self.initialize_linear_layer_with_data, path_env, gpus) if kwargs.get('data_dependent_init', False) else path_env['basemodel']
        caffe.init_glog(str(path_env['log']))

        model = self._construct_model(path_env['solver'], path_env['test_proto_file'])
        print 'log file: {0}'.format(path_env['log'])
        if not os.path.isfile(model[0]) \
                or not os.path.isfile(model[1]) \
                or kwargs.get('force_train', False):
            with open(path_env['log'], 'w') as fp, PyTee(fp, 'stdout'):
                self._train(path_env['solver'], pretrained_model=pretrained_model_file_path, **kwargs)

        return model

    def predict(self, model):
        data = self._data
        kwargs = self._kwargs
        _, model_param, mean_value, train_iter = model

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
                path_env['test_source'], 
                colkey, 
                colimg, 
                mean_value, 
                outtsv_file,
                **kwargs)

        return outtsv_file

    def evaluate(self, predict_result):
        data = self._data
        kwargs = self._kwargs
        path_env = default_data_path(data)

        eval_file = deteval(truth=path_env['test_source'],
                dets=predict_result)

        return eval_file

    def _get_test_proto_file(self, model):
        surgery = False
        test_proto_file = model[0]

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
        test_proto_file, model_param, mean_value, model_iter = model
        cc = [model_param]
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
            self.evaluate(predict_result)

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

    def _train(self, solver_prototxt, **kwargs):
        if kwargs.get('chdir', False):
            os.chdir(os.path.dirname(solver_prototxt))

        gpus = [int(float(s)) for s in kwargs.get('gpus', '-1').split(',')]

        if len(gpus) > 1:
            pretrained_model = kwargs.get('pretrained_model', '')
            parallel_train(solver_prototxt, None,
                pretrained_model, gpus, timing=False)
        elif len(gpus) == 1:
            gpu = gpus[0]
            if gpu >= 0:
                caffe.set_device(gpu)
                caffe.set_mode_gpu()
            caffe.set_random_seed(777)
            solver = caffe.SGDSolver(solver_prototxt)
            pretrained_model = kwargs.get('pretrained_model', None)
            if pretrained_model:
                solver.net.copy_from(pretrained_model, 
                        ignore_shape_mismatch=True)
            solver.solve()


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

    path_env['labelmap'] = os.path.join(path_env['data'], 'labelmap.txt')

    return path_env

def yolotrain(data, net, **kwargs):
    init_logging()
    c = CaffeWrapper(data, net, **kwargs)

    monitor_train_only = kwargs.get('monitor_train_only', False)

    if not monitor_train_only:
        model = c.train()
        predict_result = c.predict(model)
        c.evaluate(predict_result)
    else:
        c.monitor_train(data, net, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Yolo network')
    parser.add_argument('-g', '--gpus', help='GPU device id to use [0].',  
            default='0')
    parser.add_argument('-n', '--net', required=False, type=str.lower,
            help='only darknet19 is not supported', default='darknet19')
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
    return parser.parse_args()

if __name__ == '__main__':
    '''
    e.g. python scripts/yolotrain.py --data voc20 --iters 10000 --expid 789 \
            --net darknet19 \
            --gpus 4,5,6,7
    '''
    args = parse_args()
    yolotrain(**vars(args))

