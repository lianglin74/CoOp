import _init_paths
import os
import gen_prototxt
import caffe
import re
import quickcaffe.modelzoo as mzoo
from caffe.proto import caffe_pb2
import numpy as np
from google.protobuf import text_format

from tsvdet import setup_paths
from deteval import deteval
from yolodet import tsvdet

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_to_file(contxt, file_name):
    p = os.path.dirname(file_name)
    ensure_directory(p)
    with open(file_name, 'w') as fp:
        fp.write(contxt)

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
        det.add_body(n, lr=1, num_classes=num_classes, cnnmodel=model, deploy=deploy, cpp_version=cpp_version)

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
        solver_param = {
                'train_net': train_net_file, 
                'lr_policy': 'multifixed',
                'gamma': 0.1,
                'stepsize': 50000,
                'display': 1,
                'momentum': 0.9,
                'weight_decay': 0.0005,
                'snapshot': 2000,
                'snapshot_prefix': snapshot_prefix,
                'iter_size': 10,
                'max_iter': kwargs.get('max_iters', 10000),
                'stagelr': [0.0001, 0.001, 0.0001, 0.00001],
                'stageiter': [100, 5000, 9000, 10000000]
                }

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
    def train(self, data, net, **kwargs):
        expid = kwargs.get('expid', '777')
        path_env = default_paths(net, data, expid)

        if 'num_classes' not in kwargs:
            num_classes = num_non_empty_lines(path_env['labelmap'])
        else:
            num_classes = kwargs['num_classes']
        
        source = path_env['source']
        labelmap = path_env['labelmap'] 

        p = ProtoGenerator()
        p.generate_prototxt(net, num_classes, 
                 path_env['train_proto_file'], path_env['test_proto_file'],
                 path_env['solver'], detmodel='Yolo', 
                 source=source, labelmap=labelmap, **kwargs)

        #self._train(path_env['solver'], pretrained_model=path_env['basemodel'], **kwargs)

        solver_param = self._load_solver(path_env['solver'])
        last_model = '{0}_iter_{1}.caffemodel'.format(
                solver_param.snapshot_prefix, solver_param.max_iter)
        train_net_param = self._load_net(path_env['train_proto_file'])
        data_layer = train_net_param.layer[0]
        mean_value = train_net_param.layer[0].transform_param.mean_value
        model = (path_env['test_proto_file'], last_model, mean_value)

        return model

    def predict(self, data, model, **kwargs):
        test_proto_file, model_param, mean_value = model
        path_env = default_data_path(data)
        colkey = 0
        colimg = 2
        outtsv_file = model_param + '.eval'

        gpu = kwargs.get('gpu', 0)
        if gpu<0:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(gpu)
        tsvdet(test_proto_file, 
                model_param, 
                path_env['test_source'], 
                colkey, 
                colimg, 
                mean_value, 
                outtsv_file)
        return outtsv_file

    def evaluate(self, data, predict_result, **kwargs):
        path_env = default_data_path(data)

        deteval(truth=path_env['test_source'],
                dets=predict_result)

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
        gpu = kwargs.get('gpu', 0)
        if gpu >= 0:
            caffe.set_mode_gpu()
            caffe.set_device(gpu)

        caffe.set_random_seed(777)
    
        if kwargs.get('chdir', False):
            os.chdir(os.path.dirname(solver_prototxt))
    
        solver = caffe.SGDSolver(solver_prototxt)

        pretrained_model = kwargs.get('pretrained_model', None)

        if pretrained_model:
            solver.net.copy_from(pretrained_model)
    
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

def default_data_path(dataset):
    proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)));
    result = {}
    data_root = os.path.join(proj_root, 'data', dataset)
    result['source'] = os.path.join(data_root, 'train.tsv')
    result['test_source'] = os.path.join(data_root, 'test.tsv')
    result['labelmap'] = os.path.join(data_root, 'labelmap.txt')
    return result

def default_paths(net, data, expid):
    path_env = setup_paths(net, data, expid)

    output_path = path_env['output'] 

    train_file_base = 'train.prototxt'
    test_file_base = 'test.prototxt'
    solver_file_base = 'solver.prototxt'

    path_env['train_proto_file'] = os.path.join(output_path, train_file_base)
    path_env['test_proto_file'] = os.path.join(output_path, test_file_base)
    path_env['solver'] = os.path.join(output_path, solver_file_base)

    path_env['source'] = os.path.join(path_env['data'], 'train.tsv')
    path_env['test_source'] = os.path.join(path_env['data'], 'test.tsv')

    path_env['labelmap'] = os.path.join(path_env['data'], 'labelmap.txt')

    return path_env

def yolotrain(data, net, expid, **kwargs):
    c = CaffeWrapper()

    model = c.train(data, net, **kwargs)
    p = c.predict(data, model, **kwargs)

    c.evaluate(data, p, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Yolo network')
    parser.add_argument('-g', '--gpu', dest='GPU_ID', help='GPU device id to use [0].',  default=0, type=int)
    #parser.add_argument('-n', '--net', required=True, type=str.lower, help='CNN archiutecture')
    parser.add_argument('-t', '--iters', dest='max_iters',  help='number of
            iterations to train', default=10000, required=False, type=int)
    parser.add_argument('-d', '--data', help='the name of the dataset', required=True)
    parser.add_argument('-e', '--expid', help='the experiment id', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    '''
    e.g. python scripts/yolotrain.py --data voc20 --iters 10000 --expid 789 \
            --gpu 1
    '''
    args = parse_args()
    #yolotrain(data='voc20', net='darknet19', max_iters=10000, expid='777', gpu=1,
            #prefetch=4)
    yolotrain(args.data, net='darknet19', max_iters=args.iters,
            expid=args.expid, gpu=args.gpu)

