import sys
import os, os.path as op
import platform
import argparse
import re

import caffe
import _init_path   # add path to find modelzoo
import modelzoo as mzoo

model_dict = {
    'caffenet': mzoo.CaffeNet(),
    'zf': mzoo.ZFNet(),
    'zfb': mzoo.ZFBNet(),
    'vgg': mzoo.VGG(),
    'googlenet': mzoo.GoogleNet(),
    'resnet': mzoo.ResNet(),
    'resnetrcnn': mzoo.ResNet(rcnn_mode=True),
    'squeezenet': mzoo.SqueezeNet(),
    }

def gen_data_layer(data_path, format, crop_size, batchsize=(256,50), use_mean_file=True):
    mean_value = [104, 117, 123] if not use_mean_file else []
    meanfile_dir = op.split(data_path)[0]
    mean_file = op.join(meanfile_dir, 'imagenet_mean.binaryproto') if use_mean_file else ''
    
    data_path = data_path.replace('\\', '/')
    mean_file = mean_file.replace('\\', '/') # to work for both windows and linux

    batchsize_train, batchsize_test = batchsize

    if format == 'tsv': # use tsv data layer
        data, label = mzoo.tsv_imagenet(data_path, mean_file, mean_value, crop_size, batchsize_train, (256, 256), mirror=True, phase='TRAIN')
        test_str = mzoo.tsv_imagenet_str(data_path.replace('train', 'val'), mean_file, mean_value, crop_size, batchsize_test, (256, 256), mirror=False, phase='TEST')
    else:  # use leveldb or lmdb data layer
        data, label = mzoo.lmdb_imagenet(data_path, format, mean_file, mean_value, crop_size, batchsize_train, mirror=True, phase='TRAIN')
        test_str = mzoo.lmdb_imagenet_str(data_path.replace('train', 'val'), format, mean_file, mean_value, crop_size, batchsize_test, mirror=False, phase='TEST')
    
    return data, label, test_str

def to_proto_str(n, test_str='', data_str='', deploy=False):
    layers = str(n.to_proto()).split('layer {')[1:]
    layers = ['layer {' + x for x in layers]

    if not deploy:
        # insert the TEST phase data layer
        layers.insert(1, test_str)
    else:
        #generate the input information header string
        layers[0] = data_str
    return ''.join(layers)

def gen_net_prototxt(cmd, deploy=False):
    model_parts = re.findall(r'\d+|\D+', cmd.model)
    model_name = model_parts[0].lower()
    model_depth = -1 if len(model_parts) == 1 else int(model_parts[1])
    
    batchsize = [int(x) for x in cmd.batchsize.split(',')][:2]
    num_classes = cmd.num_classes;

    n = caffe.NetSpec()

    model = model_dict[model_name]

    crop_size = model.crop_size()

    # add data
    if not deploy:
        use_mean_file = cmd.use_meanfile
        n.data, n.label, test_str = gen_data_layer(cmd.data, cmd.format, crop_size, batchsize, use_mean_file=use_mean_file)
    else:
        n.data = caffe.layers.Layer()  # create a placeholder, and replace later
        data_str = 'input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\n'.format('"data"', 1, 3, crop_size, crop_size)

    # add body and extra
    model.add_body(n, lr=1, depth=model_depth, deploy=deploy)
    model.add_extra(n, num_classes, lr=1, lr_lastlayer=1, deploy=deploy)
    
    # add loss
    model.add_loss_or_prediction(n, deploy, accuracy_top5=True)
    if model_name == 'googlenet' and not deploy:
        model.add_extra_loss(n, num_classes, accuracy_top5=True)

    if not deploy:
        proto_str = to_proto_str(n, test_str=test_str, deploy=deploy)
    else:
        proto_str = to_proto_str(n, data_str=data_str, deploy=deploy)

    return 'name: "%s"\n' % cmd.model + proto_str

def sgd_solver_param(model_name, test_iter):
    solver_lr = {
        'caffenet': (0.01, 0.0005),
        'zf': (0.01, 0.0005),
        'zfb': (0.01, 0.0005),
        'vgg': (0.01, 0.0005),
        'resnet': (0.1, 0.0001),
        'resnetrcnn': (0.1, 0.0001),
    }
    base_lr, weight_decay = solver_lr[model_name]

    solver_param = {
        # Test parameters
        'test_iter': [test_iter],
        'test_interval': 1000,
        #'test_initialization': False,
        # Train parameters
        'base_lr': base_lr,
        'lr_policy': "step",
        'gamma': 0.1,
        'stepsize': 100000,
        'display': 20,
        'max_iter': 450000,
        'momentum': 0.9,
        'weight_decay': weight_decay,
        'snapshot': 5000,
        'solver_mode': caffe.params.Solver.GPU,
        }
    return solver_param

def quick_solver_param(model_name, test_iter):
    solver_param = {
        # Test parameters
        'test_iter': [test_iter],
        'test_interval': 4000,
        #'test_initialization': False,
        # Train parameters
        'base_lr': 0.01,
        'lr_policy': "poly",
        'power': 0.5,
        'display': 40,
        'average_loss': 40,
        'max_iter': 2400000,
        'momentum': 0.9,
        'weight_decay': 0.0002,
        'snapshot': 40000,
        'solver_mode': caffe.params.Solver.GPU,
        }
    return solver_param
            
def quick_solver_squeeze_param(model_name, test_iter):
    solver_param = {
        # Test parameters
        'test_iter': [test_iter],
        'test_interval': 1000,
        #'test_initialization': False,
        # Train parameters
        'base_lr': 0.04,
        'lr_policy': "poly",
        'power': 1.0, # linearly decrease LR
        'display': 40,
        'average_loss': 40,
        'max_iter': 170000,
        'momentum': 0.9,
        'weight_decay': 0.0002,
        'snapshot': 5000,
        'solver_mode': caffe.params.Solver.GPU,
        }
    return solver_param

def gen_solver_prototxt(cmd, train_net_file, test_iter):

    model_parts = re.findall(r'\d+|\D+', cmd.model)
    model_name = model_parts[0].lower()

    if model_name == 'googlenet':
        solver_param = quick_solver_param(model_name, test_iter)
    elif model_name == 'squeezenet':
        solver_param = quick_solver_squeeze_param(model_name, test_iter)
    else:
        solver_param = sgd_solver_param(model_name, test_iter)

    # Create solver.
    solver = caffe.proto.caffe_pb2.SolverParameter(
            snapshot_prefix='models/%s' % cmd.model,
            **solver_param)

    solver_str = 'net: "%s"\n' % train_net_file
    solver_str += str(solver)
    return solver_str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, 
                        help='CNN model architecture: caffenet, zf, zfb, vgg16/19, resnet10/18/34/50/101/152, squeezenet')
    parser.add_argument('--data', required=True,
                        help='ImageNet training data path. If the path includes "lmdb", lmdb data layer will be used.')
    parser.add_argument('--format', required=False, default='tsv', choices=['tsv', 'leveldb', 'lmdb'],
                        help='data format. choose one from tsv, leveldb, lmdb')
    parser.add_argument('--batchsize', required=False, default='256,50',
                        help='batch size for training and testing')
    parser.add_argument('--base_lr', required=False, type=float, default='0.01',
                        help='base learning rate, default:0.01')
    parser.add_argument('--num_classes', required=False, type=int, default='1000',
                        help='number of classes, default:1000')
    parser.add_argument('--use_meanfile', default=False, action='store_true',
                        help='Flag to use meanfile, default: False')
    return parser.parse_args()

if __name__ == "__main__":
    cmd = parse_args()
    
    train_net_file = '%s_train_val.prototxt' % cmd.model.lower()
    deploy_net_file = '%s_deploy.prototxt' % cmd.model.lower()
    with open(train_net_file, 'w') as f:
        f.write(gen_net_prototxt(cmd, deploy=False))
    with open(deploy_net_file, 'w') as f:
        f.write(gen_net_prototxt(cmd, deploy=True))

    batchsize = [int(x) for x in cmd.batchsize.split(',')][:2]
    test_iter = 50000 / batchsize[1]
    solver_file = '%s_solver.prototxt' % cmd.model.lower()
    with open(solver_file, 'w') as f:
        f.write(gen_solver_prototxt(cmd, train_net_file, test_iter))

    if not op.exists('models'):
        os.mkdir('models')

    # launch training
    if platform.system() == 'Windows':
        print 'windows run'
    else: # Linux
        print 'linux run'
