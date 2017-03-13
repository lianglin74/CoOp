import sys
import os, os.path as op
import platform
import argparse
import re

import caffe
import _init_path   # add path to find modelzoo
import modelzoo as mzoo

model_dict = {
    'zf': mzoo.ZFNet(add_last_pooling_layer=False, rcnn_mode=True),
    'zfb': mzoo.ZFBNet(add_last_pooling_layer=False, rcnn_mode=True),
    'vgg': mzoo.VGG(add_last_pooling_layer=False, rcnn_mode=True),
    'resnet': mzoo.ResNet(add_last_pooling_layer=False, rcnn_mode=True),
    'squeezenet': mzoo.SqueezeNet(add_last_pooling_layer=False, rcnn_mode=True),
    }

def gen_net_prototxt(cmd, deploy=False):
    model_parts = re.findall(r'\d+|\D+', cmd.model)
    model_name = model_parts[0].lower()
    model_depth = -1 if len(model_parts) == 1 else int(model_parts[1])
    
    rcnn = mzoo.FasterRCNN()
    model = model_dict[model_name]

    num_classes = cmd.num_classes;

    n = caffe.NetSpec()
    if not deploy:
        rcnn.add_input_data(n, num_classes)
    else:
        # create a placeholder, and replace later
        n.data = caffe.layers.Layer()
        n.im_info = caffe.layers.Layer()

    model.add_body(n, lr=1, depth=model_depth, deploy=deploy)
    rcnn.add_body(n, lr=1, num_classes=num_classes, roi_size=model.roi_size(), deploy=deploy)

    layers = str(n.to_proto()).split('layer {')[1:]
    layers = ['layer {' + x for x in layers]
    if deploy:
        layers[0] = 'input: {}\ninput_shape {{\n  dim: {}\n  dim: {}\n  dim: {}\n  dim: {}\n}}\n'.format('"data"', 1, 3, 224, 224)
        layers[1] = 'input: {}\ninput_shape {{\n  dim: {}\n  dim: {}\n}}\n'.format('"im_info"', 1, 3)
    proto_str = ''.join(layers)
    proto_str = proto_str.replace("\\'", "'")
    
    return 'name: "Faster-RCNN-%s"\n' % cmd.model + proto_str

def sgd_solver_param(model_name):
    solver_param = {
        'base_lr': 0.001,
        'lr_policy': "step",
        'gamma': 0.1,
        'stepsize': 50000,
        'display': 20,
        'average_loss': 100,
        'momentum': 0.9,
        'weight_decay': 0.0005,

        'snapshot': 0,
        'iter_size': 2,
        }
    return solver_param

def gen_solver_prototxt(cmd, train_net_file):

    model_parts = re.findall(r'\d+|\D+', cmd.model)
    model_name = model_parts[0].lower()

    solver_param = sgd_solver_param(model_name)

    # Create solver.
    solver = caffe.proto.caffe_pb2.SolverParameter(
            snapshot_prefix='%s_faster_rcnn' % cmd.model.lower(),
            **solver_param)

    solver_str = 'net: "%s"\n' % train_net_file
    solver_str += str(solver)
    return solver_str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        help='CNN model architecture: zf, zfb, vgg16/19, resnet10/18/34/50/101/152, squeezenet')
    parser.add_argument('--num_classes', required=False, type=int, default='21',
                        help='number of classes, default:21')
    return parser.parse_args()

if __name__ == "__main__":
    cmd = parse_args()
    
    train_net_file = '%s_frcnn_train.prototxt' % cmd.model.lower()
    deploy_net_file = '%s_frcnn_deploy.prototxt' % cmd.model.lower()
    with open(train_net_file, 'w') as f:
        f.write(gen_net_prototxt(cmd, deploy=False))
    with open(deploy_net_file, 'w') as f:
        f.write(gen_net_prototxt(cmd, deploy=True))

    solver_file = '%s_frcnn_solver.prototxt' % cmd.model.lower()
    with open(solver_file, 'w') as f:
        f.write(gen_solver_prototxt(cmd, train_net_file))

    print('Done.')