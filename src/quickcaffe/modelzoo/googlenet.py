from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
from os import path
from .layerfactory import *
from .cnnmodel import *

class GoogleNet(CNNModel):
    def __init__(self, last_pooling_layer=True, rcnn_mode = False):
        assert rcnn_mode == False, 'rcnn_mode not supported in GoogleNet'

    def inception_block(self, n, s, defs, lr, deploy):
        bottom = last_layer(n)
        n[s+'1x1'], n[s+'relu_1x1'] = conv_relu(bottom, ks = 1, nout = defs[0],
             weight_filler=dict(type='xavier', std=0.03), bias_filler=dict(type='constant', value=0.2), lr = lr, deploy = deploy)
    
        n[s+'3x3_reduce'], n[s+'relu_3x3_reduce'] = conv_relu(bottom, ks = 1, nout = defs[1][0],
             weight_filler=dict(type='xavier', std=0.09), bias_filler=dict(type='constant', value=0.2), lr = lr, deploy = deploy)
        n[s+'3x3'], n[s+'relu_3x3'] = conv_relu(last_layer(n), ks = 3, nout = defs[1][1], pad = 1,
             weight_filler=dict(type='xavier', std=0.03), bias_filler=dict(type='constant', value=0.2), lr = lr, deploy = deploy)

        n[s+'5x5_reduce'], n[s+'relu_5x5_reduce'] = conv_relu(bottom, ks = 1, nout = defs[2][0],
             weight_filler=dict(type='xavier', std=0.2), bias_filler=dict(type='constant', value=0.2), lr = lr, deploy = deploy)
        n[s+'5x5'], n[s+'relu_5x5'] = conv_relu(last_layer(n), ks = 5, nout = defs[2][1], pad = 2,
             weight_filler=dict(type='xavier', std=0.03), bias_filler=dict(type='constant', value=0.2), lr = lr, deploy = deploy)

        n[s+'pool'] = max_pool(bottom, ks = 3, stride = 1, pad = 1)
        n[s+'pool_proj'], n[s+'relu_pool_proj'] = conv_relu(last_layer(n), ks = 1, nout = defs[3],
             weight_filler=dict(type='xavier', std=0.1), bias_filler=dict(type='constant', value=0.2), lr = lr, deploy = deploy)

        n[s+'output'] = L.Concat(n[s+'relu_1x1'], n[s+'relu_3x3'], n[s+'relu_5x5'], n[s+'relu_pool_proj'])

        return last_layer(n)

    def add_body(self, netspec, depth=-1, lr=1, deploy=True):
        """
        Generates GoogleNet from "Going Deeper with Convolutions". 
        """
        n = netspec

        # stem
        n['conv1/7x7_s2'], n['conv1/relu_7x7'] = conv_relu(n.data, ks = 7, nout = 64, stride = 2, pad = 3, 
             weight_filler=dict(type='xavier', std=0.1), bias_filler=dict(type='constant', value=0.2), lr = lr, deploy = deploy)
        n['pool1/3x3_s2'] = max_pool(last_layer(n), ks = 3, stride = 2)
        n['pool1/norm1'] = L.LRN(last_layer(n), local_size=5, alpha=1e-4, beta=0.75)

        n['conv2/3x3_reduce'], n['conv2/relu_3x3_reduce'] = conv_relu(last_layer(n), ks = 1, nout = 64,
             weight_filler=dict(type='xavier', std=0.1), bias_filler=dict(type='constant', value=0.2), lr = lr, deploy = deploy)
        n['conv2/3x3'], n['conv2/relu_3x3'] = conv_relu(last_layer(n), ks = 3, nout = 192, pad = 1,
             weight_filler=dict(type='xavier', std=0.03), bias_filler=dict(type='constant', value=0.2), lr = lr, deploy = deploy)
        n['conv2/norm2'] = L.LRN(last_layer(n), local_size=5, alpha=1e-4, beta=0.75)
        n['pool2/3x3_s2'] = max_pool(last_layer(n), ks = 3, stride = 2)

        net_stage_units = [-1, -1, 2, 5, 2]
        net_defs = {
            '3a': (64, [96, 128], [16,32], 32),
            '3b': (128, [128, 192], [32, 96], 64),
            '4a': (192, [96, 208], [16, 48], 64),
            '4b': (160, [112, 224], [24, 64], 64),
            '4c': (128, [128, 256], [24, 64], 64),
            '4d': (112, [144, 288], [32, 64], 64),
            '4e': (256, [160, 320], [32, 128], 128),
            '5a': (256, [160, 320], [32, 128], 128),
            '5b': (384, [192, 384], [48, 128], 128),
        }
    
        for stage in (3, 4, 5):
            n_units = net_stage_units[stage-1]
            for unit in range(n_units):
                _s = str(stage) + chr(ord('a')+unit)
                s = 'inception_' + _s + '/' # layer name prefix
                self.inception_block(n, s, net_defs[_s], lr, deploy)
                if stage in (3, 4) and unit == n_units - 1:
                    n['pool' + str(stage) + '/3x3_s2'] = max_pool(last_layer(n), ks = 3, stride = 2)

    def add_extra(self, netspec, num_classes, lr=1, lr_lastlayer=1, deploy=True):
        n = netspec
        n['pool5/7x7_s1'] = ave_pool_global(last_layer(n))
        if not deploy:
            n['pool5/drop_7x7_s1'] = L.Dropout(last_layer(n), dropout_ratio = 0.4, in_place=True)

        n['loss3/classifier'] = fc(last_layer(n), num_classes,
             weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0), lr=lr_lastlayer, deploy=deploy)

    def add_loss(self, netspec, accuracy_top5=False):
        n = netspec
        bottom = n['loss3/classifier']
        n['loss3/loss3'] = L.SoftmaxWithLoss(bottom, n.label, loss_weight=1)
        n['loss3/top-1'] = L.Accuracy(bottom, n.label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
        if accuracy_top5:
            n['loss3/top-5'] = L.Accuracy(bottom, n.label, include=dict(phase=getattr(caffe_pb2, 'TEST')), accuracy_param=dict(top_k=5))

    def add_extra_loss(self, netspec, num_classes, accuracy_top5=False):
        n = netspec
        lr = 1
        lr_lastlayer = 1
        for _s, _bottom in (('loss1/', 'inception_4a/output'), ('loss2/', 'inception_4d/output')):
            bottom = n[_bottom]
            n[_s+'ave_pool'] = ave_pool(bottom, ks=5, stride=3)
            n[_s+'conv'], n[_s+'relu_conv'] = conv_relu(last_layer(n), nout=128, ks=1, 
                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0.2), lr=lr, deploy=False)
            n[_s+'fc'], n[_s+'relu_fc'] = fc_relu(last_layer(n), 1024,
                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0.2), lr=lr, deploy=False)
            n[_s+'drop'] = L.Dropout(last_layer(n), dropout_ratio = 0.7, in_place=True)
            n[_s+'classifier'] = fc(last_layer(n), num_classes,
                    weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0), lr=lr_lastlayer, deploy=False)

            bottom = last_layer(n)
            n[_s+_s[:-1]] = L.SoftmaxWithLoss(bottom, n.label, loss_weight=0.3)
            n[_s+'top-1'] = L.Accuracy(bottom, n.label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
            if accuracy_top5:
                n[_s+'top-5'] = L.Accuracy(bottom, n.label, include=dict(phase=getattr(caffe_pb2, 'TEST')), accuracy_param=dict(top_k=5))
