from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from layerfactory import *
from caffenet import *

class ZFNet(CaffeNet):
    def add_body(self, netspec, depth=-1, lr=1, deploy=True):
        '''
        This model is the same as the one used in faster-rcnn. Training cannot converge even after 20,000 iterations. 
        But it can be used to load pretrained ZF model for faster-rcnn.
        '''
        n = netspec

        n.conv1, n.relu1 = conv_relu(n.data, 7, 96, pad=3, stride=2, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.norm1 = L.LRN(n.relu1, local_size=3, alpha=0.00005, beta=0.75, norm_region=P.LRN.WITHIN_CHANNEL, engine=P.LRN.CAFFE)
        n.pool1 = max_pool(n.norm1, ks=3, stride=2)

        n.conv2, n.relu2 = conv_relu(n.pool1, 5, 256, pad=2, stride=2, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)
        n.norm2 = L.LRN(n.relu2, local_size=3, alpha=0.00005, beta=0.75, norm_region=P.LRN.WITHIN_CHANNEL, engine=P.LRN.CAFFE)
        n.pool2 = max_pool(n.norm2, ks=3, stride=2)

        n.conv3, n.relu3 = conv_relu(n.pool2, 3, 384, pad=1, stride=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)

        n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, stride=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)

        n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, stride=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)

        if self.add_last_pooling_layer:
            n.pool5 = max_pool(n.relu5, 3, stride=2)

    def add_extra(self, netspec, num_classes, lr=1, lr_lastlayer=1, deploy=True, add_fc8=True):
        n = netspec

        n.fc6, n.relu6 = fc_relu(last_layer(n), 4096, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        #n.drop6 = L.Dropout(n.relu6, in_place=True, dropout_param=dict(dropout_ratio=0.5, scale_train=False))

        n.fc7, n.relu7 = fc_relu(last_layer(n), 4096, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        #n.drop7 = L.Dropout(n.relu7, in_place=True, dropout_param=dict(dropout_ratio=0.5, scale_train=False))

        if add_fc8:
            n.fc8 = fc(last_layer(n), num_classes, lr=lr_lastlayer, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), deploy=deploy)

class ZFBNet(CaffeNet):
    def crop_size(self):
        return 224

    def roi_size(self):
        return 7

    def add_body(self, netspec, depth=-1, lr=1, deploy=True):
        '''
        Slight modification from CaffeNet:
        1. Assume input size of 224x224
        2. Add pad=5 to conv1 to get spatial resolution of 112 at stage 2
        3. Remove group=2 in conv2, conv4, and conv5
        Training converges after 2000-3000 iterations
        '''
        n = netspec

        n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, pad=5, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.pool1 = max_pool(n.relu1, 3, stride=2)
        n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

        n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)
        n.pool2 = max_pool(n.relu2, 3, stride=2)
        n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)

        n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)

        n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)

        n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)

        if self.add_last_pooling_layer:
            n.pool5 = max_pool(n.relu5, 3, stride=2)
