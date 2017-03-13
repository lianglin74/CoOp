from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from layerfactory import *
from cnnmodel import *

class CaffeNet(CNNModel):
    def crop_size(self):
        return 227

    def add_body(self, netspec, depth=-1, lr=1, deploy=True):
        n = netspec

        n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.pool1 = max_pool(n.relu1, 3, stride=2)
        n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)

        n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)
        n.pool2 = max_pool(n.relu2, 3, stride=2)
        n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)

        n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)

        n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)

        n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)
        if self.add_last_pooling_layer:
            n.pool5 = max_pool(n.relu5, 3, stride=2)

    def add_extra(self, netspec, num_classes, lr=1, lr_lastlayer=1, deploy=True, add_fc8=True):
        n = netspec

        n.fc6, n.relu6 = fc_relu(last_layer(n), 4096, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)
        if not deploy:
            n.drop6 = L.Dropout(n.relu6, dropout_ratio = 0.5, in_place=True)

        n.fc7, n.relu7 = fc_relu(last_layer(n), 4096, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=1), lr=lr, deploy=deploy)
        if not deploy:
            n.drop7 = L.Dropout(n.relu7, dropout_ratio = 0.5, in_place=True)

        if add_fc8:
            n.fc8 = fc(last_layer(n), num_classes, lr=lr_lastlayer, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), deploy=deploy)
    
    def add_body_for_feature(self, netspec, depth=-1, lr=1, deploy=True):
        self.add_body(netspec, depth=depth, lr=lr, deploy=deploy)

    def add_body_for_roi(self, netspec, bottom, lr=1, deploy=True):
        self.add_extra(netspec, num_classes=0, lr=lr, lr_lastlayer=0, deploy=deploy, add_fc8=False)
