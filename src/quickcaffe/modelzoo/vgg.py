from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from layerfactory import *
from cnnmodel import *
from caffenet import *

class VGG(CaffeNet):
    def crop_size(self):
        return 224

    def roi_size(self):
        return 7

    def add_body(self, netspec, depth=16, lr=1, deploy=True):
        assert depth in (16, 19), 'only support vgg16 and vgg19'

        n = netspec
        # the parameter fillers are referred to http://m.blog.csdn.net/article/details?id=52723666

        n.conv1_1, n.relu1_1 = conv_relu(n.data, 3, 64, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 3, 64, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.pool1 = max_pool(n.relu1_2, 2, stride=2)

        n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 3, 128, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 3, 128, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.pool2 = max_pool(n.relu2_2, 2, stride=2)

        n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 3, 256, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 3, 256, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 3, 256, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        if depth == 19:
            n.conv3_4, n.relu3_4 = conv_relu(n.relu3_3, 3, 256, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.pool3 = max_pool(last_layer(n), 2, stride=2)

        n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 3, 512, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 3, 512, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 3, 512, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        if depth == 19:
            n.conv4_4, n.relu4_4 = conv_relu(n.relu4_3, 3, 512, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.pool4 = max_pool(last_layer(n), 2, stride=2)

        n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 3, 512, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 3, 512, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 3, 512, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)
        if depth == 19:
            n.conv5_4, n.relu5_4 = conv_relu(n.relu5_3, 3, 512, pad=1, weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0), lr=lr, deploy=deploy)

        if self.add_last_pooling_layer:
            n.pool5 = max_pool(last_layer(n), 2, stride=2)

    def add_extra(self, netspec, num_classes, lr=1, lr_lastlayer=1, deploy=True, add_fc8=True):
        n = netspec

        n.fc6, n.relu6 = fc_relu(last_layer(n), 4096, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0.1), lr=lr, deploy=deploy)
        if not deploy:
            n.drop6 = L.Dropout(n.relu6, dropout_ratio = 0.5, in_place=True)

        n.fc7, n.relu7 = fc_relu(last_layer(n), 4096, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0.1), lr=lr, deploy=deploy)
        if not deploy:
            n.drop7 = L.Dropout(n.relu7, dropout_ratio = 0.5, in_place=True)

        if add_fc8:
            n.fc8 = fc(last_layer(n), num_classes, lr=lr_lastlayer, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0.1), deploy=deploy)
