from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from .layerfactory import *
from .cnnmodel import *

class SqueezeNet(CNNModel):
    def fire_block(self, n, s, nout_squeeze, nout_expand, lr, deploy):
        bottom = last_layer(n)
        n[s+'squeeze1x1'], n[s+'relu_squeeze1x1'] = conv_relu(bottom, ks=1, nout=nout_squeeze,
             weight_filler=dict(type='xavier'), lr = lr, deploy = deploy)

        bottom = last_layer(n)
        n[s+'expand1x1'], n[s+'relu_expand1x1'] = conv_relu(bottom, ks=1, nout=nout_expand,
             weight_filler=dict(type='xavier'), lr = lr, deploy = deploy)

        n[s+'expand3x3'], n[s+'relu_expand3x3'] = conv_relu(bottom, ks=3, pad=1, nout=nout_expand,
             weight_filler=dict(type='xavier'), lr = lr, deploy = deploy)

        n[s+'concat'] = L.Concat(n[s+'relu_expand1x1'], n[s+'relu_expand3x3'])

    def add_body(self, netspec, depth=-1, lr=1, deploy=True):
        n = netspec

        n.conv1, n.relu_conv1 = conv_relu(n.data, 3, 64, stride=2, pad=1, weight_filler=dict(type='xavier'), lr=lr, deploy=deploy)
        n.pool1 = max_pool(last_layer(n), 3, stride=2)

        net_defs = [
            ('fire2', 16),
            ('fire3', 16),
            ('pool3',),
            ('fire4', 32),
            ('fire5', 32),
            ('pool5',),
            ('fire6', 48),
            ('fire7', 48),
            ('fire8', 64),
            ('fire9', 64),
            ('drop9', 0.5),
        ]

        for s in net_defs:
            if s[0].startswith('fire'):
                nout = s[1]
                self.fire_block(n, s[0]+'/', nout, nout*4, lr=lr, deploy=deploy)
            elif s[0].startswith('pool'):
                n[s[0]] = max_pool(last_layer(n), 3, stride=2)
            else:
                n[s[0]] = L.Dropout(last_layer(n), dropout_ratio = s[1], in_place=True)

    def add_extra(self, netspec, num_classes, lr=1, lr_lastlayer=1, deploy=True):
        n = netspec

        n.conv10, n.relu_conv10 = conv_relu(last_layer(n), 1, nout=num_classes, stride=1, 
             weight_filler=dict(type='xavier'), lr=lr, deploy=deploy)
        n.pool10 = ave_pool_global(last_layer(n))
