from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from layerfactory import *
from cnnmodel import *

class DarkNet(CNNModel):
    def roi_size(self):
        return 7
    def get_netdefs(self):
        net_defs8 = [
            ('dark1', 16),
            ('pool1',),
            ('dark2', 32),
            ('pool2',),
            ('dark3', 64),
            ('pool3',),
            ('dark4', 128),
            ('pool4',),
            ('dark5', 256),            
            ('pool5',),     
            ('dark6', 512),            
            ('pool6_1',),
            ('dark7', 1024),            
        ]        
        net_defs16 = [
            ('dark1', 16),
            ('pool1',),
            ('dark2', 32),
            ('pool2',),
            ('dark3a_1', 16),
            ('dark3b', 128),
            ('dark3c_1', 16),
            ('dark3d', 128),
            ('pool3',),
            ('dark4a_1', 32),
            ('dark4b', 256),
            ('dark4c_1', 32),
            ('dark4d', 256),
            ('pool4',),
            ('dark5a_1', 64),            
            ('dark5b', 512),
            ('dark5c_1', 64),
            ('dark5d', 512),
            ('dark5e_1', 128),
        ]
        net_defs19 = [
            ('dark1', 32),
            ('pool1',),
            ('dark2', 64),
            ('pool2',),
            ('dark3a', 128),
            ('dark3b_1', 64),
            ('dark3c', 128),
            ('pool3',),
            ('dark4a', 256),
            ('dark4b_1', 128),
            ('dark4c', 256),
            ('pool4',),
            ('dark5a', 512),
            ('dark5b_1', 256),
            ('dark5c', 512),
            ('dark5d_1', 256),
            ('dark5e', 512),
            ('pool5',), 
            ('dark6a', 1024),
            ('dark6b_1', 512),
            ('dark6c', 1024),
            ('dark6d_1', 512),
            ('dark6e', 1024),
        ]
        return {8:net_defs8,16:net_defs16, 19:net_defs19};
    
    def dark_block(self, n, s, nout_dark,stride,  lr, deploy, bn_no_train=False):
        bottom = n.data if s.startswith('dark1') else last_layer(n);
        ks = 1 if s.endswith('_1') else 3
        pad = 0 if ks==1 else 1
        n[s+'/conv'], n[s+'/bn'], n[s+'/scale'] = conv_bn(bottom, ks=ks, stride=stride, pad=pad, bn_no_train=bn_no_train, nout=nout_dark, lr = lr, deploy = deploy)
        n[s+'/leaky'] = L.ReLU(n[s+'/scale'], negative_slope=0.1)
        
    def add_body(self, netspec, depth=-1, lr=1, deploy=True):
        n = netspec
        net_defs = self.get_netdefs();
        assert depth in net_defs.keys(), 'darknet only support depths' + str(net_defs.keys())            
        for s in net_defs[depth]:
            if s[0].startswith('dark'):
                nout = s[1]
                stride = 1 if len(s)==2 else s[2];
                self.dark_block(n, s[0], nout, stride, lr=lr, deploy=deploy)
            elif  s[0].startswith('pool'):
                if s[0].endswith('_1'):
                    n[s[0]] = max_pool(last_layer(n), 2, stride=2, pad=1);
                else:
                    n[s[0]] = max_pool(last_layer(n), 2, stride=2);
                    
    def add_extra(self, netspec, num_classes, lr=1, lr_lastlayer=1, deploy=True):
        n = netspec
        n.pool6 = ave_pool_global(last_layer(n))
        n.fc7 = fc(n.pool6, nout=num_classes,  deploy=deploy, weight_filler=dict(type='xavier'))
        
    def add_body_for_feature(self, netspec, depth=-1, lr=1, deploy=True):
        self.model_depth = depth
        n = netspec
        net_defs = self.get_netdefs();
        assert depth in net_defs.keys(), 'darknet only support depths' + str(net_defs.keys())            
        for s in net_defs[depth]:
            if s[0]=='dark6a':  break;
            if s[0].startswith('dark'):
                nout = s[1]
                stride = 1 if len(s)==2 else s[2];
                self.dark_block(n, s[0], nout, stride, lr=lr, deploy=deploy, bn_no_train=True)
            elif  s[0].startswith('pool'):
                if s[0].endswith('_1'):
                    n[s[0]] = max_pool(last_layer(n), 2, stride=2, pad=1);
                else:
                    n[s[0]] = max_pool(last_layer(n), 2, stride=2);
    def add_body_for_roi(self, netspec, bottom, lr=1, deploy=True):
        n = netspec
        net_defs = self.get_netdefs();
        assert self.model_depth in net_defs.keys(), 'darknet only support depths' + str(net_defs.keys())            
        for s in net_defs[self.model_depth]:
            if not (s[0].startswith('dark6')): 
                continue;
            else:
                nout = s[1]
                stride = 1 if len(s)==2 else s[2];
                self.dark_block(n, s[0], nout, stride, lr=lr, deploy=deploy)
        n.pool6 = ave_pool_global(last_layer(n))                

