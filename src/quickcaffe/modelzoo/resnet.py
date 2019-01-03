from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
from os import path
from .layerfactory import *
from .cnnmodel import *

class ResNet(CNNModel):
    
    def roi_size(self):
        return 14

    def residual_standard_unit(self, n, nout, s, last_stage, newdepth = False, lr = 1, bn_no_train=False, deploy=True):
        """
        This creates the "standard unit" shown on the left side of Figure 5.
        """
        bottom = n.__dict__['tops'][n.__dict__['tops'].keys()[-1]] #find the last layer in netspec
        stride = 2 if newdepth and not (self.rcnn_mode and last_stage) else 1

        n[s + 'conv1'], n[s + 'bn1'], n[s + 'scale1'] = conv_bn(bottom, ks = 3, stride = stride, nout = nout, pad = 1, lr=lr,
                                                                bn_no_train=bn_no_train, deploy=deploy)
        n[s + 'relu1'] = L.ReLU(n[s + 'scale1'], in_place=True)
        n[s + 'conv2'], n[s + 'bn2'], n[s + 'scale2'] = conv_bn(n[s + 'relu1'], ks = 3, stride = 1, nout = nout, pad = 1, lr=lr,
                                                                bn_no_train=bn_no_train, deploy=deploy)

        s_ = s[:-1]
        if newdepth:
            n[s + 'conv_expand'], n[s + 'bn_expand'], n[s + 'scale_expand'] = conv_bn(bottom, ks = 1, stride = stride, nout = nout, pad = 0, lr=lr,
                                                                                      bn_no_train=bn_no_train, deploy=deploy)
            n[s_] = L.Eltwise(n[s + 'scale2'], n[s + 'scale_expand'])
        else:
            n[s_] = L.Eltwise(n[s + 'scale2'], bottom)

        n[s + 'relu'] = L.ReLU(n[s_], in_place=True)
        return s_

    def residual_bottleneck_unit(self, n, nout, s, last_stage, newdepth = False, lr = 1, bn_no_train=False, deploy=True):
        """
        This creates the "standard unit" shown on the left side of Figure 5.
        """
        newnnout = nout * 4
        bottom = n.__dict__['tops'].keys()[-1] #find the last layer in netspec
        stride = 2 if newdepth and nout > 64 and not (self.rcnn_mode and last_stage) else 1

        n[s + 'conv1'], n[s + 'bn1'], n[s + 'scale1'] = conv_bn(n[bottom], ks = 1, stride = stride, nout = nout, pad = 0, lr=lr,
                                                                bn_no_train=bn_no_train, deploy=deploy)
        n[s + 'relu1'] = L.ReLU(n[s + 'scale1'], in_place=True)
        n[s + 'conv2'], n[s + 'bn2'], n[s + 'scale2'] = conv_bn(n[s + 'relu1'], ks = 3, stride = 1, nout = nout, pad = 1, lr=lr,
                                                                bn_no_train=bn_no_train, deploy=deploy)
        n[s + 'relu2'] = L.ReLU(n[s + 'scale2'], in_place=True)
        n[s + 'conv3'], n[s + 'bn3'], n[s + 'scale3'] = conv_bn(n[s + 'relu2'], ks = 1, stride = 1, nout = newnnout, pad = 0, lr=lr,
                                                                bn_no_train=bn_no_train, deploy=deploy)

        s_ = s[:-1]
        if newdepth:
            n[s + 'conv_expand'], n[s + 'bn_expand'], n[s + 'scale_expand'] = conv_bn(n[bottom], ks = 1, stride = stride, nout = newnnout, pad = 0, lr=lr,
                                                                                      bn_no_train=bn_no_train, deploy=deploy)
            n[s_] = L.Eltwise(n[s + 'scale3'], n[s + 'scale_expand'])
        else:
            n[s_] = L.Eltwise(n[s + 'scale3'], n[bottom])

        n[s + 'relu'] = L.ReLU(n[s_], in_place=True)
        return s_

    def add_stem(self, netspec, lr=1, bn_no_train=False, deploy=True):
        n = netspec

        n.conv1, n.bn_conv1, n.scale_conv1 = conv_bn(n.data, ks = 7, stride = 2, nout = 64, pad = 3, lr=lr,
                                                     bn_no_train=bn_no_train, deploy=deploy)
        n.conv1_relu = L.ReLU(n.scale_conv1, in_place=True)
        n.pool1 = L.Pooling(n.conv1_relu, stride = 2, kernel_size = 3)

    def add_stages(self, netspec, stage_range, depth=18, lr=1, bn_no_train=False, deploy=True):
        # figure out network structure
        net_defs = {
            10:([1, 1, 1, 1], "standard"),
            18:([2, 2, 2, 2], "standard"),
            34:([3, 4, 6, 3], "standard"),
            50:([3, 4, 6, 3], "bottleneck"),
            101:([3, 4, 23, 3], "bottleneck"),
            152:([3, 8, 36, 3], "bottleneck"),
        }
        assert depth in net_defs.keys(), "net of depth:{} not defined".format(depth)

        nunits_list, unit_type = net_defs[depth] # nunits_list a list of integers indicating the number of layers in each depth.
        nouts = [64, 128, 256, 512] # same for all nets
        strides = [4,8,16,32]   # stride of each stages
        n = netspec
        self.stages=[]
        # make the convolutional body
        for s in stage_range:
            # s is 2-based, assuming stage 1 is the stem part
            stage = s - 2
            nout = nouts[stage]
            nunits = nunits_list[stage]
            for unit in range(1, nunits + 1): # for each unit. Enumerate from 1.
                #s = str(nout) + '_' + str(unit) + '_' # layer name prefix
                if unit > 1 and nunits > 6:
                    s = 'res' + str(stage+2) + 'b' + str(unit-1) + '_' # layer name prefix
                else:
                    s = 'res' + str(stage+2) + chr(ord('a')+unit-1) + '_' # layer name prefix
                last_stage = True if stage == len(nouts) - 1 else False
                if unit_type == "standard":
                    stage_name = self.residual_standard_unit(n, nout, s, last_stage, newdepth = unit is 1 and nout > 64, lr=lr, bn_no_train=bn_no_train, deploy=deploy)
                else:
                    stage_name = self.residual_bottleneck_unit(n, nout, s, last_stage, newdepth = unit is 1, lr=lr, bn_no_train=bn_no_train, deploy=deploy)
                if unit== nunits:
                    self.stages.append((stage_name,nouts[stage], strides[stage]))

    def add_body(self, netspec, depth=18, lr=1, deploy=True, **kwargs):
        """
        Generates nets from "Deep Residual Learning for Image Recognition". Nets follow architectures outlined in Table 1.
        """
        self.add_stem(netspec, lr=lr, deploy=deploy)
        self.add_stages(netspec, stage_range=[2,3,4,5], depth=depth, deploy=deploy)

    def add_extra(self, netspec, num_classes, lr=1, lr_lastlayer=1, deploy=True):
        n = netspec
        n.pool5 = ave_pool_global(last_layer(n))
        _fc = 'fc_'+str(num_classes)
        n[_fc] = fc(last_layer(n), num_classes, lr=lr_lastlayer, deploy=deploy)

    def add_body_for_feature(self, netspec, depth=-1, lr=1, deploy=True):
        self.model_depth = depth
        self.add_stem(netspec, lr=lr, bn_no_train=True, deploy=deploy)
        self.add_stages(netspec, stage_range=[2,3,4], depth=depth, lr=lr, bn_no_train=True, deploy=deploy)

    def add_body_for_roi(self, netspec, bottom, lr=1, deploy=True):
        self.add_stages(netspec, stage_range=[5], depth=self.model_depth, lr=lr, bn_no_train=True, deploy=deploy)
        netspec.pool5 = ave_pool_global(last_layer(netspec))
    
    def get_stages(self):
        return self.stages