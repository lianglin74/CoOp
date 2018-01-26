from collections import OrderedDict, Counter
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from contextlib import contextmanager
import caffe
import os.path as op
from quickercaffe import NeuralNetwork,saveproto

class VggNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def get_topname(self, layertype, prefix, layername, postfix):
        prefix =  self.prefix if prefix is None and self.prefix is not None else prefix
        layername = layertype if layername is None else layername
        name = ''.join([layername,postfix]) if postfix is not None else layername
        if name=='' and prefix is not None:     return prefix;
        return '_'.join([prefix,name]) if prefix is not None else name
    def gen_net(self,nclass,depth=16,deploy=False):
        netdefs ={16: [(64,2),(128,2),(256,3),(512,3),(512,3)], 19: [(64,2),(128,2),(256,4),(512,4),(512,4)] }
        assert (depth in netdefs)
        for i,stagedef in enumerate(netdefs[depth]):
            nout = stagedef[0];
            for j in range(stagedef[1]):
                postfix=str(i+1)+'_'+str(j+1)
                self.convrelu(nout,3,pad=1,postfix=postfix)
            self.maxpool(2,stride=2,layername='pool'+str(i+1))
        self.fcrelu(4096,postfix='6')
        self.dropout(deploy=deploy,postfix='6')
        self.fcrelu(4096,postfix='7')
        self.dropout(deploy=deploy,postfix='7')
        self.fc(nclass,bias=True,postfix='8')
        if deploy==False:
            self.set_conv_params( weight_filler = dict(type='gaussian', std=0.01), blacklist=['fc6','fc7','fc8'] )
            self.set_conv_params( weight_filler = dict(type='gaussian', std=0.005), bias_filler= dict(type='constant', value=0.1), whitelist=['fc6','fc7','fc8'] )

class CaffeNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def get_topname(self, layertype, prefix, layername, postfix):
        prefix =  self.prefix if prefix is None and self.prefix is not None else prefix
        layername = layertype if layername is None else layername
        name = ''.join([layername,postfix]) if postfix is not None else layername
        if name=='' and prefix is not None:     return prefix;
        return '_'.join([prefix,name]) if prefix is not None else name
    def gen_net(self,nclass,deploy=False):
        self.convrelu(96,11,stride=4,postfix='1')
        self.maxpool(3,layername='pool1',stride=2)
        self.lrn(postfix='1',layername='norm')
        self.convrelu(256,5,pad=2,group=2,postfix='2')
        self.maxpool(3,stride=2,layername='pool2')
        self.lrn(postfix='2',layername='norm')
        self.convrelu(384,3,pad=1,postfix='3')
        self.convrelu(384,3,pad=1,group=2,postfix='4')
        self.convrelu(256,3,pad=1,group=2,postfix='5')
        self.maxpool(3,stride=2,layername='pool5')
        self.fcrelu(4096,postfix='6')
        self.dropout(deploy=deploy,postfix='6')
        self.fcrelu(4096,postfix='7')
        self.dropout(deploy=deploy,postfix='7')
        self.fc(nclass,bias=True,postfix='8')
        if deploy==False:
            self.set_conv_params( weight_filler = dict(type='gaussian', std=0.01), whitelist=['conv1','conv3','fc8'] )
            self.set_conv_params( weight_filler = dict(type='gaussian', std=0.01), bias_filler= dict(type='constant', value=1), whitelist=['conv2','conv4','conv5'] )
            self.set_conv_params( weight_filler = dict(type='gaussian', std=0.005), bias_filler= dict(type='constant', value=1), whitelist=['fc6','fc7'] )

class ResNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def get_topname(self, layertype, prefix, layername, postfix):
        prefix =  self.prefix if prefix is None and self.prefix is not None else prefix
        layername = layertype if layername is None else layername
        name = ''.join([layername,postfix]) if postfix is not None else layername
        if name=='' and prefix is not None:     return prefix;
        return '_'.join([prefix,name]) if prefix is not None else name
    def standard(self, nin, nout, s, stride ):
        bottom = self.bottom
        with self.scope(s):
            self.conv(nout,3, pad=1,stride=stride,postfix='1')
            self.bnscalerelu(postfix='1')
            self.conv(nout,3, pad=1, postfix='2')
            scale2 = self.bnscale(postfix='2')
            self.set_bottom(bottom)
            if nin!=nout:
                self.conv(nout,1, stride=stride,postfix='_expand')
                self.bnscale(postfix='_expand')
            self.eltwise(scale2,layername='')
            self.relu()
        return nout;
    def bottleneck(self, nin, nout, s, stride):
        bottom = self.bottom
        with self.scope(s):
            self.conv(nout,1,stride=stride,postfix='1')
            self.bnscalerelu(postfix='1')
            self.conv(nout,3,pad=1, postfix='2')
            self.bnscalerelu(postfix='2')
            self.conv(nout*4,1, postfix='3')
            scale3=self.bnscale(postfix='3')
            self.set_bottom(bottom)
            if nin!=nout*4:
                self.conv(nout*4,1,stride=stride,postfix='_expand')
                self.bnscale(postfix='_expand')
            self.eltwise(scale3,layername='')
            self.relu()
        return nout*4
    def gen_net(self, nclass, depth=18, deploy=False):
        net_defs = {
            10:([1, 1, 1, 1], self.standard),
            18:([2, 2, 2, 2], self.standard),
            34:([3, 4, 6, 3], self.standard),
            50:([3, 4, 6, 3], self.bottleneck),
            101:([3, 4, 23, 3], self.bottleneck),
            152:([3, 8, 36, 3], self.bottleneck),
        }
        assert depth in net_defs.keys(), "net of depth:{} not defined".format(depth)
        nunits_list, block_func = net_defs[depth] # nunits_list a list of integers indicating the number of layers in each depth.
        nouts = [64, 128, 256, 512] # same for all nets
        #add stem
        self.conv(64,7,stride=2,pad=3,layername='conv1')
        self.bnscale(postfix='_conv1')
        self.relu(prefix='conv1')
        self.maxpool(3,2,layername='pool1')
        nin = 64;
        for s in range(4):
            nunits = nunits_list[s]
            for unit in range(1, nunits + 1): # for each unit. Enumerate from 1.
                if unit > 1 and nunits > 6:
                    block_prefix = 'res' + str(s+2) + 'b' + str(unit-1)  # layer name prefix
                else:
                    block_prefix = 'res' + str(s+2) + chr(ord('a')+unit-1) # layer name prefix
                stride = 2 if unit==1 and s>0  else 1
                nin = block_func(nin, nouts[s], block_prefix, stride)
        self.avepoolglobal(layername='pool5')
        self.fc(nclass,bias=True,postfix='_'+str(nclass))
        if deploy==False:
            self.set_conv_params()

class DarkNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, nout):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks==3))
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def gen_net(self,nclass,deploy=False):
        stages = [(32,1),(64,1),(128,3),(256,3),(512,5),(1024,5)]
        for i,stage in enumerate(stages):
            for j in range(stage[1]):
                s = 'dark'+str(i+1)+chr(ord('a')+j) if stage[1]>1 else 'dark'+str(i+1)
                if j%2==0:
                    self.dark_block(s,3,stage[0])
                else:
                    self.dark_block(s+'_1',1,stage[0]//2)
            if i<len(stages)-1:
                self.maxpool(2,2,layername='pool'+str(i+1));
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()

class MobileNetV1(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dw_block(self,  s, nin, nout, stride=1):
        with self.scope(s):
            self.dwconv(nin, 3, stride=stride, pad=1, group=nin, postfix='dw' )
            self.bnscalerelu(postfix='dw')
            self.conv(nout,1,postfix='sep')
            self.bnscalerelu(postfix='sep')
        return nout
    def gen_net(self, nclass, deploy=False):
        #add stem
        nin = 32;   #first stage always 32 out channels
        self.conv(nin,3, pad=1,stride=2,postfix='stem')
        self.bnscalerelu(postfix='stem')
        
        stages = [(2,128),(2,256),(2,512),(6,1024),(1,1024)]
        for i,stage in enumerate(stages):
            nunit = stage[0]
            nout = stage[1]
            for j in range(1,nunit+1):
                block_prefix = 'mn%d_%d'%(i+2,j) # layer name prefix
                stride = 2 if nunit>1 else 1
                if j<nunit:
                    nin = self.dw_block(block_prefix,nin,nout//2)
                else:
                    nin = self.dw_block(block_prefix,nin,nout,stride=stride)
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,postfix=str(nclass))
        if deploy==False:
            self.set_conv_params()
            
# Relu6 not supported by caffe, use relu instead
class MobileNetV2(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def bottleneck_block(self,  s, nin, nout, stride=1, factor=1):
        with self.scope(s):
            inputlayer = self.bottom
            self.conv(nin*factor,1,postfix='pw')
            self.bnscalerelu(postfix='pw')
            self.dwconv(nin*factor, 3, stride=stride, pad=1, group=nin*factor, postfix='dw' )
            self.bnscalerelu(postfix='dw') 
            self.conv(nout,1,postfix='linear')
            self.bnscale(postfix='linear')
            if stride==1 and nin==nout:
                self.eltwise(inputlayer,postfix='add')
        return nout
    def gen_net(self, nclass, factor=6, deploy=False):
        #add stem
        nin = 32;   #first stage always 32 out channels
        self.conv(nin,3, pad=1,stride=2,postfix='stem')
        self.bnscalerelu(postfix='stem')
        stages = [(1,16,1),(2,24,2),(3,32,2),(4,64,2),(3,96,1),(3,160,2),(1,320,1),]
        for i,stage in enumerate(stages):
            nunit  = stage[0]
            nout   = stage[1]
            stage_stride = stage[2]
            stage_factor = factor if i>0 else 1
            for j in range(1,nunit+1):
                block_prefix = 'mn%d_%d'%(i+1,j) # layer name prefix
                stride = 1 if stage_stride==1 or j>1 else 2
                nin = self.bottleneck_block(block_prefix, nin, nout, stride=stride, factor=stage_factor) 
        self.conv(1280,1,postfix='8')
        self.avepoolglobal(layername='pool8')
        self.fc(nclass,bias=True,postfix=str(nclass))
        if deploy==False:
            self.set_conv_params()

class ShuffleNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def shuffle_block(self, nin, nout, s, stride=1, group=3, bn_ratio=0.25):
        bottom = self.bottom
        noutnew = nout if stride==1 else nout-nin
        bn_channels = int(noutnew*bn_ratio)
        with self.scope(s):
            if nin==24:
                self.conv1x1(bn_channels,stride=1, postfix='1')
            else:
                self.conv1x1(bn_channels,stride=1, group=group, postfix='1')
            self.bnscalerelu(postfix='1')
            if nin!=24:
                self.shuffle(group=group)
            self.dwconv(bn_channels,3, group=bn_channels, stride=stride, pad=1, postfix='2')
            self.bnscale(postfix='2')
            self.conv1x1(noutnew,stride=1, group=group, postfix='3')
            rightbranch = self.bnscale(postfix='3')
            if stride==1:
                self.eltwise(bottom, postfix='join')
            else:
                self.avepool(3,2,bottom=bottom)
                self.concat(rightbranch,postfix='join')
        return nout;
    def gen_net(self, nclass, group=3,bn_ratio=0.25,deploy=False):
        nout_cfg = { 1:144, 2:200,3:240,4:272, 8:384 }
        net_stages = [4,8,4]

        assert group in nout_cfg.keys(), "net of group:{} not defined".format(group)
        nout_base = nout_cfg[group];
        #add stem
        nin = 24;   #first stage always 24 out channels
        self.conv3x3(nin,stride=2,postfix='1')
        self.bnscalerelu(postfix='1')
        self.maxpool(3,2,layername='pool1')

        for s in range(3):
            nunits = net_stages[s]
            nout = nout_base * (2**s)
            for unit in range(nunits): # for each unit. Enumerate from 1.
                block_prefix = 'resx' + str(s+3) + chr(ord('a')+unit) # layer name prefix
                stride = 2 if unit==0  else 1
                nin = self.shuffle_block(nin, nout, block_prefix, stride=stride, group=group, bn_ratio=bn_ratio)
        self.avepoolglobal(layername='pool5')
        self.fc(nclass,bias=True,postfix=str(nclass))
        if deploy==False:
            self.set_conv_params()

def test_shufflenet(nclass=1000, group=3,bn_ratio=0.25):
    name = 'shuffle'+str(group)+'_'+str(bn_ratio)
    trainnet = ShuffleNet(name)
    testnet = ShuffleNet(name)
    saveproto(trainnet,testnet, 'fc_'+str(nclass),'label', nclass, [224,256],[64,50], group=3, bn_ratio=0.25)

def test_resnet(nclass, depth):
    name = 'resnet'+str(depth)
    trainnet = ResNet(name)
    testnet = ResNet(name)
    saveproto(trainnet,testnet, 'fc_'+str(nclass),'label', nclass, [224,256],[64,50], depth=depth)

def test_darknet(nclass):
    name = 'darknet'
    trainnet = DarkNet(name)
    testnet = DarkNet(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,50])

def test_caffenet(nclass):
    name = 'caffenet'
    trainnet = CaffeNet(name)
    testnet = CaffeNet(name)
    saveproto(trainnet,testnet, 'fc8','label', nclass, [227,256],[64,50])

def test_vggnet(nclass,depth):
    name = 'vgg'+str(depth)
    trainnet = VggNet(name)
    testnet = VggNet(name)
    saveproto(trainnet,testnet, 'fc8','label', nclass, [227,256],[64,50],depth=depth)

def test_mobilenetv1(nclass):
    name = 'mobilenetv1'
    trainnet = MobileNetV1(name)
    testnet = MobileNetV1(name)
    saveproto(trainnet,testnet, 'fc_'+str(nclass),'label', nclass, [224,256],[64,100])


def test_mobilenetv2(nclass):
    name = 'mobilenetv2'
    trainnet = MobileNetV2(name)
    testnet = MobileNetV2(name)
    saveproto(trainnet,testnet, 'fc_'+str(nclass),'label', nclass, [224,256],[64,100])
        
if __name__ == "__main__":
    #test_resnet(1000,101);
    #test_resnet(1000,18);
    #test_darknet(1000)
    #test_caffenet(1000)
    #test_vggnet(1000,16)
    #test_shufflenet(1000,3,0.25)
    test_mobilenetv2(1000)