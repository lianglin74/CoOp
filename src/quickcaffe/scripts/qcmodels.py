from collections import OrderedDict, Counter
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from contextlib import contextmanager
import caffe
import os.path as op
from quickercaffe import NeuralNetwork


class VggNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def get_topname(self, layertype, prefix, layername, postfix):
        prefix =  self.prefix if prefix is None and self.prefix is not None else prefix
        layername = layertype if layername is None else layername
        name = ''.join([layername,postfix]) if postfix is not None else layername
        if name=='' and prefix is not None:     return prefix;
        return '_'.join([prefix,name]) if prefix is not None else name
    def vggnet(self,nclass,depth,deploy=False):
        netdefs ={16: [(64,2),(128,2),(256,3),(512,3),(512,3)], 16: [(64,2),(128,2),(256,4),(512,4),(512,4)] }
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

class CaffeNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def get_topname(self, layertype, prefix, layername, postfix):
        prefix =  self.prefix if prefix is None and self.prefix is not None else prefix
        layername = layertype if layername is None else layername
        name = ''.join([layername,postfix]) if postfix is not None else layername
        if name=='' and prefix is not None:     return prefix;
        return '_'.join([prefix,name]) if prefix is not None else name
    def caffenet(self,nclass,deploy=False):
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
    def resnet(self, nclass, depth):
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

class DarkNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, nout):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks==3))
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def darknet(self,nclass):
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
        return self.bottom;

def test_resnet(depth):
    name = 'resnet'+str(depth)
    n = ResNet(name)
    n.input([1,3,224,224])
    nclass = 1000
    n.resnet(nclass,depth)
    with open(name+'_deploy.prototxt','w') as fout:
        fout.write(n.toproto());
    n = ResNet(name)
    n.tsv_inception_layer("tsv480/train.resize480.shuffled.tsv",224,batchsize=(64,50),new_image_size =(256,256),phases=[caffe.TRAIN,caffe.TEST])
    n.resnet(nclass,depth)
    n.set_conv_params()
    n.softmaxwithloss('fc_%d'%nclass,'label',layername='loss')
    n.accuracy('fc_%d'%nclass,'label',layername='accuracy')
    n.accuracy('fc_%d'%nclass,'label',top_k=5, testonly=True, layername='accuracy_top_5')
    with open(name+'_trainval.prototxt','w') as fout:
        fout.write(n.toproto());

def test_darknet(nclass):
    name = 'darknet'
    n = DarkNet(name)
    n.input([1,3,224,224])
    n.darknet(nclass)
    with open(name+'_deploy.prototxt','w') as fout:
        fout.write(n.toproto());
    n = DarkNet(name)
    n.tsv_inception_layer("tsv480/train.resize480.shuffled.tsv",224,batchsize=(64,50),new_image_size =(256,256),phases=[caffe.TRAIN,caffe.TEST])
    n.darknet(nclass)
    n.set_conv_params()
    n.softmaxwithloss('fc7','label',layername='loss')
    n.accuracy('fc7','label',layername='accuracy')
    n.accuracy('fc7','label',top_k=5, testonly=True, layername='accuracy_top_5')
    with open(name+'_trainval.prototxt','w') as fout:
        fout.write(n.toproto());

def test_caffenet(nclass):
    name = 'caffenet'
    n = CaffeNet(name)
    n.input([1,3,227,227])
    n.caffenet(nclass,deploy=True)
    with open(name+'_deploy.prototxt','w') as fout:
        fout.write(n.toproto());
    n = CaffeNet(name)
    n.tsv_inception_layer("tsv480/train.resize480.shuffled.tsv",227,batchsize=(64,50),new_image_size =(256,256),phases=[caffe.TRAIN,caffe.TEST])
    n.caffenet(nclass,deploy=False)
    n.set_conv_params( weight_filler = dict(type='gaussian', std=0.01), whitelist=['conv1','conv3','fc8'] )
    n.set_conv_params( weight_filler = dict(type='gaussian', std=0.01), bias_filler= dict(type='constant', value=1), whitelist=['conv2','conv4','conv5'] )
    n.set_conv_params( weight_filler = dict(type='gaussian', std=0.005), bias_filler= dict(type='constant', value=1), whitelist=['fc6','fc7'] )
    n.softmaxwithloss('fc8','label',layername='loss')
    n.accuracy('fc8','label',testonly=True, layername='accuracy')
    n.accuracy('fc8','label',top_k=5, testonly=True, layername='accuracy_top_5')
    with open(name+'_trainval.prototxt','w') as fout:
        fout.write(n.toproto());

def test_vggnet(nclass,depth):
    name = 'vgg'+str(depth)
    n = VggNet(name)
    n.input([1,3,227,227])
    n.vggnet(nclass,depth,deploy=True)
    with open(name+'_deploy.prototxt','w') as fout:
        fout.write(n.toproto());
    n = VggNet(name)
    n.tsv_inception_layer("tsv480/train.resize480.shuffled.tsv",227,batchsize=(64,50),new_image_size =(256,256),phases=[caffe.TRAIN,caffe.TEST])
    n.vggnet(nclass,depth,deploy=False)
    n.set_conv_params( weight_filler = dict(type='gaussian', std=0.01), blacklist=['fc6','fc7','fc8'] )
    n.set_conv_params( weight_filler = dict(type='gaussian', std=0.005), bias_filler= dict(type='constant', value=0.1), whitelist=['fc6','fc7','fc8'] )
    n.softmaxwithloss('fc8','label',layername='loss')
    n.accuracy('fc8','label',testonly=True, layername='accuracy')
    n.accuracy('fc8','label',top_k=5, testonly=True, layername='accuracy_top_5')
    with open(name+'_trainval.prototxt','w') as fout:
        fout.write(n.toproto());

if __name__ == "__main__":
    #test_resnet(101);
    #test_darknet(1000)
    #test_caffenet(1000)
    test_vggnet(1000,16)
