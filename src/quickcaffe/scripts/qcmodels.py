from collections import OrderedDict, Counter
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from contextlib import contextmanager
import caffe
import os.path as op
from quickercaffe import NeuralNetwork,saveproto

class FastDarkNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, group, nout):
        with self.scope(s):
            self.conv(nout,ks, group=group, pad=(ks==3))
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def gen_net(self,nclass,ratio=1.25, deploy=False):
        stages = [(32,1),(64,1),(128,3),(256,3),(512,5),(1024,5)]
        for i,stage in enumerate(stages):
            for j in range(stage[1]):
                s = 'dark'+str(i+1)+chr(ord('a')+j) if stage[1]>1 else 'dark'+str(i+1)
                if i<2: #stem
                    self.dark_block(s,3, 1, stage[0])
                elif i==4 and j==3:     #fix the dimension in this channel for iris feature extraction
                    self.dark_block(s,3, 1, int(stage[0]//2))
                elif j%2==0:
                    self.dark_block(s,1, 1, int(stage[0]*ratio))
                else:
                    self.dark_block(s,3, 4, int(stage[0]*ratio//2))
            if i<len(stages)-1:
                self.maxpool(2,2,layername='pool'+str(i+1));
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()
class FasterDarkNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, group, nout):
        with self.scope(s):
            self.conv(nout,ks, group=group, pad=(ks==3))
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def gen_net(self,nclass,ratio=1.25, deploy=False):
        stages = [(32,1),(64,1),(128,3),(256,3),(512,5),(1024,5)]
        for i,stage in enumerate(stages):
            for j in range(stage[1]):
                s = 'dark'+str(i+1)+chr(ord('a')+j) if stage[1]>1 else 'dark'+str(i+1)
                if i<2: #stem
                    self.dark_block(s,3, 1, stage[0])
                elif i==4 and j==3:     #fix the dimension in this channel for iris feature extraction
                    self.dark_block(s,3, 1, int(stage[0]//2))
                elif j%2==0:
                    self.dark_block(s,1, 2, int(stage[0]*ratio))
                else:
                    self.dark_block(s,3, 5, int(stage[0]*ratio//2))
            if i<len(stages)-1:
                self.maxpool(2,2,layername='pool'+str(i+1));
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()

class ShuffleDarkNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, nout):
        with self.scope(s):
            self.conv(nout,3, pad=1)
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def shuffle_block(self, s, group, stride, nout):
        with self.scope(s):
            self.conv1x1(nout//2,group=group, postfix='1')
            self.bnscale(postfix='1')
            self.leakyrelu(0.1,postfix='1')
            self.shuffle(group=group)
            self.dwconv(nout//2,3, pad=1,group=nout//2, stride=stride, postfix='2')
            self.bnscale(postfix='2')
            self.conv1x1(nout,group=group,postfix='3')
            self.bnscale(postfix='3')
            self.leakyrelu(0.1,postfix='3')            
        return self.bottom;
    def gen_net(self,nclass, group=4, ratio=1.25, deploy=False):
        nouts = [32,64,128,256,512,1024]
        self.dark_block('dark1',nouts[0])
        self.maxpool(2,2,layername='pool1');
        self.dark_block('dark2',nouts[1])
        self.shuffle_block('dark3', group, 2, int(nouts[2]*ratio))
        self.shuffle_block('dark4', group, 2, int(nouts[3]*ratio))
        self.shuffle_block('dark5a', group, 2, int(nouts[4]*ratio))
        self.dark_block('dark5b',nouts[4])
        self.shuffle_block('dark5c', group, 1, int(nouts[4]*ratio))
        self.shuffle_block('dark6a', group, 2, int(nouts[5]*ratio))
        self.shuffle_block('dark6b', group, 1, int(nouts[5]*ratio))
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()

class DenseDarkNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def dark_block(self, s, ks, nout,stride=1):
        with self.scope(s):
            self.conv(nout,ks, pad=(ks==3),stride=stride)
            self.bnscale()
            self.leakyrelu(0.1)
        return self.bottom;
    def gen_net(self,nclass,deploy=False):
        stages = [(32,1),(64,1),(128,3),(256,3),(512,5),(1024,5)]
        self.dark_block('dark1',3,32)
        self.maxpool(2,2,layername='pool1');
        self.dark_block('dark2',3,64)
        self.maxpool(2,2,layername='pool2');
        self.dark_block('dark3a',3,128)
        self.dark_block('dark3b',1,64)
        self.dark_block('dark3c',3,128)
        self.maxpool(2,2,layername='pool3');
        self.dark_block('dark4a',3,256)
        self.dark_block('dark4b',1,128)
        self.dark_block('dark4c',3,256)
        self.maxpool(2,2,layername='pool4');
        self.dark_block('dark5a',3,512)
        self.dark_block('dark5b',1,256)
        self.dark_block('dark5c',3,512)
        self.dark_block('dark5d',1,256)
        dark6=self.dark_block('dark5e',3,512,stride=2)
        for i in range(16):
            postfix = '6'+ chr(ord('a')+i)
            self.dark_block('dark%s_1'%postfix,1,128)
            self.dark_block('dark%s_2'%postfix,3,32)
            dark6 = self.concat(dark6,layername='concat'+postfix)
        self.avepoolglobal(layername='pool6')
        self.fc(nclass,bias=True,layername='fc7')
        if deploy==False:
            self.set_conv_params()
def test_densedarknet(nclass):
    name = 'densedarknet'
    trainnet = DenseDarkNet(name)
    testnet = DenseDarkNet(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,100])
            
def test_fastdarknet(nclass,ratio):
    name = 'fastdarknet_'+str(ratio)
    trainnet = FastDarkNet(name)
    testnet = FastDarkNet(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,50], ratio=ratio)
def test_fasterdarknet(nclass,ratio):
    name = 'fasterdarknet_'+str(ratio)
    trainnet = FasterDarkNet(name)
    testnet = FasterDarkNet(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,50], ratio=ratio)

def test_shuffledarknet(nclass,group,ratio):
    name = '_'.join(['shuffledarknet_',str(group),str(ratio)])
    trainnet = ShuffleDarkNet(name)
    testnet = ShuffleDarkNet(name)
    saveproto(trainnet,testnet, 'fc7','label', nclass, [224,256],[64,50], ratio=ratio,group=group)
    

if __name__ == "__main__":
    #test_fastdarknet(1000,1.25)
    #test_shuffledarknet(1000,4,1.25)
    #test_fasterdarknet(1000,1.25)
    test_densedarknet(1000)
