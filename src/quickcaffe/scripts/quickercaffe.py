from collections import OrderedDict, Counter
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from contextlib import contextmanager
import caffe
import os.path as op

def _last_layer(n):
    return n.__dict__['tops'][n.__dict__['tops'].keys()[-1]]
def _input(blobshape):
    return L.Input(shape={'dim':blobshape})
def _conv(bottom, nout, ks, stride, pad, bias):
    return L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad, bias_term=bias);
def _deconv(bottom, nout, ks, stride, pad, bias):
    return L.Deconvolution(bottom, convolution_param=dict(kernel_size=ks, stride=stride, num_output=nout, pad=pad, bias_term=bias));
def _fc(bottom, nout, bias):
    return L.InnerProduct(bottom, num_output = nout, bias_term=bias)
def _bn(bottom,in_place=True ):
    bnlayer = L.BatchNorm(bottom, in_place=in_place)
    bnlayer.fn.params['param']=[dict(lr_mult=0, decay_mult=0)]*3
    return bnlayer
def _scale(bottom ,in_place=True ):
    return L.Scale(bottom, in_place=in_place, scale_param=dict(bias_term=True))
def _relu(bottom,negslope,in_place=True):
    if negslope!=0:
        return L.ReLU(bottom, in_place=in_place, negative_slope=negslope)
    else:
        return L.ReLU(bottom, in_place=in_place)
def _maxpool(bottom, ks, stride, pad):
    return L.Pooling(bottom, pool=P.Pooling.MAX, stride = stride, kernel_size = ks, pad = pad)
def _avepool(bottom, ks, stride, pad):
    return L.Pooling(bottom, pool=P.Pooling.AVE, stride = stride, kernel_size = ks, pad = pad)
def _avepoolglobal(bottom):
    return L.Pooling(bottom, pool = P.Pooling.AVE, global_pooling = True)
def _maxpoolglobal(bottom):
    return L.Pooling(bottom, pool = P.Pooling.MAX, global_pooling = True)
def _softmaxwithloss(bottom, label):
    return L.SoftmaxWithLoss(bottom, label)
def _eltwise(bottom,layer):
    return L.Eltwise(bottom,layer)
def _accuracy(bottom, label, top_k, testonly=False):
    if testonly:
        evalphase = dict(phase=getattr(caffe_pb2, 'TEST'))
        return L.Accuracy(bottom, label, include=evalphase, accuracy_param=dict(top_k=top_k))
    else:
        return L.Accuracy(bottom, label, accuracy_param=dict(top_k=top_k))
def _resize(bottom, target, intepolation=1):            #Nearest: 1, Bilinear=2
    return L.Resize(bottom,target, function_type=2, intepolation_type=intepolation);
#layer_name syntax   {prefix}/{layername|layertype}_{postfix}
class NeuralNetwork :
    def __init__ (self, name):
        self.n = caffe.NetSpec();
        self.bottom = None;
        self.name = 'name: "%s"'%name;
        self.trainstr = '';
        self.__prefix = None
    @contextmanager
    def scope(self,prefix):
        self.__prefix = prefix;
        yield
        self.__prefix = None;
    def last_layer(self):
        return _last_layer(self.n)
    def get_nin(self):
        nout = self.bottom.fn.params.get('num_output')
        if nout is None:
            layers = OrderedDict()
            autonames = Counter()
            self.bottom.fn._to_proto(layers, {}, autonames)
            for i in range(len(layers)):
                btlayer_param = layers.items()[-1-i][1]
                if btlayer_param.type=='Convolution':
                    return btlayer_param.convolution_param.num_output
        return nout;
    def getlayers(self, blacklist,whitelist):
        layerlist = [];
        if whitelist is not None:
            layerlist = [self.n.tops[key] for key in whitelist]
        else:
            for key,layer in self.n.tops.items():
                if blacklist is not None and key in blacklist:
                    continue;
                layerlist += [layer]
        return layerlist
    def toproto(self):
        net = self.n.to_proto()
        #net.input.extend(inputs)
        #net.input_dim.extend(dims)
        return '\n'.join([self.name, self.trainstr,str(net)])
    def set_bottom(self,layer):
        self.bottom = layer
    def set_bottombyname(self,layername):
        self.bottom = self.n[layername];
    def set_conv_params(self,
        weight_param = dict(lr_mult=1, decay_mult=1),
        weight_filler = dict(type='msra'),
        bias_param   = dict(lr_mult=2, decay_mult=0),
        bias_filler  = dict(type='constant', value=0),
        blacklist=None, whitelist=None
        ):
        for layer in self.getlayers(blacklist,whitelist):
            layertype = layer.fn.type_name
            layerparams = layer.fn.params
            if layertype in ['Convolution', 'Deconvolution', 'InnerProduct']:
                if 'bias_term' not in layerparams or  layerparams['bias_term']==True:
                    layerparams['param']=[weight_param, bias_param]
                else:
                    layerparams['param']=[weight_param]
                if 'convolution_param' in layerparams:
                    layerparams = layerparams['convolution_param']
                if 'bias_term' not in layerparams or  layerparams['bias_term']==True:
                    layerparams['bias_filler']=bias_filler
                layerparams['weight_filler']=weight_filler
    def lock_batchnorm(self, lockbn=False, blacklist=None, whitelist=None):
        for layer in self.getlayers(blacklist,whitelist):
            layertype = layer.fn.type_name
            layerparams = layer.fn.params
            if layertype == 'BatchNorm':
                if lockbn:
                    layerparams['batch_norm_param']=dict(use_global_stats=True)
    def __get_topname(self, layertype, prefix, layername, postfix):
        prefix =  self.__prefix if prefix is None and self.__prefix is not None else prefix
        layername = layertype if layername is None else layername
        name = '_'.join([layername,postfix]) if postfix is not None else layername
        return '/'.join([prefix,name]) if prefix is not None else name
    def conv(self, nout, ks, prefix=None, layername=None, postfix=None, stride=1, pad=0, bias=False, bottom=None):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('conv', prefix,layername, postfix)
        self.n[topname] = _conv(bottom, nout,ks,stride,pad,False)
        self.bottom = self.n[topname];
        return self.bottom
    def conv3x3(self, nout, prefix=None, layername=None, postfix=None, stride=1, pad=1, bias=False, bottom=None):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('conv3x3', prefix,layername, postfix)
        self.n[topname] = _conv(bottom, nout,3,stride,pad,False)
        self.bottom = self.n[topname];
        return self.bottom
    def conv1x1(self, nout,  prefix=None, layername=None, postfix=None, stride=1, pad=0, bias=False, bottom=None):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('conv1x1', prefix,layername, postfix)
        self.n[topname] = _conv(bottom, nout,1,stride,pad,False)
        self.bottom = self.n[topname];
        return self.bottom
    def deconv(self, nout, ks,  prefix=None, layername=None, postfix=None, stride=1, pad=0, bias=False, bottom=None):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('deconv', prefix,layername, postfix)
        self.n[topname] = _deconv(bottom, nout,ks,stride,pad,False)
        self.bottom = self.n[topname];
        return self.bottom
    def fc(self, nout,  prefix=None, layername=None, postfix=None, bias=False, bottom=None):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('fc', prefix,layername, postfix)
        self.n[topname] = _fc( bottom, nout,bias)
        self.bottom = self.n[topname];
        return self.bottom
    def bn(self,  prefix=None, layername=None, postfix=None, bottom=None,in_place=True ):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('bn', prefix,layername, postfix)
        self.n[topname] = _bn(bottom,in_place=in_place)
        self.bottom = self.n[topname];
        return self.bottom
    def scale(self,  prefix=None, layername=None, postfix=None, bottom=None,in_place=True ):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('scale', prefix,layername, postfix)
        self.n[topname] = _scale(bottom,in_place=in_place)
        self.bottom = self.n[topname];
        return self.bottom
    def relu(self,  prefix=None, layername=None, postfix=None, bottom=None,in_place=True ):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('relu', prefix,layername, postfix)
        self.n[topname] = _relu(bottom,0,in_place=in_place)
        self.bottom = self.n[topname];
        return self.bottom
    def leakyrelu(self, negslope, prefix=None, layername=None, postfix=None, bottom=None,in_place=True):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('leaky', prefix,layername, postfix)
        self.n[topname] = _relu(bottom, negslope,in_place=in_place)
        self.bottom = self.n[topname];
        return self.bottom
    def maxpool(self, ks, stride,  prefix=None, layername=None, postfix=None, pad=0, bottom=None):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('maxpool', prefix,layername, postfix)
        self.n[topname] = _maxpool(bottom, ks, stride,pad)
        self.bottom = self.n[topname];
        return self.bottom
    def avepool(self, ks, stride,  prefix=None, layername=None, postfix=None, pad=0, bottom=None):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('avepool', prefix,layername, postfix)
        self.n[topname] = _avepool(bottom, ks, stride, pad)
        self.bottom = self.n[topname];
        return self.bottom
    def maxpoolglobal(self,  prefix=None, layername=None, postfix=None, bottom=None):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('maxpoolg', prefix,layername, postfix)
        self.n[topname] = _maxpoolglobal(bottom)
        self.bottom = self.n[topname];
        return self.bottom
    def avepoolglobal(self,  prefix=None, layername=None, postfix=None, bottom=None):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('avepoolg', prefix,layername, postfix)
        self.n[topname] = _avepoolglobal(bottom)
        self.bottom = self.n[topname];
        return self.bottom
    def input(self,  blobshape,  prefix=None, layername=None, postfix=None, bottom=None):
        topname = self.__get_topname('data', prefix,layername, postfix)
        self.n[topname] = _input(blobshape)
        self.bottom = self.n[topname];
        return self.bottom
    def eltwise(self, layer, prefix=None, layername=None, postfix=None, bottom=None):
        bottom = self.bottom if bottom is None else bottom
        topname = self.__get_topname('elt', prefix,layername, postfix)
        if isinstance(layer, str):
            self.n[topname] = _eltwise(bottom,self.n[layer]);
        else:
            self.n[topname] = _eltwise(bottom,layer);
        self.bottom = self.n[topname]
        return self.bottom
    def softmaxwithloss(self, score_lname, label_lname,  prefix=None, layername=None, postfix=None):
        topname = self.__get_topname('softmax', prefix,layername, postfix)
        if isinstance(score_lname, str):
            self.n[topname] = _softmaxwithloss(self.n[score_lname], self.n[label_lname])
        else:
            self.n[topname] = _softmaxwithloss(score_lname, self.n[label_lname])
    def accuracy(self, score_lname, label_lname, top_k=1, prefix=None, layername=None, postfix=None, testonly=False):
        topname = self.__get_topname('acc@%d'%top_k, prefix,layername, postfix)
        if isinstance(score_lname, str):
            self.n[topname] = _accuracy(self.n[score_lname], self.n[label_lname],top_k,testonly=testonly)
        else:
            self.n[topname] = _accuracy(score_lname, self.n[label_lname],top_k,testonly=testonly)
    def bnscale(self,  prefix=None, postfix=None,bottom=None,in_place=True):
        self.bn(prefix=prefix, postfix=postfix, bottom=bottom,in_place=in_place)
        self.scale(prefix=prefix, postfix=postfix)
        return self.bottom
    def bnscalerelu(self, prefix=None, postfix=None, bottom=None,in_place=True):
        self.bn(prefix=prefix, postfix=postfix, bottom=bottom,in_place=in_place)
        self.scale(prefix=prefix, postfix=postfix)
        self.relu(prefix=prefix, postfix=postfix)
        return self.bottom
    def resize(self, target, prefix=None, postfix=None, layer_name=None, bottom=None):
        topname = self.__get_topname('resize', prefix, layer_name,postfix)
        bottom = self.bottom if bottom is None else bottom
        if isinstance(target, str):
            self.n[topname] = _resize(bottom, self.n[target])
        else:
            self.n[topname] = _resize(bottom, target)
        self.bottom = self.n[topname]
        return self.bottom
    def _tsv_inception_layer(self, data_path, crop_size, phase, batchsize=(256,50), new_image_size = (256, 256)):
        mean_value = [104, 117, 123]
        colorkl_file = op.join(op.split(data_path)[0], 'train.resize480.shuffled.kl.txt').replace('\\', '/')
        transform_param=dict(crop_size=crop_size, mean_value=mean_value, mirror=(phase==caffe.TRAIN))
        if phase == caffe.TRAIN:
            data_param  = dict(source=data_path, batch_size=batchsize[0], col_data=2, col_label=1, new_width=new_image_size[0], new_height=new_image_size[1], crop_type=2, color_kl_file=colorkl_file)
        else:
            data_param  = dict(source=data_path.replace('train', 'val'), batch_size=batchsize[1], col_data=2, col_label=1, new_width=new_image_size[0], new_height=new_image_size[1], crop_type=2)
        data,label = L.TsvData(ntop=2,  transform_param=transform_param, tsv_data_param=data_param, include=dict(phase=phase))
        return data,label;
    def tsv_inception_layer(self, data_path, crop_size, batchsize=(256,50), new_image_size = (256, 256), phases=[caffe.TRAIN]):
        for phase in phases:
            self.n['data'],self.n['label'] = self._tsv_inception_layer(data_path,crop_size, phase, batchsize,new_image_size);
            if phase==caffe.TRAIN:
                self.trainstr = str(self.n.to_proto())
            else:
                self.bottom=self.n['data']

class SPNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def standard(self, stageins, nin, nout, s, stride,factor):
        stageouts = [None]*len(stageins)
        #resnet building block
        for i, bottom in enumerate(stageins):
            with self.scope(s+'_'+str(i)):
                noutnew = nout if i==0 else int(nout*factor)
                convout = self.conv3x3(noutnew,stride=stride,postfix='1',bottom=bottom)
                self.bnscalerelu(postfix='1',bottom=convout)
                self.conv3x3(nout,postfix='2')
                scale2 = self.bnscale(postfix='2')
                self.set_bottom(bottom)
                if stride>1:
                    self.conv1x1(nout,stride=stride,postfix='expand')
                    self.bnscale(postfix='expand')
                stageouts[i] = self.eltwise(scale2)
        #mutli-scale residual link
        for i in range(len(stageouts)):
            with self.scope(s+'_'+str(i)):
                if i > 0:
                    upsamplelayer = self.resize(stageouts[i],bottom=stageouts[i-1])
                    stageouts[i] = self.eltwise(upsamplelayer,postfix='ms',bottom=stageouts[i])
        #relu
        for i in range(len(stageouts)):
            with self.scope(s+'_'+str(i)):
                stageouts[i]=self.relu(bottom=stageouts[i],in_place=False)
        return stageouts;
    def bottleneck(self, nin, nout, s, stride):
        bottom = self.bottom
        with self.scope(s):
            self.conv1x1(nout,stride=stride,postfix='1')
            self.bnscalerelu(postfix='1')
            self.conv3x3(nout,postfix='2')
            self.bnscalerelu(postfix='2')
            self.conv1x1(nout*4,postfix='3')
            scale3=self.bnscale(postfix='3')
            self.set_bottom(bottom)
            if nin!=nout*4:
                self.conv1x1(nout*4,stride=stride,postfix='expand')
                self.bnscale(postfix='expand')
            self.eltwise(scale3)
            self.relu()
        return self.bottom
    def stem(self, stageins, nout):
        stageouts = [None]*len(stageins)
        for i in range(len(stageins)):
            with self.scope('stem_'+str(i)):
                stageouts[i] = self.conv(nout,7,stride=2,pad=3, bottom = stageins[i])
        for i in range(len(stageouts)):
            with self.scope('stem_'+str(i)):        
                if i > 0:
                    upsamplelayer = self.resize(stageouts[i],bottom=stageouts[i-1])
                    stageouts[i] =  self.eltwise(upsamplelayer,postfix='ms',bottom=stageouts[i])
        for i in range(len(stageouts)):
            with self.scope('stem_'+str(i)):        
                self.bnscalerelu(bottom = stageouts[i],in_place=False)
                stageouts[i]=self.maxpool(3,2)
        return stageouts;

    def spnet(self, nclass, depth,scale,factor):
        net_defs = {
            10:([1,1, 1, 1, 1], self.standard),
            18:([1,2, 2, 2, 2], self.standard),
            34:([1,3, 4, 6, 3], self.standard),
            50:([1,3, 4, 6, 3], self.bottleneck),
            101:([1,3, 4, 23, 3], self.bottleneck),
            152:([1,3, 8, 36, 3], self.bottleneck),
        }
        assert depth in net_defs.keys(), "net of depth:{} not defined".format(depth)
        nunits_list, block_func = net_defs[depth] # nunits_list a list of integers indicating the number of layers in each depth.
        nouts = [('stem',64),('res2',64),('res3', 128),('res4', 256), ('res5',512)] #  layer prefix and output for each stage
        stageouts = [None]*scale
        for i in reversed(range(scale)):
            with self.scope('data'):
                stageouts[i] = self.avepool(2,2, postfix=str(scale-1-i)) if i!=scale-1 else self.bottom
        #add stem
        for i, nout in enumerate(nouts):
            if i==0:
                stageouts = self.stem(stageouts,nout[1])
            else:
                nunits = nunits_list[i]
                for unit in range(nunits):
                        stride = 2 if unit==0 and i>1 else 1       #stride=2 start from res3
                        stageouts=block_func(stageouts,nin, nout[1], nouts[i][0]+chr(ord('a')+unit), stride,factor)
            nin = nout[1]
        for i,stageout in enumerate(stageouts):
            with self.scope('fc_' + str(i)):
                self.avepoolglobal(postfix='5', bottom=stageout)
                stageouts[i]=self.fc(nclass,bias=True,postfix=str(nclass))
        return stageouts

class ResNet(NeuralNetwork):
    def __init__ (self, name ):
        NeuralNetwork.__init__(self,name)
    def standard(self, nin, nout, s, stride ):
        bottom = self.bottom
        with self.scope(s):
            self.conv3x3(nout,stride=stride,postfix='1')
            self.bnscalerelu(postfix='1')
            self.conv3x3(nout,postfix='2')
            scale2 = self.bnscale(postfix='2')
            self.set_bottom(bottom)
            if stride>1:
                self.conv1x1(nout,stride=stride,postfix='expand')
                self.bnscale(postfix='expand')
            self.eltwise(scale2)
            self.relu()
        return self.bottom;
    def bottleneck(self, nin, nout, s, stride):
        bottom = self.bottom
        with self.scope(s):
            self.conv1x1(nout,stride=stride,postfix='1')
            self.bnscalerelu(postfix='1')
            self.conv3x3(nout,postfix='2')
            self.bnscalerelu(postfix='2')
            self.conv1x1(nout*4,postfix='3')
            scale3=self.bnscale(postfix='3')
            self.set_bottom(bottom)
            if nin!=nout*4:
                self.conv1x1(nout*4,stride=stride,postfix='expand')
                self.bnscale(postfix='expand')
            self.eltwise(scale3)
            self.relu()
        return self.bottom
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
        self.conv(64,7,stride=2,pad=3,postfix='1')
        self.bnscalerelu(postfix='1')
        self.maxpool(3,2,postfix='1')
        nin = 64;
        for s in range(4):
            nunits = nunits_list[s]
            for unit in range(1, nunits + 1): # for each unit. Enumerate from 1.
                if unit > 1 and nunits > 6:
                    block_prefix = 'res' + str(s+2) + 'b' + str(unit-1)  # layer name prefix
                else:
                    block_prefix = 'res' + str(s+2) + chr(ord('a')+unit-1) # layer name prefix
                stride = 2 if unit==1 and s>0  else 1
                block_func(nin, nouts[s], block_prefix, stride)
                nin = nouts[s]
        self.avepoolglobal(postfix='5')
        self.fc(nclass,bias=True,postfix=str(nclass))

def test_spnet(depth):
    name = 'spnet'+str(depth)
    n = SPNet(name)
    n.input([1,3,224,224])
    n.spnet(1000,depth,2,0.25)
    with open(name+'_deploy.prototxt','w') as fout:
        fout.write(n.toproto());
    n = SPNet(name)
    n.tsv_inception_layer("tsv480/train.resize480.shuffled.tsv",224,batchsize=(64,50),new_image_size =(256,256),phases=[caffe.TRAIN,caffe.TEST])
    stageouts = n.spnet(1000,depth,2,0.25)
    n.set_conv_params()
    for i,stageout in enumerate(stageouts):
        with n.scope('output_' + str(i)):
            n.softmaxwithloss(stageout,'label')
            n.accuracy(stageout,'label')
            n.accuracy(stageout,'label',top_k=5, testonly=True)
    with open(name+'_trainval.prototxt','w') as fout:
        fout.write(n.toproto());


def test_resnet(depth):
    name = 'resnet'+str(depth)
    n = ResNet(name)
    n.input([1,3,224,224])
    n.resnet(1000,depth)
    with open(name+'_deploy.prototxt','w') as fout:
        fout.write(n.toproto());
    n = ResNet(name)
    n.tsv_inception_layer("tsv480/train.resize480.shuffled.tsv",224,batchsize=(64,50),new_image_size =(256,256),phases=[caffe.TRAIN,caffe.TEST])
    n.resnet(1000,depth)
    n.set_conv_params()
    n.softmaxwithloss('fc_1000','label')
    n.accuracy('fc_1000','label')
    n.accuracy('fc_1000','label',top_k=5, testonly=True)
    with open(name+'_trainval.prototxt','w') as fout:
        fout.write(n.toproto());

if __name__ == "__main__":
    test_resnet(18);