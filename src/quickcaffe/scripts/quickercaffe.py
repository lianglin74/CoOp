from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
from contextlib import contextmanager
import caffe
import os.path as op

def _last_layer(n):
    return n.__dict__['tops'][n.__dict__['tops'].keys()[-1]]
def _input(blobshape):
    return L.Input(shape={'dim':blobshape})
def _lrn(bottomlayer, local_size,alpha, beta):
    return L.LRN(bottomlayer, local_size=local_size, alpha=alpha, beta=beta);
def _dropout(bottomlayer, dropout_ratio, in_place):
    return L.Dropout(bottomlayer,dropout_ratio=dropout_ratio, in_place=in_place)
def _conv(bottomlayer, nout, ks, stride, pad, bias, **kwargs):
    if bias==True:
        return L.Convolution(bottomlayer, kernel_size=ks, stride=stride, num_output=nout, pad=pad, **kwargs);
    else:
        return L.Convolution(bottomlayer, kernel_size=ks, stride=stride, num_output=nout, pad=pad, bias_term=bias,**kwargs);
def _deconv(bottomlayer, nout, ks, stride, pad, bias):
    return L.Deconvolution(bottomlayer, convolution_param=dict(kernel_size=ks, stride=stride, num_output=nout, pad=pad, bias_term=bias));
def _fc(bottomlayer, nout, bias, **kwargs):
    if bias==True:
        return L.InnerProduct(bottomlayer, num_output = nout, **kwargs)
    else:
        return L.InnerProduct(bottomlayer, num_output = nout, bias_term=bias, **kwargs)
def _bn(bottomlayer,in_place=True ):
    bnlayer = L.BatchNorm(bottomlayer, in_place=in_place)
    bnlayer.fn.params['param']=[dict(lr_mult=0, decay_mult=0)]*3
    return bnlayer
def _scale(bottomlayer ,in_place=True ):
    return L.Scale(bottomlayer, in_place=in_place, scale_param=dict(bias_term=True))
def _relu(bottomlayer,negslope,in_place=True):
    if negslope!=0:
        return L.ReLU(bottomlayer, in_place=in_place, negative_slope=negslope)
    else:
        return L.ReLU(bottomlayer, in_place=in_place)
def _maxpool(bottomlayer, ks, stride, pad):
    return L.Pooling(bottomlayer, pool=P.Pooling.MAX, stride = stride, kernel_size = ks, pad = pad)
def _avepool(bottomlayer, ks, stride, pad):
    return L.Pooling(bottomlayer, pool=P.Pooling.AVE, stride = stride, kernel_size = ks, pad = pad)
def _avepoolglobal(bottomlayer):
    return L.Pooling(bottomlayer, pool = P.Pooling.AVE, global_pooling = True)
def _maxpoolglobal(bottomlayer):
    return L.Pooling(bottomlayer, pool = P.Pooling.MAX, global_pooling = True)
def _softmaxwithloss(bottomlayer, label, loss_weight=1):
    if loss_weight==1.0 :
        return L.SoftmaxWithLoss(bottomlayer, label)
    else:
        return L.SoftmaxWithLoss(bottomlayer, label, loss_weight=loss_weight)
def _eltwise(bottomlayer,layer):
    return L.Eltwise(bottomlayer,layer)
def _accuracy(bottomlayer, label, top_k, testonly=False):
    if testonly:
        evalphase = dict(phase=getattr(caffe_pb2, 'TEST'))
        return L.Accuracy(bottomlayer, label, include=evalphase, accuracy_param=dict(top_k=top_k))
    else:
        return L.Accuracy(bottomlayer, label, accuracy_param=dict(top_k=top_k))
def _resize(bottomlayer, target, intepolation=1):            #Nearest: 1, Bilinear=2
    return L.Resize(bottomlayer,target, function_type=2, intepolation_type=intepolation);
def _concat(bottomlayer,layer, axis=1):
    return L.Concat(bottomlayer,layer,concat_param=dict(axis=axis))
def _reorg(bottomlayer, stride=2, reverse=False):        #reverse=False   C*stride*stride, W/stride, H/stride,  
    return L.Reorg(bottomlayer,stride=stride, reverse=reverse)

#layer_name syntax   {prefix}/{layername|layertype}_{postfix}
def layerinit(name):
    def real_decorator(function):
        def wrapper(*args, **kwargs):
            args[0].set_bottom( kwargs.pop('bottom',None) )
            topname = args[0].get_topname(name,  kwargs.pop('prefix',None), kwargs.pop('layername',None),  kwargs.pop('postfix',None))
            newlayer = function(*args, **kwargs)
            if newlayer is not None:
                args[0].n[topname] = newlayer
                args[0].bottom = newlayer
            return newlayer
        return wrapper
    return real_decorator

class NeuralNetwork :
    def __init__ (self, name):
        self.n = caffe.NetSpec();
        self.bottom = None;
        self.name = 'name: "%s"'%name;
        self.trainstr = '';
        self.prefix = None

    def get_topname(self, layertype, prefix, layername, postfix):
        prefix =  self.prefix if prefix is None and self.prefix is not None else prefix
        layername = layertype if layername is None else layername
        name = '_'.join([layername,postfix]) if postfix is not None else layername
        return '/'.join([prefix,name]) if prefix is not None else name
    @contextmanager
    def scope(self,prefix):
        self.prefix = prefix;
        yield
        self.prefix = None;
    def last_layer(self):
        return _last_layer(self.n)
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
        if layer is None: return;
        if isinstance(layer, str):
            self.bottom = self.n[layername];
        else:
            self.bottom = layer
    def set_conv_params(self,
        weight_param = dict(lr_mult=1, decay_mult=1),
        weight_filler = dict(type='msra'),
        bias_param   = dict(lr_mult=2, decay_mult=0),
        bias_filler  = dict(type='constant', value=0),
        blacklist=None, whitelist=None
        ):
        for layer in self.getlayers(blacklist,whitelist):
            layerobj = self.n[layer] if isinstance(layer,str) else layerobj
            layertype = layeobj.fn.type_name
            layerparams = layerobj.fn.params
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
    @layerinit('dropout')                    
    def dropout(self,  dropout_ratio=0.5, in_place=True,deploy=False,**kwargs):
        if deploy==True: return None;
        return  _dropout(self.bottom,dropout_ratio,in_place)
    @layerinit('lrn')                    
    def lrn(self, local_size=5, alpha=1e-4, beta=0.75, **kwargs):
        return _lrn(self.bottom,local_size,alpha,beta)
    @layerinit('conv')       
    def conv(self, nout, ks, stride=1, pad=0, bias=False,  **kwargs):
        return _conv(self.bottom, nout,ks,stride,pad,bias,**kwargs)
    @layerinit('conv3x3')       
    def conv3x3(self, nout, stride=1, pad=1, bias=False, **kwargs):
        return _conv(self.bottom, nout,3,stride,pad,bias,**kwargs)
    @layerinit('conv1x1')
    def conv1x1(self, nout, stride=1, pad=0, bias=False,**kwargs):
        return  _conv(self.bottom, nout,1,stride,pad,bias,**kwargs)
    @layerinit('deconv')
    def deconv(self, nout, ks,   stride=1, pad=0, bias=False, **kwarg):
        return _deconv(self.bottom, nout,ks,stride,pad,False)
    @layerinit('fc')
    def fc(self, nout,  bias=False, **kwargs):
        return _fc( self.bottom, nout,bias,**kwargs)
    @layerinit('bn')
    def bn(self,  in_place=True, **kwarg ):
        return _bn(self.bottom,in_place=in_place)
    @layerinit('scale')
    def scale(self,  in_place=True , **kwarg):
        return  _scale(self.bottom,in_place=in_place)
    @layerinit('relu')
    def relu(self,  in_place=True , **kwarg):
        return  _relu(self.bottom,0,in_place=in_place)
    @layerinit('leaky')
    def leakyrelu(self, negslope, in_place=True, **kwarg):
        return _relu(self.bottom, negslope,in_place=in_place)
    @layerinit('maxpool')
    def maxpool(self, ks, stride,  pad=0, **kwarg ):
        return _maxpool(self.bottom, ks, stride,pad)
    @layerinit('avepool')
    def avepool(self, ks, stride,  pad=0, **kwarg):
        return _avepool(self.bottom, ks, stride, pad)
    @layerinit('maxpoolg')
    def maxpoolglobal(self,**kwarg):
        return _maxpoolglobal(self.bottom)
    @layerinit('avepoolg')
    def avepoolglobal(self,  **kwarg):
        return _avepoolglobal(self.bottom)
    @layerinit('data')
    def input(self,  blobshape,  **kwarg):
        return _input(blobshape)
    @layerinit('elt')
    def eltwise(self, layer, **kwarg):
        layerobj = self.n[layer]   if isinstance(layer, str) else layer
        return _eltwise(self.bottom,layerobj);
    @layerinit('concat')
    def concat(self, layer, axis=1,**kwarg):
        layerobj = self.n[layer]   if isinstance(layer, str) else layer
        return _concat(self.bottom,layerobj,axis=axis);
    @layerinit('softmax')
    def softmaxwithloss(self, score_layer, label_lname, loss_weight=1.0):
        layerobj = self.n[score_layer]   if isinstance(score_layer, str) else score_layer
        return _softmaxwithloss(layerobj, self.n[label_lname], loss_weight=loss_weight)
    @layerinit('acc')
    def accuracy(self, score_layer, label_lname, top_k=1, testonly=False, **kwarg):
        layerobj = self.n[score_layer]   if isinstance(score_layer, str) else score_layer
        return _accuracy(layerobj, self.n[label_lname],top_k,testonly=testonly)
    @layerinit('resize')
    def resize(self, target, **kwargs ):
        layerobj = self.n[target]   if isinstance(target, str) else target
        return _resize(self.bottom, layerobj)
    @layerinit('reorg')    
    def reorg(self, stride=2,reverse=False,**kwargs):
        return _reorg(self.bottom, stride=stride, reverse=reverse)
    def convrelu(self, nout,ks, bias=True, **kwargs):
        nameargs = {k:kwargs.get(k,None) for k in ['prefix','postfix']}
        self.conv(nout,ks, bias=bias,**kwargs)
        return self.relu(**nameargs)
    def fcrelu(self, nout,  bias=True,**kwargs):
        nameargs = {k:kwargs.get(k,None) for k in ['prefix','postfix']}
        self.fc(nout, bias=bias,**kwargs)
        return self.relu(**nameargs)
    def bnscale(self,  in_place=True,**kwargs):
        nameargs = {k:kwargs.get(k,None) for k in ['prefix','postfix']}
        self.bn(in_place=in_place,**kwargs)
        return self.scale(**nameargs)
    def bnscalerelu(self, in_place=True,**kwargs):
        nameargs = {k:kwargs.get(k,None) for k in ['prefix','postfix']}
        self.bn(in_place=in_place,**kwargs)
        self.scale(**nameargs)
        return self.relu(**nameargs)
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
''' #the trick to find output channels from previous linear layer       
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
'''        
