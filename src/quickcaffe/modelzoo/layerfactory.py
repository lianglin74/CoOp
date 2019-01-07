import os.path as op
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

def last_layer(n):
    return n.__dict__['tops'][list(n.__dict__['tops'].keys())[-1]]

def deconv(bottom, nout, ks=3, stride=1, pad=0, lr=1, weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), deploy=True):
    if deploy:
        _deconv = L.Deconvolution(bottom,convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=pad))
    else:
        param = [dict(lr_mult=lr, decay_mult=1), dict(lr_mult=2 * lr, decay_mult=0) ]
        _deconv = L.Deconvolution(bottom,convolution_param=dict(num_output=nout, kernel_size=ks, stride=stride, pad=pad, weight_filler=weight_filler, bias_filler=bias_filler), param = param)
        
    return _deconv

def conv(bottom, nout, ks=3, stride=1, pad=0, lr=1, weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), deploy=True):
    if deploy:
        _conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                num_output=nout, pad=pad)
    else:
        param = [dict(lr_mult=lr, decay_mult=1), dict(lr_mult=2 * lr, decay_mult=0)]
        _conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                num_output=nout, pad=pad, param = param, weight_filler=weight_filler, bias_filler=bias_filler)
    return _conv

def conv_nobias(bottom, nout, ks=3, stride=1, pad=0, lr=1,
        weight_filler=dict(type='msra'), deploy=True, group=0):
    if deploy:
        if group > 0:
            _conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                    num_output=nout, pad=pad, bias_term=False, group=group)
        else:
            _conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                    num_output=nout, pad=pad, bias_term=False)
    else:
        param = [dict(lr_mult=lr, decay_mult=1)]
        if group > 0:
            _conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                    num_output=nout, pad=pad, bias_term=False, param = param,
                    weight_filler=weight_filler, group=group)
        else:
            _conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                    num_output=nout, pad=pad, bias_term=False, param = param, weight_filler=weight_filler)
    return _conv

def fc(bottom, nout, lr=1, weight_filler=dict(type='msra'),
        bias_filler=dict(type='constant', value=0), deploy=True, **kwargs):
    if deploy:
        _fc = L.InnerProduct(bottom, num_output = nout)
    else:
        _fc = L.InnerProduct(bottom, num_output = nout,
            param=[dict(lr_mult=lr, decay_mult=1), dict(lr_mult=2 * lr, decay_mult=0)],
            weight_filler=weight_filler, bias_filler=bias_filler)
    return _fc

def fc_nobias(bottom, nout, lr=1, weight_filler=dict(type='msra'), deploy=True):
    if deploy:
        _fc = L.InnerProduct(bottom, num_output = nout)
    else:
        _fc = L.InnerProduct(bottom, num_output = nout, bias_term=False,
            param=[dict(lr_mult=lr, decay_mult=1)],
            weight_filler=weight_filler)
    return _fc

def bn(bottom, in_place=True, bn_no_train=False, deploy=True):
    if deploy:
        _bn = L.BatchNorm(bottom, in_place=True)
    else:
        if bn_no_train:
            # used for faster-rcnn
            _bn = L.BatchNorm(bottom, batch_norm_param=dict(use_global_stats=True), in_place=True,
                              param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
        else:
            _bn = L.BatchNorm(bottom, in_place=True,
                              param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    return _bn

def fc_bn(bottom, nout, lr=1, deploy=True):
    _fc = fc(bottom, nout, lr=1, deploy=deploy)
    _bn = bn(_fc, in_place=True, deploy=deploy)
    scale = L.Scale(_bn, in_place=True, scale_param=dict(bias_term=True))
    return _fc, _bn, scale

def conv_bn(bottom, nout, ks=3, stride=1, pad=0, lr=1, bn_no_train=False,
        deploy=True, group=0):
    _conv = conv_nobias(bottom, nout, ks=ks, stride=stride, pad=pad, lr=lr,
            deploy=deploy, group=group)
    _bn = bn(_conv, in_place=True, bn_no_train=bn_no_train, deploy=deploy)
    scale = L.Scale(_bn, in_place=True, scale_param=dict(bias_term=True))
    return _conv, _bn, scale

def max_pool(bottom, ks, stride = 1, pad = 0):
    return L.Pooling(bottom, pool=P.Pooling.MAX, stride = stride, kernel_size = ks, pad = pad)

def ave_pool(bottom, ks, stride = 1, pad = 0):
    return L.Pooling(bottom, pool=P.Pooling.AVE, stride = stride, kernel_size = ks, pad = pad)

def ave_pool_global(bottom):
    return L.Pooling(bottom, pooling_param = dict(pool = P.Pooling.AVE, global_pooling = True))

# for caffenet
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, lr=1, weight_filler=dict(type='msra'), bias_filler=dict(type="constant", value=0), deploy=True):
    if deploy:
        if group != 1:
            _conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                    num_output=nout, pad=pad, group=group)
        else: # simply not to pass the default group parameter
            _conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                    num_output=nout, pad=pad)
    else:
        param = [dict(lr_mult=lr, decay_mult=lr), dict(lr_mult=lr*2, decay_mult=0)]
        if group != 1:
            _conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                    num_output=nout, pad=pad, group=group, param=param, weight_filler=weight_filler, bias_filler=bias_filler)
        else: # simply not to pass the default group parameter
            _conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                    num_output=nout, pad=pad, param=param, weight_filler=weight_filler, bias_filler=bias_filler)

    return _conv, L.ReLU(_conv, in_place=True)

# for caffenet
def fc_relu(bottom, nout, lr=1, weight_filler=dict(type="msra"), bias_filler=dict(type="constant", value=0), deploy=True):
    _fc = fc(bottom, nout, weight_filler=weight_filler, bias_filler=bias_filler, deploy=deploy)
    return _fc, L.ReLU(_fc, in_place=True)

def tsv_data(ntop, transform_param, tsv_data_param, phase='TRAIN'):
    '''
    A generic function for creating a tsv data layer.
    transform_param and tsv_data_param need to be prepared outside to include the required fields in tsv data layer.
    This is to leave the flexibility to caller and avoid to complicate the function interface
    An example of the two params is as follows:
      transform_param {
        mirror: true
        crop_size: 224
        mean_file: "imagenet_mean.binaryproto"
      }
      tsv_data_param {
        source: "train.tsv"
        source_label: "train.label.tsv"
        source_shuffle: "train.label.shuffle"
        batch_size: 128
        new_height: 256
        new_width: 256
        col_data: 2
        col_label: 1
      }
    '''
    data, label = L.TsvData(ntop, transform_param, tsv_data_param, include=dict(phase=getattr(caffe_pb2, phase)))

    return data, label

def tsv_data_str(ntop, transform_param, tsv_data_param, phase = 'TRAIN'):
    ''' generate string of a data layer, which can be inserted to prototxt string later'''
    n = caffe.NetSpec()
    n.data, n.label = tsv_data(ntop, transform_param, tsv_data_param, phase)
    return str(n.to_proto())

def tsv_imagenet(tsv_data, mean_file, mean_value, crop_size, batch_size = 256, new_image_size = (256, 256), mirror = True, phase = 'TRAIN' ):
    '''
    Create tsv data layer for ImageNet data.
    if mean_file is provided, mean_value will be ignored
    '''
    if len(mean_file) > 0:
        transform_param=dict(crop_size=crop_size, mean_file=mean_file, mirror=mirror)
    else:
        transform_param=dict(crop_size=crop_size, mean_value=mean_value, mirror=mirror)

    tsv_data_param=dict(source=tsv_data, source_shuffle=op.splitext(tsv_data)[0] + '.shuffle',
                        batch_size=batch_size, col_data=2, col_label=1, new_width=new_image_size[0], new_height=new_image_size[1])
    data, label = L.TsvData(ntop=2, transform_param=transform_param, tsv_data_param=tsv_data_param, include=dict(phase=getattr(caffe_pb2, phase)))

    return data, label

def tsv_imagenet_str(tsv_data, mean_file, mean_value, crop_size, batch_size = 256, new_image_size = (256, 256), mirror = True, phase = 'TRAIN' ):
    ''' generate string of an imagenet data layer, which can be inserted to prototxt string later'''
    n = caffe.NetSpec()
    n.data, n.label = tsv_imagenet(tsv_data, mean_file, mean_value, crop_size, batch_size, new_image_size, mirror, phase)
    return str(n.to_proto())

def lmdb_imagenet(lmdb_data, format, mean_file, mean_value, crop_size, batch_size = 256, mirror = True, phase = 'TRAIN' ):
    '''
    Create lmdb data layer for ImageNet data.
    if mean_file is provided, mean_value will be ignored
    '''
    if len(mean_file) > 0:
        transform_param=dict(crop_size=crop_size, mean_file=mean_file, mirror=mirror)
    else:
        transform_param=dict(crop_size=crop_size, mean_value=mean_value, mirror=mirror)

    backend = P.Data.LMDB if format=='lmdb' else P.Data.LEVELDB
    data_param=dict(source=lmdb_data, backend=backend, batch_size=batch_size)
    data, label = L.Data(ntop=2, transform_param=transform_param, data_param=data_param, include=dict(phase=getattr(caffe_pb2, phase)))

    return data, label

def lmdb_imagenet_str(lmdb_data, format, mean_file, mean_value, crop_size, batch_size = 256, mirror = True, phase = 'TRAIN' ):
    ''' generate string of an imagenet data layer, which can be inserted to prototxt string later'''
    n = caffe.NetSpec()
    n.data, n.label = lmdb_imagenet(lmdb_data, format, mean_file, mean_value, crop_size, batch_size, mirror, phase)
    return str(n.to_proto())
