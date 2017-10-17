from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from layerfactory import conv_bn, last_layer, conv
from darknet import DarkNet
import math

class Yolo(object):
    def add_input_data(self, n, num_classes, **kwargs):
        include = {'phase': getattr(caffe_pb2, 'TRAIN')}
        transform_param = {'mirror': True, 
                    'crop_size': 224,
                    'mean_value': [104, 117, 123]}

        source_file = kwargs.get('source', 'train.tsv')

        if 'gpus' in kwargs:
            num_threads = len(kwargs['gpus'].split(','))
        else:
            num_threads = 1
        effective_batch_size = kwargs.get('effective_batch_size', 64.0)
        batch_size = int(math.ceil(effective_batch_size / num_threads))
        tsv_data_param = {'source': source_file, 
                'batch_size': batch_size, 
                'new_height': 256,
                'new_width': 256,
                'col_data': 2,
                'col_label': 1}

        if 'source_label' in kwargs:
            tsv_data_param['source_label'] = kwargs['source_label']

        data_param = {}
        if 'prefetch' in kwargs:
            data_param['prefetch'] = kwargs['prefetch']

        box_data_param = {'jitter': 0.2,
                'hue': 0.1,
                'exposure': 1.5,
                'saturation': 1.5,
                'max_boxes': 30,
                'iter_for_resize': 80,
                'random_min': 416,
                'random_max': 416}

        box_data_param['labelmap'] = kwargs.get('labelmap', 'labelmap.txt')
        
        n.data, n.label = L.TsvBoxData(ntop=2, 
                include=include,
                transform_param=transform_param, 
                tsv_data_param=tsv_data_param,
                box_data_param=box_data_param,
                data_param=data_param)

    def dark_block(self, n, s, nout_dark,stride,  lr, deploy, bn_no_train=False):
        bottom = n.data if s.startswith('dark1') else last_layer(n);
        ks = 1 if s.endswith('_1') else 3
        pad = 0 if ks==1 else 1
        n[s], n[s+'/bn'], n[s+'/scale'] = conv_bn(bottom, ks=ks, stride=stride, 
                pad=pad, bn_no_train=bn_no_train, nout=nout_dark, lr = lr, deploy = deploy)
        n[s+'/leaky'] = L.ReLU(n[s+'/scale'], in_place=True, negative_slope=0.1)

    def before_last_reduce_spatial_resolution(self, n):
        '''
        return the layer name before the last pool
        '''
        tops = n.__dict__['tops'].keys()
        found = None
        for i in xrange(len(tops) - 1, -1, -1):
            if n[tops[i]].fn.type_name == 'Convolution' or \
                    n[tops[i]].fn.type_name == 'Pooling':
                s = n[tops[i]].fn.params.get('stride', 1)
                if s == 2:
                    found = n[tops[i]].fn.inputs[0]
                    break
                elif s != 1:
                    # before the spatial resolution gets decreased by a factor
                    # of 2, if it got decreased by, e.g. 3, we will not handle
                    # this case
                    break
        return found

    def add_body(self, n, lr, num_classes, cnnmodel, 
            deploy=False, cpp_version=False, **kwargs):
        kernal_size = 3
        stride = 1
        pad = 1
        bn_no_train = deploy
        num_output = 1024
        num_extra_convs = kwargs.get('num_extra_convs', 3)
        conv_id = 19
        for i in range(0, num_extra_convs - 1):
            name = 'extra_conv{}'.format(conv_id)
            conv_id = conv_id + 1
            self.dark_block(n, name, num_output, stride, lr, deploy,
                    bn_no_train)
        
        if kwargs.get('add_reorg', True):
            assert num_extra_convs == 3, 'when add_reorg is true, add 3 conv'
            reorg_from = self.before_last_reduce_spatial_resolution(n)
            n['reorg'] = L.Reorg(reorg_from, stride=2)
            n.concat20 = L.Concat(n['reorg'], n['extra_conv20/leaky'])

        if num_extra_convs > 0:
            name = 'extra_conv{}'.format(conv_id)
            self.dark_block(n, name, num_output, stride, lr, deploy,
                    bn_no_train)

        biases = [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
        coords = 4
        num_output = (num_classes + 1 + coords) * len(biases) / 2
        last = last_layer(n)
        n['last_conv'] = conv(last, num_output, ks=1,
                deploy=deploy)
        if not deploy:
            if not kwargs.get('yolo_full_gpu', False):
                n['region_loss'] = L.RegionLoss(n['last_conv'], 
                        n['label'],
                        classes=num_classes,
                        coords=coords,
                        softmax=True,
                        rescore=True,
                        bias_match=True,
                        param={'decay_mult': 0, 'lr_mult': 0},
                        object_scale=5,
                        noobject_scale=1.0,
                        class_scale=1.0,
                        coord_scale=1.0,
                        thresh=0.6,
                        biases=biases)
            else:
                add_yolo_train_loss(n, biases, num_classes)
        else:
            if not kwargs.get('yolo_full_gpu', False):
                n.bbox, n.prob = L.RegionOutput(n['last_conv'], n['im_info'],
                        ntop=2,
                        classes=num_classes,
                        thresh=0.005, # 0.24
                        nms=0.45,
                        biases=biases)
            else:
                add_yolo_test_loss(n, biases, num_classes)

def add_yolo_train_loss(n, biases, num_classes):
    last_top = last_layer(n)
    num_anchor = len(biases) / 2

    n.xy, n.wh, n.obj, n.conf = L.Slice(last_top, 
            ntop=4,
            name='slice_region',
            slice_point=[num_anchor * 2, num_anchor * 4,
                num_anchor * 5])
    n.sigmoid_xy = L.Sigmoid(n.xy, in_place=True)
    n.sigmoid_obj = L.Sigmoid(n.obj, in_place=True)
    regionTargetOutput = L.RegionTarget(n.sigmoid_xy, n.wh, n.sigmoid_obj,
            n.label, 
            name='region_target',
            param={'decay_mult': 0, 'lr_mult': 0},
            ntop=6, 
            propagate_down=[False, False, False, False],
            biases=biases)
    n.t_xy, n.t_wh, n.t_xywh_weight, n.t_o_obj, n.t_o_noobj, n.t_label = regionTargetOutput
    n.xy_loss = L.EuclideanLoss(n.xy, n.t_xy, n.t_xywh_weight, 
            propagate_down=[True, False, False], 
            loss_weight=1)
    n.wh_loss = L.EuclideanLoss(n.wh, n.t_wh, n.t_xywh_weight, 
            propagate_down=[True, False, False], 
            loss_weight=1)
    n.o_obj_loss = L.EuclideanLoss(n.obj, n.t_o_obj, 
            propagate_down=[True, False], loss_weight=5)
    n.o_noobj_loss = L.EuclideanLoss(n.obj, n.t_o_noobj, 
            propagate_down=[True, False], loss_weight=1)
    n.reshape_t_label = L.Reshape(n.t_label, shape={'dim': [-1, 1, 0, 0]})
    n.reshape_conf = L.Reshape(n.conf, shape={'dim': [-1, num_classes, 0, 0]})
    n.softmax_loss = L.SoftmaxWithLoss(n.reshape_conf, n.reshape_t_label,
            propagate_down=[True, False],
            loss_weight=num_anchor, loss_param={'ignore_label': -1, 'normalization':
                P.Loss.BATCH_SIZE})

def add_yolo_test_loss(n, biases, num_classes):
    last_top = last_layer(n)
    num_anchor = len(biases) / 2

    n.xy, n.wh, n.obj, n.conf = L.Slice(last_top, 
            ntop=4,
            name='slice_region',
            slice_point=[num_anchor * 2, num_anchor * 4, num_anchor * 5])
    n.sigmoid_xy = L.Sigmoid(n.xy, in_place=True)
    n.sigmoid_obj = L.Sigmoid(n.obj, in_place=True)
    n.reshape_conf = L.Reshape(n.conf, shape={'dim': [-1, num_classes, 0, 0]})
    n.softmax_conf = L.Softmax(n.reshape_conf)
    n.bbox, n.prob = L.RegionPrediction(n.sigmoid_xy, n.wh, n.sigmoid_obj,
            n.softmax_conf, n.im_info, ntop=2, 
            thresh=0.005, # 0.24
            biases=biases)

