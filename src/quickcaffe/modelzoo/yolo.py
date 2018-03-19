from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import numpy as np
from layerfactory import conv_bn, last_layer, conv
from darknet import DarkNet
import math
import logging

class Yolo(object):
    def __init__(self):
        self.batch_size = []

    def add_input_data(self, n, num_classes, **kwargs):
        #include = {'phase': getattr(caffe_pb2, 'TRAIN')}
        transform_param = {'mirror': kwargs.get('yolo_mirror', True), 
                    'crop_size': 224,
                    'mean_value': [104, 117, 123]}
        data_param = {}
        if 'prefetch' in kwargs:
            data_param['prefetch'] = kwargs['prefetch']
        net_input_size_min = kwargs.get('net_input_size_min', 416)
        net_input_size_max = kwargs.get('net_input_size_max', 416)
        
        jitter = kwargs.get('yolo_jitter', 0.2)
        hue = kwargs.get('yolo_hue', 0.1)
        exposure = kwargs.get('yolo_exposure', 1.5)
        saturation = kwargs.get('yolo_saturation', 1.5)

        box_data_param = {'jitter': jitter, 
                'hue': hue,
                'exposure': exposure,
                'saturation': saturation,
                'max_boxes': 30,
                'iter_for_resize': 80,
                'random_min': net_input_size_min,
                'random_max': net_input_size_max}
        if 'yolo_random_scale_min' in kwargs:
            box_data_param['random_scale_min'] = kwargs['yolo_random_scale_min']
        if 'yolo_random_scale_max' in kwargs:
            box_data_param['random_scale_max'] = kwargs['yolo_random_scale_max']
        if 'rotate_max' in kwargs:
            box_data_param['rotate_max'] = kwargs['rotate_max']
        box_data_param['labelmap'] = kwargs.get('labelmap', 'labelmap.txt')

        source_files = kwargs.get('sources', ['train.tsv'])
        assert 'source' not in kwargs, 'not supported, use sources: a list'
        if 'gpus' in kwargs:
            num_threads = len(kwargs['gpus'])
        else:
            num_threads = 1

        # Only with -fg set use the new layer structure
        with_new_layers = kwargs.get('yolo_full_gpu', False) and kwargs.get('target_synset_tree', False)
        effective_batch_size = kwargs.get('effective_batch_size', 64.0)
        batch_size = int(math.ceil(effective_batch_size / num_threads))
        assert batch_size > 0
        if len(source_files) == 1:
            assert batch_size > 0
            self.batch_size.append(batch_size)
            tsv_data_param = {'batch_size': batch_size, 
                    'new_height': 256,
                    'new_width': 256,
                    'col_data': 2,
                    'col_label': 1}

            tsv_data_param['source'] = source_files[0]

            if 'source_labels' in kwargs and kwargs['source_labels']:
                assert len(kwargs['source_labels']) == 1
                tsv_data_param['source_label'] = kwargs['source_labels'][0]

            if 'source_shuffles' in kwargs and kwargs['source_shuffles']:
                assert len(kwargs['source_shuffles']) == 1
                tsv_data_param['source_shuffle'] = kwargs['source_shuffles'][0]
            
            n.data, n.label = L.TsvBoxData(ntop=2,
                    transform_param=transform_param, 
                    tsv_data_param=tsv_data_param,
                    box_data_param=box_data_param,
                    data_param=data_param)
        else:
            data_weights = kwargs['data_batch_weights']
            assert len(data_weights) == len(source_files)
            np_weights = np.asarray(data_weights)
            np_weights = np_weights * batch_size / np.sum(np_weights)
            data_blobs, label_blobs = [], []
            for i in xrange(len(source_files)):
                w = float(np_weights[i])
                assert w > 0
                assert w.is_integer()
                self.batch_size.append(w)
                tsv_data_param = {'batch_size': int(w),
                        'new_height': 256,
                        'new_width': 256,
                        'col_data': 2,
                        'col_label': 1}

                tsv_data_param['source'] = source_files[i]

                if 'source_labels' in kwargs and kwargs['source_labels']:
                    if len(kwargs['source_labels']) != 0:
                        assert len(kwargs['source_labels']) == len(source_files)
                        tsv_data_param['source_label'] = kwargs['source_labels'][i]

                if 'source_shuffles' in kwargs and kwargs['source_shuffles']:
                    if len(kwargs['source_shuffles']) != 0:
                        assert len(kwargs['source_shuffles']) == len(source_files)
                        tsv_data_param['source_shuffle'] = kwargs['source_shuffles'][i]

                label_name = 'label%s' % (i if i > 0 or not with_new_layers else '')
                n['data' + str(i)], n[label_name] = L.TsvBoxData(ntop=2,
                        #include=include,
                        transform_param=transform_param, 
                        tsv_data_param=tsv_data_param,
                        box_data_param=box_data_param,
                        data_param=data_param)
                data_blobs.append(n['data' + str(i)])
                if not with_new_layers:
                    label_blobs.append(n[label_name])

            n.data = L.Concat(*data_blobs, axis=0)
            if not with_new_layers:
                n.label = L.Concat(*label_blobs, axis=0)
                return

            assert len(self.batch_size) == 2
            # assume second one is always nobb
            n.label_nobb_xywh, n.label_nobb, n.label_nobb_multi = L.Slice(n.label1,
                                                ntop=3,
                                                name='slice_label',
                                                slice_point=[4, 5])
            n.silence_label_nobb = L.Silence(n.label_nobb_xywh, n.label_nobb_multi,
                                             ntop=0, name='ignore_label_nobb')

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
        extra_train_param, extra_test_param = get_region_param(**kwargs)
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
                        bias_match=True,
                        param={'decay_mult': 0, 'lr_mult': 0},
                        thresh=0.6,
                        biases=biases,
                        **extra_train_param)
            else:
                add_yolo_train_loss(n, biases, num_classes, self.batch_size, **kwargs)
        else:
            if not kwargs.get('yolo_full_gpu', False):
                n.bbox, n.prob = L.RegionOutput(n['last_conv'], n['im_info'],
                        ntop=2,
                        classes=num_classes,
                        thresh=0.005, # 0.24
                        nms=0.45,
                        biases=biases,
                        **extra_test_param)
            else:
                add_yolo_test_loss(n, biases, num_classes, kwargs.get('target_synset_tree'),
                                   kwargs.get('class_specific_nms', True))

def add_yolo_train_loss_bb_only(n, bb_top, biases, num_classes, tree_file,
        weight_scale=1,
        **kwargs):
    num_anchor = len(biases) / 2
    n.xy, n.wh, n.obj, n.conf = L.Slice(bb_top,
            ntop=4,
            name='slice_region',
            slice_point=[num_anchor * 2, num_anchor * 4, num_anchor * 5])
    n.sigmoid_xy = L.Sigmoid(n.xy, in_place=True)
    n.sigmoid_obj = L.Sigmoid(n.obj, in_place=True)
    n.t_xy, n.t_wh, n.t_xywh_weight, n.t_o_obj, n.t_o_noobj, n.t_label = L.RegionTarget(
        n.sigmoid_xy, n.wh, n.sigmoid_obj,
        n.label,
        name='region_target',
        param={'decay_mult': 0, 'lr_mult': 0},
        ntop=6,
        propagate_down=[False, False, False, False],
        biases=biases)
    n.xy_loss = L.EuclideanLoss(n.xy, n.t_xy, n.t_xywh_weight,
                                propagate_down=[True, False, False],
                                loss_weight=1 * weight_scale)
    n.wh_loss = L.EuclideanLoss(n.wh, n.t_wh, n.t_xywh_weight,
                                propagate_down=[True, False, False],
                                loss_weight=1 * weight_scale)
    n.o_obj_loss = L.EuclideanLoss(n.obj, n.t_o_obj,
                                   propagate_down=[True, False],
                                   loss_weight=5 * weight_scale)
    n.o_noobj_loss = L.EuclideanLoss(n.obj, n.t_o_noobj,
            propagate_down=[True, False], loss_weight=1 * weight_scale)

    norm_type = P.Loss.BATCH_SIZE
    loss_weight = num_anchor if not tree_file else 1 * weight_scale
    if kwargs.get('yolo_softmax_norm_by_valid', False):
        norm_type = P.Loss.VALID
        assert not tree_file, 'not tested yet'
        loss_weight = 1

    if not tree_file:
        # flat structure (uses softmax)
        n.reshape_t_label = L.Reshape(n.t_label, shape={'dim': [-1, 1, 0, 0]})
        n.reshape_conf = L.Reshape(n.conf, shape={'dim': [-1, num_classes, 0, 0]})
        n.softmax_loss = L.SoftmaxWithLoss(n.reshape_conf, n.reshape_t_label,
                                           propagate_down=[True, False],
                                           loss_weight=loss_weight,
                                           loss_param={
                                               'ignore_label': -1,
                                               'normalization': norm_type 
                                            })
        return

    n.reshape_conf = L.Reshape(n.conf, axis=1, num_axes=1, shape={'dim': [num_classes, num_anchor]})
    n.softmaxtree_loss = L.SoftmaxTreeWithLoss(n.reshape_conf, n.t_label,
                                           propagate_down=[True, False],
                                           loss_weight=1 * weight_scale,
                                           loss_param={
                                               'ignore_label': -1,
                                               'normalization': P.Loss.BATCH_SIZE
                                           },
                                           softmaxtree_param={
                                               'tree': tree_file
                                           })

def add_yolo_train_loss(n, biases, num_classes, batch_size, 
        **kwargs):
    tree_file = kwargs.get('target_synset_tree')
    last_top = last_layer(n)

    if len(batch_size) == 1 or not tree_file:
        add_yolo_train_loss_bb_only(n, last_top, biases, num_classes,
                tree_file, **kwargs)
        return
    assert len(batch_size) == 2
    n.conv_bb, n.conv_no_bb = L.Slice(last_top,
                                      ntop=2,
                                      name='slice_batch',
                                      axis=0,
                                      slice_point=[int(batch_size[0])]
                                      )
    weight_bb = float(batch_size[0]) / np.sum(batch_size)
    weight_nobb = float(batch_size[1]) / np.sum(batch_size)
    add_yolo_train_loss_bb_only(n, n.conv_bb, biases, num_classes, tree_file,
            weight_scale=weight_bb,
            **kwargs)

    num_anchor = len(biases) / 2

    n.xywh_nobb, n.obj_nobb, n.conf_nobb = L.Slice(n.conv_no_bb,
                                                   ntop=3,
                                                   name='slice_region_nobb',
                                                   slice_point=[num_anchor * 4, num_anchor * 5])
    n.silence_nobb = L.Silence(n.xywh_nobb,
                               ntop=0, name='ignore_nobb')
    n.sigmoid_obj_nobb = L.Sigmoid(n.obj_nobb, in_place=True)


    n.reshape_conf_nobb = L.Reshape(n.conf_nobb, axis=1, num_axes=1,
                                    shape={'dim': [num_classes, num_anchor]})
    n.softmaxtree_loss_nobb, n.obj_index = L.SoftmaxTreeWithLoss(
        n.reshape_conf_nobb, n.label_nobb, n.sigmoid_obj_nobb,
        ntop=2,
        propagate_down=[True, False, False],
        loss_weight=[weight_nobb, 0],
        loss_param={
            'normalization': P.Loss.BATCH_SIZE
        },
        softmaxtree_param={
            'tree': tree_file
        },
        softmaxtree_loss_param={
            'with_objectness': True
        })

    extra_index_weight = kwargs.get('yolo_index_threshold_loss_extra_weight', 0)
    n.obj_loss_nobb = L.IndexedThresholdLoss(
        n.sigmoid_obj_nobb, n.obj_index,
        loss_weight=weight_nobb*extra_index_weight,
        propagate_down=[True, False])


def add_yolo_test_loss(n, biases, num_classes, tree_file, class_specific_nms):
    last_top = last_layer(n)
    num_anchor = len(biases) / 2

    n.xy, n.wh, n.obj, n.conf = L.Slice(last_top, 
            ntop=4,
            name='slice_region',
            slice_point=[num_anchor * 2, num_anchor * 4, num_anchor * 5])
    n.sigmoid_xy = L.Sigmoid(n.xy, in_place=True)
    n.sigmoid_obj = L.Sigmoid(n.obj, in_place=True)
    if tree_file:
        n.reshape_conf = L.Reshape(n.conf, axis=1, num_axes=1,
                                   shape={'dim': [num_classes, num_anchor]})
        n.softmaxtree_conf = L.SoftmaxTree(n.reshape_conf,
                                           softmaxtree_param={
                                               'tree': tree_file
                                           })
        n.top_preds = L.SoftmaxTreePrediction(n.softmaxtree_conf,
                                              n.sigmoid_obj,
                                              softmaxtreeprediction_param={
                                                'tree': tree_file,
                                                'threshold': 0.5
                                               })
        n.bbox = L.YoloBBs(n.sigmoid_xy, n.wh, n.im_info,
                           yolobbs_param={
                               'biases': biases,
                               })
        if class_specific_nms:
            # per-class NMS on the classes (leave the last column unchanged for compatibility)
            n.nms_prob = L.NMSFilter(n.bbox, n.top_preds,
                                     nmsfilter_param={
                                         'classes': num_classes,
                                         'threshold': 0.45,
                                         'pre_threshold': 0.005  # 0.24
                                     })
        else:
            # class-independent NMS (on the last channel that holds the max)
            n.nms_prob = L.NMSFilter(n.bbox, n.top_preds,
                                     nmsfilter_param={
                                         'classes': 1,
                                         'first_class': num_classes,
                                         'threshold': 0.45,
                                         'pre_threshold': 0.005  # 0.24
                                     })
        # just move the axis to the end for compatibility
        n.prob = L.YoloEvalCompat(n.nms_prob,
                                  yoloevalcompat_param={
                                      'append_max': False
                                  })
        return
    n.reshape_conf = L.Reshape(n.conf, shape={'dim': [-1, num_classes, 0, 0]})
    n.softmax_conf = L.Softmax(n.reshape_conf)
    n.bbox, n.prob = L.RegionPrediction(n.sigmoid_xy, n.wh, n.sigmoid_obj,
                                        n.softmax_conf, n.im_info, ntop=2,
                                        thresh=0.005,  # 0.24
                                        class_specific_nms=class_specific_nms,
                                        biases=biases)


def get_region_param(**kwargs):
    extra_train_param, extra_test_param = {}, {}
    train_test_keys = []

    train_keys = ['object_scale', 
            'noobject_scale',
            'class_scale',
            'rescore', 
            'coord_scale']

    test_keys = ['nms']

    train_keys.extend(train_test_keys)
    test_keys.extend(train_test_keys)

    for key in test_keys:
        yolo_key = 'yolo_' + key
        if yolo_key in kwargs:
            extra_test_param[key] = kwargs[yolo_key]

    for key in train_keys:
        yolo_key = 'yolo_' + key
        if yolo_key in kwargs:
            extra_train_param[key] = kwargs[yolo_key]

    if 'target_synset_tree' in kwargs:
        extra_train_param['tree'] = kwargs['target_synset_tree']
        extra_test_param['tree'] = kwargs['target_synset_tree']

    if 'region_layer_map' in kwargs:
        extra_test_param['map'] = kwargs['region_layer_map']
    return extra_train_param, extra_test_param

