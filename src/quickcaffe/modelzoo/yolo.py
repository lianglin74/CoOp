import copy
import logging
import json
import numpy as np
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from .layerfactory import conv_bn, last_layer, conv
from .darknet import DarkNet
import math


def calc_out_idx(outputs, num_anchor):
    '''
    outputs: xy, wh, obj, cls
    layout: xxxyyy, wwwhhh, ooo, ccc (num_anchor = 3)
    '''
    assert len(outputs) == 4
    assert outputs[2] == num_anchor
    assert outputs[0] == 2 * num_anchor 
    assert outputs[1] == 2 * num_anchor
    coords = (outputs[0] + outputs[1]) / num_anchor
    result = [None] * sum(outputs)
    anchor_size = len(result) / num_anchor
    classes = outputs[-1] / num_anchor
    for i in range(num_anchor):
        # x
        result[i] = i * anchor_size

    for i in range(num_anchor, 2 * num_anchor):
        # y
        result[i] = (i - num_anchor) * anchor_size + 1

    for i in range(2 * num_anchor, 3 * num_anchor):
        # w
        result[i] = (i - 2 * num_anchor) * anchor_size + 2

    for i in range(3 * num_anchor, 4 * num_anchor):
        # h
        result[i] = (i - 3 * num_anchor) * anchor_size + 3

    for i in range(4 * num_anchor, 5 * num_anchor):
        # o
        result[i] = (i - 4 * num_anchor) * anchor_size + 4

    for i in range(5 * num_anchor, len(result)):
        idx_anchor = (i - 5 * num_anchor) / classes
        j = (i - 5 * num_anchor) % classes
        result[i] = idx_anchor * anchor_size + 5 + j

    return result



def add_dark_block(n, bottom, s, nout_dark,
        stride,  lr, deploy,
        bn_no_train=False, ks=None):
    if ks is None:
        ks = 1 if s.endswith('_1') else 3
    pad = 0 if ks==1 else 1
    n[s], n[s+'/bn'], n[s+'/scale'] = conv_bn(bottom, ks=ks, stride=stride, 
            pad=pad, bn_no_train=bn_no_train, nout=nout_dark, lr = lr, deploy = deploy)
    n[s+'/leaky'] = L.ReLU(n[s+'/scale'], in_place=True, negative_slope=0.1)
    return n[s + '/leaky']

class Yolo(object):
    def __init__(self):
        self.batch_size = []
    def add_input_data(self, n, num_classes, **kwargs):
        if type(num_classes) == list:
            assert len(num_classes) == 1
            num_classes = num_classes[0]
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
        if 'yolo_fix_offset' in kwargs:
            box_data_param['fix_offset'] = kwargs['yolo_fix_offset']
        if 'incorporate_at_least_one_box' in kwargs:
            box_data_param['incorporate_at_least_one_box'] = \
                kwargs['incorporate_at_least_one_box']
        if 'scale_constrained_by_one_box_area_min' in kwargs:
            box_data_param['scale_constrained_by_one_box_area_min'] = \
                    kwargs['scale_constrained_by_one_box_area_min']
        if 'scale_constrained_by_one_box_area' in kwargs:
            box_data_param['scale_constrained_by_one_box_area'] = \
                    kwargs['scale_constrained_by_one_box_area']
        assert len(kwargs['labelmap']) == 1
        box_data_param['labelmap'] = kwargs['labelmap'][0]

        if 'yolo_fixed_target' in kwargs:
            box_data_param['fixed_target'] = kwargs['yolo_fixed_target']

        if 'scale_relative_input' in kwargs:
            box_data_param['scale_relative_input'] = kwargs['scale_relative_input']

        if 'cutout_prob' in kwargs:
            box_data_param['cutout_prob'] = kwargs['cutout_prob']

        source_files = kwargs.get('sources', ['train.tsv'])

        if 'box_data_param_weightss' in kwargs:
            box_data_paramss = []
            for curr_data_box_weights in kwargs['box_data_param_weightss']:
                box_data_params = []
                for curr_data_box_weight in curr_data_box_weights:
                    p = copy.deepcopy(box_data_param)
                    p['weight'] = curr_data_box_weight
                    box_data_params.append(p)
                box_data_paramss.append(box_data_params)
        else:
            box_data_paramss = []
            for _ in source_files:
                box_data_paramss.append([copy.deepcopy(box_data_param)])
        
        if 'rotate_max' in kwargs:
            assert len(kwargs['rotate_max']) == len(box_data_paramss)
            for box_data_params, rotate_maxs in zip(box_data_paramss,
                    kwargs['rotate_max']):
                for box_data_param, rotate_max in zip(box_data_params,
                        rotate_maxs):
                    box_data_param['rotate_max'] = rotate_max

        if 'rotate_with_90' in kwargs:
            assert len(kwargs['rotate_with_90']) == len(box_data_paramss)
            for box_data_params, rotate_with_90s in zip(box_data_paramss,
                    kwargs['rotate_with_90']):
                for box_data_param, rotate_with_90 in zip(box_data_params,
                        rotate_with_90s):
                    box_data_param['rotate_with_90'] = rotate_with_90
        if 'tsv_box_max_samples' in kwargs:
            assert len(kwargs['tsv_box_max_samples']) == len(box_data_paramss)
            for box_data_params, max_sampless in zip(box_data_paramss,
                    kwargs['tsv_box_max_samples']):
                for box_data_param, max_samples in zip(box_data_params,
                        max_sampless):
                    box_data_param['max_samples'] = max_samples

        assert 'source' not in kwargs, 'not supported, use sources: a list'
       
        if 'gpus' in kwargs:
            num_threads = len(kwargs['gpus'])
        else:
            num_threads = 1
        # Only with -fg set use the new layer structure
        with_new_layers = kwargs.get('yolo_full_gpu', False) and kwargs.get('target_synset_tree', False)
        effective_batch_size = kwargs.get('effective_batch_size', 64.0)
        iter_size = kwargs.get('iter_size', 1)
        batch_size = int(math.ceil(effective_batch_size / num_threads /
            iter_size))
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
                assert len(source_files[0]) == len(kwargs['source_labels'][0])

            if 'source_shuffles' in kwargs and kwargs['source_shuffles']:
                assert len(kwargs['source_shuffles']) == 1
                tsv_data_param['source_shuffle'] = kwargs['source_shuffles'][0]
            
            n.data, n.label = L.TsvBoxData(ntop=2, 
                    transform_param=transform_param, 
                    tsv_data_param=tsv_data_param,
                    box_data_param=box_data_paramss[0],
                    data_param=data_param)
        else:
            data_weights = kwargs['data_batch_weights']
            assert len(data_weights) == len(source_files)
            np_weights = np.asarray(data_weights)
            np_weights = np_weights * batch_size / np.sum(np_weights)
            self.batch_weights = np_weights
            data_blobs, label_blobs = [], []
            for i in range(len(source_files)):
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
                        box_data_param=box_data_paramss[i],
                        data_param=data_param)
                data_blobs.append(n['data' + str(i)])
                if not with_new_layers:
                    label_blobs.append(n['label' + str(i)])

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

    def dark_block(self, n, bottom, s, nout_dark,stride,  lr, deploy,
            bn_no_train=False, ks=None, group=0):
        if ks is None:
            ks = 1 if s.endswith('_1') else 3
        pad = 0 if ks==1 else 1
        n[s], n[s+'/bn'], n[s+'/scale'] = conv_bn(bottom, ks=ks, stride=stride, 
                pad=pad, bn_no_train=bn_no_train, nout=nout_dark, lr = lr,
                deploy = deploy, group=group)
        n[s+'/leaky'] = L.ReLU(n[s+'/scale'], in_place=True, negative_slope=0.1)
        return n[s + '/leaky']

    def before_last_reduce_spatial_resolution(self, n):
        '''
        return the layer name before the last pool
        '''
        tops = n.__dict__['tops'].keys()
        found = None
        for i in range(len(tops) - 1, -1, -1):
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

    def _add_extra_conv_reorg(self, n, bottom, num_extra_convs, add_reorg, 
            bn_no_train, extra_conv_channels, stride, lr, deploy,
            suffix='', 
            extra_conv_kernel=None,
            extra_conv_groups=None):
        conv_id = 19
        last_top = bottom
        extra_conv_kernel = extra_conv_kernel if extra_conv_kernel else [3]*num_extra_convs
        # 0 is by default
        extra_conv_groups = extra_conv_groups if extra_conv_groups else [0]*num_extra_convs
        for i in range(0, num_extra_convs - 1):
            name = 'extra_conv{}{}'.format(conv_id, suffix)
            conv_id = conv_id + 1
            last_top = self.dark_block(n, last_top, name,
                    extra_conv_channels[i], stride, lr, deploy,
                    bn_no_train, extra_conv_kernel[i], extra_conv_groups[i])
            
        if add_reorg:
            last = last_layer(n)
            reorg_from = self.before_last_reduce_spatial_resolution(n)
            reorg_name = 'reorg{}'.format(suffix)
            n[reorg_name] = L.Reorg(reorg_from, stride=2)
            concat_name = 'concat20{}'.format(suffix)
            n[concat_name] = L.Concat(n[reorg_name], last)
            last_top = n[concat_name]

        if num_extra_convs > 0:
            name = 'extra_conv{}{}'.format(conv_id, suffix)
            last_top = self.dark_block(n, last_top, name,
                    extra_conv_channels[num_extra_convs - 1], stride, lr, deploy,
                    bn_no_train, extra_conv_kernel[num_extra_convs - 1],
                    extra_conv_groups[num_extra_convs - 1])
        return last_top 

    def add_body(self, n, lr, num_classes, cnnmodel, 
            deploy=False, cpp_version=False, **kwargs):
        if type(num_classes) == list:
            assert len(num_classes) == 1
            num_classes = num_classes[0]
        extra_train_param, extra_test_param = get_region_param(**kwargs)

        last_conv_bias = kwargs.get('yolo_last_conv_bias', True)

        background_class = 1 if kwargs.get('yolo_background_class', False) \
                else 0
        num_classes = num_classes + background_class

        coords = 0 
        if 'yolo_multibin_wh_count' in kwargs:
            coords = 2 * kwargs['yolo_multibin_wh_count']
        else:
            coords = coords + 2

        if 'yolo_multibin_xy_count' in kwargs:
            coords = coords + 2 * kwargs['yolo_multibin_xy_count']
        else:
            coords = coords + 2

        if 'multibin_wh' in kwargs and kwargs['multibin_wh']:
            biases = [6.5, 6.5]
            num_anchor = 1
        else:
            if 'anchor_bias' in kwargs and kwargs['anchor_bias']:
                biases = kwargs['anchor_bias']
                num_anchor = len(biases) / 2
                assert len(biases) % 2 == 0
            else:
                num_anchor = kwargs.get('num_anchor', 5)
                if num_anchor == 5:
                    biases = [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
                elif num_anchor == 3:
                    biases = [0.77871, 1.14074, 3.00525, 4.31277, 9.22725, 9.61974]
                else:
                    raise ValueError('invalid anchor {}'.format(num_anchor))
        
        residual_loss = kwargs.get('residual_loss', False)
        yolo_per_class_obj = kwargs.get('yolo_per_class_obj', False)
        num_obj = 1 if not yolo_per_class_obj else num_classes
        if yolo_per_class_obj:
            assert not background_class
        kernal_size = 3
        stride = 1
        pad = 1
        bn_no_train = deploy
        num_extra_convs = kwargs.get('num_extra_convs', [3])
        if residual_loss:
            assert len(num_extra_convs) != 4
        if len(num_extra_convs) > 1:
            assert extra_conv_channels not in kwargs
        extra_conv_channels = kwargs.get('extra_conv_channels', [1024] *
                num_extra_convs[0])
        add_reorg = kwargs.get('add_reorg', True)
        last_top = last_layer(n)
        if len(num_extra_convs) > 1:
            assert 'extra_conv_kernel' not in kwargs
        else:
            extra_conv_kernel = kwargs.get('extra_conv_kernel', [3] *
                    num_extra_convs[0])

        if len(num_extra_convs) == 4:
            assert len(num_extra_convs) == 4
            assert num_obj == 1
            assert not residual_loss
            suffixes = ['xy', 'wh', 'obj', 'conf']
            outputs = [2 * num_anchor, (coords - 2) * num_anchor, num_anchor,
                    num_anchor * (num_classes)]
            comp_layers = []
            for extra_convs, suffix, num_output in zip(num_extra_convs, suffixes, outputs):
                last = self._add_extra_conv_reorg(n, last_top, extra_convs, add_reorg, 
                    bn_no_train, extra_conv_channels, stride, lr, deploy,
                    suffix='_' + suffix)
                out_name = '{}'.format(suffix)
                n[out_name] = conv(last, num_output, ks=1,
                        deploy=deploy)
                comp_layers.append(n[out_name])
            if not kwargs.get('yolo_full_gpu', False):
                n.xywhobjcls = L.Concat(*comp_layers)
                param_str = json.dumps(calc_out_idx(outputs, num_anchor))
                n['region_shuffle'] = L.Python(n.xywhobjcls, ntop=1, 
                        python_param={'module':'shuffle_layer', 
                            'layer':'ShuffleLayer', 
                            'param_str':param_str})
            assert not kwargs.get('yolo_deconv_to_increase_dim', False)
        elif len(num_extra_convs) == 1:
            if 'multi_feat_anchor' not in kwargs:
                last_top = self._add_extra_conv_reorg(n, last_top,
                        num_extra_convs[0], add_reorg, 
                        bn_no_train, extra_conv_channels, stride, lr, deploy,
                        suffix='',
                        extra_conv_kernel=extra_conv_kernel,
                        extra_conv_groups=kwargs.get('extra_conv_groups'))
            num_output = (num_classes + num_obj + coords) * len(biases) / 2
            if kwargs.get('yolo_deconv_to_increase_dim', False):
                assert not residual_loss
                n['last_conv'] = L.Deconvolution(last_top, convolution_param={'kernel_size':1,
                        'stride':1, 'num_output':num_output, 'pad':0})
                extra_test_param['feat_stride'] = 16
                if kwargs.get('yolo_deconv_to_increase_dim_adapt_bias', True):
                    biases = map(lambda x: x * 2, biases)
            else:
                if 'multi_feat_anchor' in kwargs:
                    multi_feat_anchor = kwargs['multi_feat_anchor']
                    if not deploy:
                        param_str = [f['anchor_bias'] for f in multi_feat_anchor]
                        x = L.Python(n.data, n.label,
                                *[n[f['feature']] for f in multi_feat_anchor],
                                ntop=len(multi_feat_anchor), 
                                name='label_distribute',
                                propagate_down=[False]*(len(multi_feat_anchor)+2),
                                python_param={'module':'label_distribution_layer', 
                                    'layer':'LabelDistributionLayer', 
                                    'param_str': json.dumps(param_str)})
                        if type(x) is not tuple:
                            x = (x,)
                        for i, l in enumerate(x):
                            label_name = 'label{}'.format(i + 1)
                            n[label_name] = l
                            multi_feat_anchor[i]['label_name'] = label_name

                    for feature_anchor in multi_feat_anchor:
                        self.add_feature_anchor(n, 
                                num_classes, 
                                extra_train_param, 
                                feature_anchor,
                                deploy, 
                                num_output,
                                **kwargs)
                    if deploy:
                        all_bb_name, all_prob_name = [], []
                        for feature_anchor in multi_feat_anchor:
                            s = feature_anchor['suffix']
                            f = 'bbox{}'.format(s)
                            t = 'reshape_{}'.format(f)
                            n[t] = L.Reshape(n[f], shape={'dim': [0, -1, 4]})
                            all_bb_name.append(t)
                            f = 'prob{}'.format(s)
                            t = 'reshape_{}'.format(f)
                            n[t] = L.Reshape(n[f], shape={'dim': [0, -1,
                                num_classes + 1]})
                            all_prob_name.append(t)
                        n['prob'] = L.Concat(*[n[name] for name in all_prob_name],
                                axis=1, name='concat_prob')
                        n['bbox'] = L.Concat(*[n[name] for name in all_bb_name],
                                axis=1, name='concat_bbox')
                    return
                n['last_conv'] = conv(last_top, num_output, ks=1,
                        deploy=deploy)
                last_conv_param = {}
                last_conv_param['param'] = [{'lr_mult': 1, 'decay_mult': 1}]
                if last_conv_bias:
                    last_conv_param['param'].append({'lr_mult': 2,
                        'decay_mult': 0})
                    last_conv_param['bias_filler'] = {'type': 'constant', 'value': 0}
                else:
                    last_conv_param['bias_term'] = False
                last_conv_param['weight_filler'] = {'type': 'msra'}
                n['last_conv'] = L.Convolution(last_top, kernel_size=1,
                        num_output=num_output, **last_conv_param)
        else:
            assert False
        
        last = last_layer(n)

        if not deploy and kwargs.get('ignore_negative_first_batch', False):
            extra_train_param['images_ignore_negative'] = \
                    n.__dict__['tops'].items()[0][1].fn.params['tsv_data_param']['batch_size']

        if not deploy:
            if not kwargs.get('yolo_full_gpu', False):
                n['region_loss'] = L.RegionLoss(last, 
                        n['label'],
                        classes=num_classes,
                        softmax=True,
                        bias_match=True,
                        param={'decay_mult': 0, 'lr_mult': 0},
                        thresh=0.6,
                        biases=biases,
                        **extra_train_param)
            else:
                if not residual_loss:
                    last_top = last_layer(n)
                    self.add_yolo_train_loss(n, biases, num_classes, 
                            n.__dict__['tops'].keys()[-1], 
                            '', 
                            extra_train_param,
                            **kwargs)
                else:
                    last_residual_froms = kwargs['residual_loss_froms']
                    last_pre_loss = None
                    for i, residual_from in enumerate(last_residual_froms):
                        last_conv_name = 'last_conv_{}'.format(residual_from)
                        n[last_conv_name] = L.Convolution(n[residual_from], kernel_size=1,
                                num_output=num_output, **last_conv_param)
                        if i > 0:
                            ele_name = 'elewise_{}'.format(last_conv_name)
                            n[ele_name] = L.Eltwise(n[last_conv_name],
                                    n[last_pre_loss])
                            last_pre_loss = ele_name
                            self.add_yolo_train_loss(n, biases, num_classes,
                                ele_name, last_conv_name, **kwargs)
                        else:
                            last_pre_loss = last_conv_name
                            self.add_yolo_train_loss(n, biases, num_classes,
                                last_conv_name, last_conv_name, **kwargs)
                    ele_name = 'elewise_last_conv'
                    n[ele_name] = L.Eltwise(n[last_pre_loss],
                            n['last_conv'])
                    self.add_yolo_train_loss(n, biases, num_classes,
                        ele_name, 'last_conv', **kwargs)
        else:
            if not kwargs.get('yolo_full_gpu', False):
                n.bbox, n.prob = L.RegionOutput(last, n['im_info'],
                        ntop=2,
                        classes=num_classes,
                        thresh=0.005, # 0.24
                        #nms=0.45,
                        biases=biases,
                        **extra_test_param)
            else:
                if not residual_loss:
                    self.add_yolo_test_loss(n, biases, num_classes, 
                            n.__dict__['tops'].keys()[-1], 
                            '',
                            extra_test_param,
                            **kwargs)
                else:
                    last_residual_froms = kwargs['residual_loss_froms']
                    #last_top = last_layer(n)
                    last_pre_loss = None
                    all_suffix = []
                    use_branch_out = True
                    for i, residual_from in enumerate(last_residual_froms):
                        last_conv_name = 'last_conv_{}'.format(residual_from)
                        n[last_conv_name] = L.Convolution(n[residual_from], 
                                kernel_size=1,
                                num_output=num_output, 
                                **last_conv_param)
                        if i > 0:
                            ele_name = 'elewise_{}'.format(last_conv_name)
                            n[ele_name] = L.Eltwise(n[last_conv_name],
                                    n[last_pre_loss])
                            last_pre_loss = ele_name
                            if use_branch_out:
                                self.add_yolo_test_loss(n, biases, 
                                        num_classes, 
                                        ele_name,
                                        last_conv_name,
                                        extra_test_param,
                                        **kwargs)
                        else:
                            last_pre_loss = last_conv_name
                            if use_branch_out:
                                self.add_yolo_test_loss(n, biases, 
                                        num_classes, 
                                        last_conv_name, 
                                        last_conv_name,
                                        extra_test_param,
                                        **kwargs)
                        all_suffix.append(last_conv_name)

                    assert len(last_residual_froms) > 0
                    ele_name = 'elewise_last_conv'
                    n[ele_name] = L.Eltwise(n[last_pre_loss],
                            n['last_conv'])
                    assert len(all_suffix) > 0
                    if use_branch_out:
                        self.add_yolo_test_loss(n, biases, num_classes, 
                                ele_name, 
                                'last_conv', 
                                **kwargs)
                        all_suffix.append('last_conv')
                        all_bb_name, all_prob_name = [], []
                        for s in all_suffix:
                            f = 'bbox{}'.format(s)
                            t = 'reshape_{}'.format(f)
                            n[t] = L.Reshape(n[f], shape={'dim': [0, -1, 4]})
                            all_bb_name.append(t)
                            f = 'prob{}'.format(s)
                            t = 'reshape_{}'.format(f)
                            n[t] = L.Reshape(n[f], shape={'dim': [0, -1,
                                num_classes + 1]})
                            all_prob_name.append(t)
                        n['prob'] = L.Concat(*[n[name] for name in all_prob_name],
                                axis=1)
                        n['bbox'] = L.Concat(*[n[name] for name in all_bb_name],
                                axis=1)
                    else:
                        self.add_yolo_test_loss(n, biases, num_classes, 
                                ele_name, 
                                '', 
                                **kwargs)

    def add_feature_anchor(self, n, num_classes, extra_train_param, feature_anchor,
            deploy, num_output, **kwargs):
        biases = feature_anchor['anchor_bias']
        
        last_top_name = feature_anchor['feature']
        for i, (c, ks) in enumerate(zip(feature_anchor['extra_conv_channels'], 
                feature_anchor['extra_conv_kernels'])):
            s = 'extra_conv_{}_{}'.format(feature_anchor['feature'], i)
            add_dark_block(n, n[last_top_name], s,
                    nout_dark=c,
                    stride=1,
                    lr=1,
                    deploy=deploy,
                    ks=ks)
            last_top_name = s + '/leaky'
        
        last_conv_name = 'last_conv{}'.format(feature_anchor['feature'])
        n[last_conv_name] = conv(n[last_top_name], (4 + 1 + num_classes) *
                len(biases) / 2, ks=1,
                deploy=deploy)
        suffix = '{}_region'.format(feature_anchor['feature'])
    
        if not deploy:
            self.add_yolo_train_loss(n, biases, num_classes, 
                    last_conv_name, 
                    suffix, 
                    extra_train_param,
                    label_name=feature_anchor['label_name'],
                    loss_weight_multiplier=feature_anchor['loss_weight_multiplier'],
                    **kwargs)
        else:
            self.add_yolo_test_loss(n, biases, num_classes, 
                    last_conv_name, 
                    suffix,
                    **kwargs)
    
        feature_anchor['suffix'] = suffix

    def add_yolo_train_loss(self, n, biases, num_classes, last_top_name, suffix, 
            extra_train_param={},
            label_name = None,
            loss_weight_multiplier=1.,
            **kwargs):
        tree_file = kwargs.get('target_synset_tree')
        batch_size = self.batch_size
        if len(batch_size) == 1 or not tree_file:
            self.add_yolo_train_loss_bb_only(n, biases, num_classes,
                    last_top_name, suffix, 
                    extra_train_param=extra_train_param,
                    label_name = label_name,
                    loss_weight_multiplier=loss_weight_multiplier,
                    **kwargs)
            return

        assert len(batch_size) == 2
        n.conv_bb, n.conv_no_bb = L.Slice(n[last_top_name],
                                          ntop=2,
                                          name='slice_batch',
                                          axis=0,
                                          slice_point=[int(batch_size[0])]
                                          )
        weight_bb = float(batch_size[0]) / np.sum(batch_size)
        weight_nobb = float(batch_size[1]) / np.sum(batch_size)
        self.add_yolo_train_loss_bb_only(n, biases, num_classes, 'conv_bb', '',
                extra_train_param=extra_train_param,
                label_name=label_name,
                loss_weight_multiplier=weight_bb * loss_weight_multiplier,
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
        softmax_norm_type = P.Loss.VALID if kwargs.get('yolo_softmax_norm_by_valid', False) else P.Loss.BATCH_SIZE
        n.softmaxtree_loss_nobb, n.obj_index = L.SoftmaxTreeWithLoss(
            n.reshape_conf_nobb, n.label_nobb, n.sigmoid_obj_nobb,
            ntop=2,
            propagate_down=[True, False, False],
            loss_weight=[weight_nobb, 0],
            loss_param={
                'normalization': softmax_norm_type
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
            loss_weight=weight_nobb * extra_index_weight,
            propagate_down=[True, False])

    def add_yolo_train_loss_bb_only(self, n, biases, num_classes, last_top, suffix, 
            extra_train_param={},
            label_name = None,
            loss_weight_multiplier=1.,
            **kwargs):
        num_anchor = len(biases) / 2
        if len(kwargs.get('num_extra_convs', [3])) == 1:
            xy = 'xy{}'.format(suffix)
            wh = 'wh{}'.format(suffix)
            obj = 'obj{}'.format(suffix)
            conf = 'conf{}'.format(suffix)
            n[xy], n[wh], n[obj], n[conf] = L.Slice(n[last_top], 
                    ntop=4,
                    name='slice_region{}'.format(suffix),
                    slice_point=[num_anchor * 2, num_anchor * 4,
                        num_anchor * 5])
        else:
            assert len(kwargs['num_extra_convs']) == 4
        sigmoid_xy = 'sigmoid_xy{}'.format(suffix)
        sigmoid_obj = 'sigmoid_obj{}'.format(suffix)
        n[sigmoid_xy] = L.Sigmoid(n[xy], in_place=True)
        n[sigmoid_obj] = L.Sigmoid(n[obj], in_place=True)
        if kwargs.get('first_batch_objectiveness_enhancement', False):
            if len(self.batch_weights) == 1:
                enhance_name = sigmoid_obj
            else:
                n['first_sigmoid_obj'], n['second_sigmoid_obj'] = L.Slice(n[sigmoid_obj], ntop=2,
                        name='slice_dim0',
                        slice_point=[self.batch_weights[0]],
                        slice_dim=0)
                n['silence_second_sigmoid_obj'] = L.Silence(n['second_sigmoid_obj'], ntop=0,
                        name='silence_second_sigmoid_obj')
                enhance_name = 'first_sigmoid_obj'
            enhance_weight = kwargs.get('first_batch_objectiveness_enhancement_weight', 1)
            n['obj_enhance'] = L.Python(n[enhance_name],
                    ntop=1,
                    name='obj_enhancement',
                    loss_weight=enhance_weight*loss_weight_multiplier,
                    propagate_down=[True],
                    python_param={'module': 'prediction_enhancement_layer',
                        'layer': 'PredictionEnhancementLayer',
                        'param_str': json.dumps({})})
        if label_name is None:
            label_name = 'label'
        regionTargetOutput = L.RegionTarget(n[sigmoid_xy], n[wh], n[sigmoid_obj],
                n[label_name], 
                name='region_target{}'.format(suffix),
                param={'decay_mult': 0, 'lr_mult': 0},
                ntop=6, 
                propagate_down=[False, False, False, False],
                biases=biases,
                **extra_train_param)
        t_xy = 't_xy{}'.format(suffix)
        t_wh = 't_wh{}'.format(suffix)
        t_xywh_weight = 't_xywh_weight{}'.format(suffix)
        t_o_obj = 't_o_obj{}'.format(suffix)
        t_o_noobj = 't_o_noobj{}'.format(suffix)
        t_label = 't_label{}'.format(suffix)
        n[t_xy], n[t_wh], n[t_xywh_weight], n[t_o_obj], n[t_o_noobj], n[t_label] = regionTargetOutput
        xy_loss = 'xy_loss{}'.format(suffix)
        xywh_norm_type = P.Loss.VALID
        if kwargs.get('yolo_xywh_norm_by_weight_sum', False):
            xywh_norm_type = P.Loss.SUM_WEIGHT
        n[xy_loss] = L.EuclideanLoss(n[xy], n[t_xy], n[t_xywh_weight], 
                propagate_down=[True, False, False], 
                loss_weight=1*loss_weight_multiplier,
                loss_param={'normalization': xywh_norm_type})
        wh_loss = 'wh_loss{}'.format(suffix)
        n[wh_loss] = L.EuclideanLoss(n[wh], n[t_wh], n[t_xywh_weight], 
                propagate_down=[True, False, False], 
                loss_weight=1*loss_weight_multiplier,
                loss_param={'normalization': xywh_norm_type})
        o_obj_loss = 'o_obj_loss{}'.format(suffix)
        n[o_obj_loss] = L.EuclideanLoss(n[obj], n[t_o_obj], 
                propagate_down=[True, False], loss_weight=5*loss_weight_multiplier)
        o_noobj_loss = 'o_noobj_loss{}'.format(suffix)
        n[o_noobj_loss] = L.EuclideanLoss(n[obj], n[t_o_noobj], 
                propagate_down=[True, False], loss_weight=1*loss_weight_multiplier)
        tree_file = kwargs.get('target_synset_tree')
        if not tree_file:
            reshape_t_label = 'reshape_t_label{}'.format(suffix)
            n[reshape_t_label] = L.Reshape(n[t_label], shape={'dim': [-1, 1, 0, 0]})
            reshape_conf = 'reshape_conf{}'.format(suffix)
            n[reshape_conf] = L.Reshape(n[conf], shape={'dim': [-1, num_classes, 0, 0]})
            softmax_loss = 'softmax_loss{}'.format(suffix)
            norm_type = P.Loss.BATCH_SIZE
            if kwargs.get('yolo_softmax_norm_by_valid', False):
                norm_type = P.Loss.VALID
                loss_weight = 1
            else:
                loss_weight = num_anchor
            if kwargs.get('softmax_loss_extra_weight'):
                loss_weight = loss_weight * kwargs['softmax_loss_extra_weight']
            assert 'yolo_softmax_extra_weight' not in kwargs
            n[softmax_loss] = L.SoftmaxWithLoss(n[reshape_conf], n[reshape_t_label],
                    propagate_down=[True, False],
                    loss_weight=loss_weight * loss_weight_multiplier, 
                    loss_param={'ignore_label': -1, 'normalization':
                        norm_type})
            return
        loss_weight = kwargs.get('softmax_loss_extra_weight', 1)
        n.reshape_conf = L.Reshape(n.conf, axis=1, num_axes=1, shape={'dim': [num_classes, num_anchor]})
        softmax_norm_type = P.Loss.VALID if kwargs.get('yolo_softmax_norm_by_valid', False) else P.Loss.BATCH_SIZE
        n.softmaxtree_loss = L.SoftmaxTreeWithLoss(n.reshape_conf, n.t_label,
                propagate_down=[True, False],
                loss_weight=loss_weight * loss_weight_multiplier,
                loss_param={
                    'ignore_label': -1,
                    'normalization': softmax_norm_type 
                },
                softmaxtree_param={
                    'tree': tree_file
                })

    def add_yolo_test_loss(self, n, biases, num_classes, last_top, suffix, 
            region_prediction_param={}, **kwargs):
        tree_file = kwargs.get('target_synset_tree')
        num_anchor = len(biases) / 2
        class_specific_nms = kwargs.get('class_specific_nms', True)
        if len(kwargs.get('num_extra_convs', [3])) == 1:
            xy = 'xy{}'.format(suffix)
            wh = 'wh{}'.format(suffix)
            obj = 'obj{}'.format(suffix)
            conf = 'conf{}'.format(suffix)
            n[xy], n[wh], n[obj], n[conf] = L.Slice(n[last_top], 
                    ntop=4,
                    name='slice_region{}'.format(suffix),
                    slice_point=[num_anchor * 2, num_anchor * 4, num_anchor * 5])
    
        sigmoid_xy = 'sigmoid_xy{}'.format(suffix)
        n[sigmoid_xy] = L.Sigmoid(n[xy], in_place=True)
        sigmoid_obj  = 'sigmoid_obj{}'.format(suffix)
        n[sigmoid_obj] = L.Sigmoid(n[obj], in_place=True)
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
                  'threshold': kwargs.get('softmax_tree_prediction_threshold',
                      0.5),
                  'output_tree_path': kwargs.get('output_tree_path', False)
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
                # class-independent NMS on the classes
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

        reshape_conf = 'reshape_conf{}'.format(suffix)
        n[reshape_conf] = L.Reshape(n[conf], shape={'dim': [-1, num_classes, 0, 0]})
        softmax_conf = 'softmax_conf{}'.format(suffix)
        n[softmax_conf] = L.Softmax(n[reshape_conf])
        bbox = 'bbox{}'.format(suffix)
        prob = 'prob{}'.format(suffix)
        logging.info(bbox)
        n[bbox], n[prob] = L.RegionPrediction(n[sigmoid_xy], n[wh], n[sigmoid_obj],
                n[softmax_conf], n.im_info, n.data, ntop=2, 
                thresh=0.005, # 0.24
                name='region_predict{}'.format(suffix),
                biases=biases,
                **region_prediction_param)

def get_region_param(**kwargs):
    extra_train_param, extra_test_param = {}, {}
    train_test_keys = ['multibin_xy', 
            'multibin_xy_low',
            'multibin_xy_high', 
            'multibin_xy_count',
            'multibin_wh',
            'multibin_wh_low', 
            'per_class_obj', 
            'multibin_wh_high',
            'multibin_wh_count', 
            'background_class',
            'sigmoid_xy']

    train_keys = ['obj_set1_center_around', 
            'obj_kl_distance', 'xy_kl_distance', 
            'object_scale', 
            'noobject_scale',
            'class_scale',
            'obj_only', 
            'obj_nonobj_align_to_iou', 
            'obj_ignore_center_around', 
            'rescore', 
            'use_background_class_to_reduce_obj',
            'obj_cap_center_around', 
            'coord_scale',
            'exp_linear_wh',
            'iou_th_to_use_bkg_cls',
            'anchor_aligned_images',
            'nonobj_extra_power',
            'obj_extra_power',
            'obj_nonobj_nopenaltyifsmallthaniou', 
            'delta_region3',
            'disable_no_penalize_if_iou_large',
            'force_negative_with_partial_overlap',
            'not_ignore_negative_seen_images',
            ]

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

    if 'target_synset_tree' in kwargs and not kwargs.get('yolo_full_gpu', False):
        extra_train_param['tree'] = kwargs['target_synset_tree']
        extra_test_param['tree'] = kwargs['target_synset_tree']

    if 'region_layer_map' in kwargs:
        extra_test_param['map'] = kwargs['region_layer_map']
    return extra_train_param, extra_test_param

def test_yolo():
    y = Yolo()
    n = caffe.NetSpec()
    y.add_input_data(n, 20)
    y.add_body(n, 0.1, 20, 'xyz', )

def test_calc_out_idx():
    x = np.asarray([2, 2, 1, 1])
    x = x * 2
    y = calc_out_idx(x, 2)

if __name__ == '__main__':
    #test_yolo()
    test_calc_out_idx()




