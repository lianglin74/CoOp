from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
from layerfactory import conv_bn, last_layer, conv
from darknet import DarkNet

class Yolo(object):
    def add_input_data(self, n, num_classes, **kwargs):
        include = {'phase': getattr(caffe_pb2, 'TRAIN')}
        transform_param = {'mirror': True, 
                    'crop_size': 224,
                    'mean_value': [104, 117, 123]}

        source_file = kwargs.get('source', 'train.tsv')

        tsv_data_param = {'source': source_file, 
                'batch_size': 16, 
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

    def add_body(self, n, lr, num_classes, cnnmodel, 
            deploy=False, cpp_version=False, **kwargs):
        last = last_layer(n)
        kernal_size = 3
        stride = 1
        pad = 1
        bn_no_train = deploy
        num_output = 1024

        for name in ['conv19', 'conv20']:
            self.dark_block(n, name, num_output, stride, lr, deploy,
                    bn_no_train)

        # the default is for darknet19.
        reorg_from = kwargs.get('reorg_from', 'dark5e/leaky')

        n['dark5e/leaky_reorg'] = L.Reorg(n[reorg_from], stride=2)
        n.concat20 = L.Concat(n['dark5e/leaky_reorg'], n['conv20/leaky'])
        self.dark_block(n, 'conv21', num_output, stride, lr, deploy,
            bn_no_train)

        num_output = (num_classes + 1 + 4) * 5
        n['conv_reg'] = conv(n['conv21/leaky'], num_output, ks=1,
                deploy=deploy)
        biases = [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
        if not deploy:
            n['loss'] = L.RegionLoss(n['conv_reg'], 
                    n['label'],
                    classes=num_classes,
                    coords=4,
                    softmax=True,
                    rescore=True,
                    bias_match=True,
                    object_scale=5,
                    noobject_scale=1.0,
                    class_scale=1.0,
                    coord_scale=1.0,
                    thresh=0.6,
                    debug_info=False,
                    biases=biases)
        else:
            n.bbox, n.prob = L.RegionOutput(n['conv_reg'], n['im_info'],
                    ntop=2,
                    classes=num_classes,
                    thresh=0.005, # 0.24
                    nms=0.45,
                    biases=biases)


