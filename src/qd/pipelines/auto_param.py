import re
import logging
from qd.qd_common import set_if_not_exist
from qd.qd_common import load_from_yaml_file
from qd.qd_common import dict_has_path, dict_get_path_value
from qd.qd_common import dict_set_path_if_not_exist
from qd.qd_common import get_mpi_size
import random
from qd.qd_common import dict_ensure_path_key_converted
from qd.pipeline import load_pipeline
from qd.pipeline import get_all_test_data
from qd.qd_common import dict_update_nested_dict
from qd.qd_common import iter_swap_param_simple
from qd.qd_common import dict_update_path_value
from qd.process_tsv import populate_dataset_details
from qd.tsv_io import TSVDataset, TSVFile, tsv_writer, tsv_reader
from tqdm import tqdm
import json
import copy
from qd.qd_common import max_iter_mult
from pprint import pformat


def update_by_condition_default_until_no_change(all_condition_default,
        param, place_holder=None, force=False):
    while True:
        change_list = update_by_condition_default(all_condition_default,
                param, place_holder, force=force)
        if len(change_list) == 0:
            break
        else:
            logging.info(pformat(change_list))

def update_by_condition_default(all_condition_default, param,
        place_holder=None, force=False):
    change_list = []
    if isinstance(all_condition_default, dict):
        assert len(all_condition_default) == 2 and \
                'condition' in all_condition_default and \
                'default' in all_condition_default
        condition, default = all_condition_default['condition'], all_condition_default['default']
        def condition_met():
            for k, v in condition.items():
                if not dict_has_path(param, k):
                    return False
                pv = dict_get_path_value(param, k)
                if isinstance(v, str):
                    if v.startswith('$'):
                        continue
                    if v == pv:
                        continue
                    if isinstance(pv, str) and re.match(v, pv) is not None:
                        continue
                    return False
                else:
                    if pv != v:
                        return False
            return True
        if condition_met():
            curr_place_holder = {v: dict_get_path_value(param, k)
                            for k, v in condition.items()
                                if isinstance(v, str) and v.startswith('$')}
            if place_holder is None:
                place_holder = curr_place_holder
            else:
                place_holder = copy.deepcopy(place_holder)
                place_holder.update(curr_place_holder)
            if isinstance(default, dict):
                for k, v in default.items():
                    if type(v) is str and v in place_holder:
                        v = place_holder[v]
                    if isinstance(v, str) and '$' in v:
                        for p in place_holder.keys():
                            if p in v:
                                v = v.replace(p, place_holder[p])
                    if not force:
                        if dict_set_path_if_not_exist(param, k, v):
                            change_list.append((k, v))
                    else:
                        if not dict_has_path(param, k) or \
                                dict_get_path_value(param, k) != v:
                            change_list.append((k, v))
                        dict_update_path_value(param, k, v)
            else:
                change_list.extend(update_by_condition_default(default, param,
                        place_holder, force))
    elif isinstance(all_condition_default, list):
        for condition_default in all_condition_default:
            change_list.extend(update_by_condition_default(condition_default,
                    param, place_holder=place_holder, force=force))
    else:
        raise NotImplementedError

    return change_list

def get_gt_closeness(data, split, net_input, version=None):
    dataset = TSVDataset(data)
    populate_dataset_details(data, check_image_details=True)
    label_iter = dataset.iter_data(split, 'label', version=version)
    hw_iter = dataset.iter_data(split, 'hw')

    all_min_dist = []
    for i, ((label_key, str_rects), (hw_key, str_hw)) in tqdm(
            enumerate(zip(label_iter, hw_iter))):
        h, w = map(int, str_hw.split(' '))
        rects = json.loads(str_rects)
        all_xy = []
        for rect in rects:
            r = rect['rect']
            x_cent = (r[0] + r[2]) / 2. / w
            y_cent = (r[1] + r[3]) / 2. / h
            all_xy.append((x_cent * net_input, y_cent * net_input))
        for i in range(len(all_xy)):
            curr_min_dist = None
            for j in range(i + 1, len(all_xy)):
                pi = all_xy[i]
                pj = all_xy[j]
                curr_dist = (abs(pi[0] - pj[0]) + abs(pi[1] - pj[1])) / 2.
                if curr_min_dist is None:
                    curr_min_dist = curr_dist
                elif curr_dist < curr_min_dist:
                    curr_min_dist = curr_dist
            if curr_min_dist is not None:
                all_min_dist.append(curr_min_dist)
    return sum(all_min_dist) / len(all_min_dist)

class AutoParam(object):
    run_type_aml = 'aml'
    run_type_debug = 'debug'

    def __init__(self):
        #self.run_type = run_type
        logging.info('pass run_type by calling functions')
        self.all_condition_default = []
        self.all_func_default = [
            (
                lambda p: p.get('pipeline_type') == 'maskrcnn_benchmark',
                {
                    'MaskTSVDataset$remove_images_without_annotations': False,
                }
            ),
            (
                lambda p: 'images_per_gpu' in p and 'num_gpu' in p,
                {'effective_batch_size': (lambda p: p['images_per_gpu'] * p['num_gpu'])}
            )
        ]

    def update_mmdet(self, param, env):
        all_default = [
                ('expid_prefix', 'MM'),
                ('pipeline_type', 'MMDetPipeline'),
                ('effective_batch_size', 16),
                ]
        self.update_by_default(all_default, param)

        if 'num_gpu' not in env:
            env['num_gpu'] = self.get_default_num_gpu_for_maskrcnn(param, env)

        all_condition_default = [
                ({'data': 'coco2017Full'}, {'evaluate_method': 'coco_box'}),
                ({'data': 'voc20'}, {'ignore_filter_img': True}),
                ({'data': 'coco2017Full'}, {'ignore_filter_img': True}),
                ]
        self.update_by_condition_default(all_condition_default,
                param)

        all_func_default = [
                (
                    lambda p: p['data'] == 'OpenImageV5C' or 'OI5C' in param['data'],
                    {
                        'evaluate_method': 'neg_aware_gmap',
                        'apply_nms_gt': False}
                    )
        ]
        self.update_by_func_defaults(all_func_default, param)

        self.update_mmdet_opt(param, env)

        if env['run_type'] == 'debug':
            env['num_gpu'] = 1
            param['effective_batch_size'] = 2
            param['base_lr'] = 0.02

        all_condition_assert = [
                (
                    {'classification_loss_type': 'IBCE0.1_0.9_1'},
                    {'multi_hot_label': True},
                )
        ]
        self.check_by_condition_assert(all_condition_assert, param)

    def check_by_condition_assert(self, all_condition_assert, param):
        for con, a in all_condition_assert:
            if all(param.get(k) == v for k, v in con.items()):
                assert all(param[k] == v for k, v in a.items())

    def update_mmdet_opt(self, param, env):
        default_bs = 16
        default_lr = 0.02
        set_if_not_exist(param, 'effective_batch_size', default_bs)
        bs_reduce_factor = 1. * param['effective_batch_size'] / default_bs
        set_if_not_exist(param, 'base_lr', default_lr * bs_reduce_factor)

    def update_yolo(self, param, env):
        all_condition_default = load_from_yaml_file('./aux_data/auto_param/yolo.yaml')
        self.update_by_condition_default(all_condition_default,
                param)

        if dict_has_path(param, 'yolo_train_session_param$anchors'):
            dict_set_path_if_not_exist(param,
                    'yolo_predict_session_param$anchors',
                    dict_get_path_value(param, 'yolo_train_session_param$anchors'))
        if 'xywhobj_ratio' in param:
            ratio = dict_get_path_value(param, 'xywhobj_ratio')
            dict_set_path_if_not_exist(param,
                    'rt_param$xy_scale',
                    ratio)
            dict_set_path_if_not_exist(param,
                    'rt_param$wh_scale',
                    ratio)
            dict_set_path_if_not_exist(param,
                    'rt_param$object_scale',
                    5 * ratio)
            del param['xywhobj_ratio']

    def expand_params(self, param, env):
        pass

    def update_pipeline_param(self, param, env):
        if 'scale' in param:
            scale = param['scale']
            param['effective_batch_size'] = int(param['effective_batch_size']
                    * scale)
            from qd.pipelines.auto_param import max_iter_mult
            if isinstance(param['max_iter'], int):
                # otherwise, it is epoch, and no need to update
                param['max_iter'] = max_iter_mult(param['max_iter'], 1. / scale)
            param['base_lr'] *= scale
            param['env']['num_gpu'] = int(param['env']['num_gpu'] * scale)
            del param['scale']

        if 'init_full_expid' in param:
            init_full_expid = param['init_full_expid']
            if init_full_expid:
                pip = load_pipeline(full_expid=init_full_expid)
                base_model  = pip._get_checkpoint_file()
                assert base_model
                assert 'basemodel' not in param
                param['basemodel'] = base_model
            del param['init_full_expid']

        # in this funciton, remove most of the codes except this one
        condition_default = load_from_yaml_file('./aux_data/auto_param/pipeline.yaml')
        update_by_condition_default_until_no_change(condition_default, param)

        condition_default = load_from_yaml_file('./aux_data/auto_param/force_parameter.yaml')
        update_by_condition_default_until_no_change(condition_default, param,
                force=True)
        # in the case of prediction only, we don't need to have template param
        template = param.get('param_template')
        if template == 'classification_for_maskrcnn':
            self.update_pipeline_param_for_classification_for_maskrcnn(param,
                    env)
        elif template == 'imagenet_classification':
            self.update_pipeline_param_for_imagenet_classification(param, env)
        elif template == 'classification_by_maskrcnn':
            self.update_classification_by_mask(param, env)
        elif template == 'imagenet_classification_by_mask':
            self.update_classification_by_mask(param, env)
        elif template in ['maskrcnn_benchmark', 'maskrcnn_cls_only']:
            self.update_pipeline_param_for_maskrcnn(param, env)
        elif template == 'maskrcnn_continous':
            self.update_pipeline_param_for_maskrcnn(param, env)
            self.update_pipeline_param_for_maskrcnn_conti(param, env)
        elif template == 'mmdet':
            self.update_mmdet(param, env)
        elif template == 'YoloV2PtPipeline':
            self.update_yolo(param, env)
        dict_ensure_path_key_converted(param)

        self.update_by_condition_default(self.all_condition_default, param)
        self.update_by_func_defaults(self.all_func_default, param)

        if param.get('pipeline_type') == 'MaskRCNNPipeline':
            self.update_pipeline_param_for_maskrcnn(param, env)

        if 'test_batch_size' not in param:
            # 8 work for x152, use 4 for all
            param['test_batch_size'] = 4 * env['num_gpu']
        else:
            assert param['test_batch_size'] >= env['num_gpu']
            assert (param['test_batch_size'] % env['num_gpu']) == 0

        if 'param_template' in param:
            del param['param_template']

        if get_mpi_size() > 1 and 'pipeline_type' in param:
            param['dist_url_tcp_port'] = 23456
        else:
            old_state = random.getstate()
            from time import time
            random.seed(time())
            param['dist_url_tcp_port'] = int(random.random() * 10000 + 20000)
            random.setstate(old_state)

        if env['run_type'] == 'debug' and 'pipeline_type' in param:
            if param['pipeline_type'] == 'YoloV2PtPipeline':
                #dict_update_nested_dict(param,
                        #{'yolo_train_session_param': {'debug_train': True}})
                dict_update_nested_dict(param,
                        {'yolo_train_session_param': {'display': 1}})
                param['effective_batch_size'] = 8
            elif param['pipeline_type'] == 'classification':
                param['effective_batch_size'] = 16
            #param['display'] = 1
            #param['force_train'] = True

        if 'expid' not in param and \
                'pipeline_type' in param and \
                'full_expid' not in param:
            from qd.pipeline import generate_expid
            expid = generate_expid(param)
            param['expid'] = expid
        if 'full_expid' not in param:
            ks = ['data', 'net', 'expid']
            if all(k in param for k in ks):
                param['full_expid'] = '_'.join(
                    map(str, [param[k] for k in ks]))

    def update_pipeline_param_for_maskrcnn_conti(self, param, env):
        defaults = [('init_model_only', False)]
        for k, v in defaults:
            set_if_not_exist(param, k, v)
        # sometimes, we want to first train a model with infinite iterations,
        # then we lauch another training based on previous latest model. In
        # this case, we normally use this field to indicate how many iterations
        # we can have
        if 'continous_extra_iter' in param:
            all_condition_default = [
                    (
                        # does not work for this.
                        {
                            'data': 'TaxOI5CV1_1_5k_with_bb',
                            'net': 'e2e_faster_rcnn_R_34_FPN_1x',
                            'MaskTSVDataset$multi_hot_label': True,
                            'with_dcn': True,
                        },
                        {
                            'basemodel': './output/TaxOI5CV1_1_5k_with_bb_e2e_faster_rcnn_R_34_FPN_1x_M_BS16_MaxIter90000000_LR0.02_IBCE0.1_0.9_1_DCN_MultiHot_Gamma1.0/snapshot/model_0647500.pth',
                        },
                    ),
                    (
                        {
                            'data': 'TaxOI5CV1_1_5k_with_bb',
                            'net': 'e2e_faster_rcnn_R_50_FPN_1x',
                            'MaskTSVDataset$multi_hot_label': True,
                            'with_dcn': True,
                        },
                        {
                            'basemodel': './output/TaxOI5CV1_1_5k_with_bb_e2e_faster_rcnn_R_50_FPN_1x_M_BS16_MaxIter90000000_LR0.02_IBCE0.1_0.9_1_DCN_MultiHot_Gamma1.0/snapshot/model_2877500.pth',
                        },
                    ),
            ]
        else:
            all_condition_default = [
                    (
                        {
                            'data': 'TaxOI5CV1_1_5k_with_bb',
                            'net': 'e2e_faster_rcnn_R_50_FPN_1x',
                            'MaskTSVDataset$multi_hot_label': True,
                        },
                        {
                            'basemodel': './output/TaxOI5CV1_1_5k_with_bb_e2e_faster_rcnn_R_50_FPN_1x_M_RemoveEmptyFalse_BS16_MaxIter942175_LR0.02_IBCE0.1_0.9_1_MultiHot/snapshot/model_0627500.pth'
                        },
                    ),
            ]
        self.update_by_condition_default(all_condition_default, param)
        if 'continous_extra_iter' in param:
            from qd.qd_common import parse_iteration
            last_iter = parse_iteration(param['basemodel'])
            extras = param['continous_extra_iter']
            param['max_iter'] = int(last_iter + sum(extras))
            steps = []
            pre = last_iter
            for e in extras[:-1]:
                pre = pre + e
                steps.append(int(pre))
            param['stageiter'] = tuple(steps)
            del param['continous_extra_iter']

    def update_mask_opt(self, param, env):
        d = param.get('data', '')
        df = {'default_bs': 16,
              'default_max_iter': '12e',
              'default_lr': 0.02,
              'default_gpu': 4}
        if any(d.startswith(p) for p in ['TaxOI5C',
                'OpenImageV5C']):
            df['default_max_iter'] = 942175
        elif d == 'voc20':
            df['default_max_iter'] = 9000
        elif d == 'coco2017Full':
            df['default_max_iter'] = 90000

        if param.get('net') == 'retinanet_X_101_32x8d_FPN_1x':
            df.update({
                'default_bs': 4,
                'default_lr': 0.0025,
                })
            df['default_max_iter'] = max_iter_mult(df['default_max_iter'], 4)
        self.update_opt_by_default(df, param, env)

    def update_opt_by_default(self, df, param, env):
        max_iter_x = param.pop('max_iter_x', 1)
        lr_factor = param.pop('lr_factor', 1)

        s = param.pop('gpu_x_adapt_lr_iter_bs', 1)
        param['base_lr'] *= s * lr_factor
        param['max_iter'] = max_iter_mult(param['max_iter'], max_iter_x * 1./s)
        param['effective_batch_size'] *= s

        assert (param['effective_batch_size'] % param['images_per_gpu']) == 0

        set_if_not_exist(env, 'num_gpu', param['effective_batch_size'] //
                param['images_per_gpu'])

    def update_by_func_defaults(self, all_func_default, param):
        for func, default in all_func_default:
            if func(param):
                for k, v in default.items():
                    import inspect
                    if inspect.isfunction(v):
                        dict_set_path_if_not_exist(param, k, v(param))
                    else:
                        dict_set_path_if_not_exist(param, k, v)

    def update_by_condition_default(self, all_condition_default, param):
        for condition, default in all_condition_default:
            if all(dict_has_path(param, k) and dict_get_path_value(param, k) == v for k, v in condition.items()):
                for k, v in default.items():
                    dict_set_path_if_not_exist(param, k, v)

    def update_by_default(self, all_default, param):
        for k, v in all_default:
            set_if_not_exist(param, k, v)

    def get_default_num_gpu_for_maskrcnn(self, param, env):
        if env['run_type'] == 'debug':
            return 1

        #if param['net'] in [
                #'e2e_faster_rcnn_X_101_32x8d_FPN_1x_tb',
                #]:
            #return param['effective_batch_size']

        return param['effective_batch_size'] // 4

    def update_pipeline_param_for_maskrcnn(self, param, env):
        all_condition_default = load_from_yaml_file('./aux_data/auto_param/maskrcnn_benchmark_basemodel.yaml')
        self.update_by_condition_default(all_condition_default,
                param)

        assert 'expid_prefix' in param, 'deprecate the following'
        #set_if_not_exist(param, 'expid_prefix', 'M')

        assert param['pipeline_type'] == 'MaskRCNNPipeline'
        #param['pipeline_type'] = 'MaskRCNNPipeline'

        coexist_defaults = [('MaskTSVDataset$multi_hot_label', True),
                            ('MODEL$ROI_BOX_HEAD$CLASSIFICATION_LOSS', 'IBCE0.1_0.9_1'),
                            ('MODEL$ROI_BOX_HEAD$CLASSIFICATION_ACTIVATE', 'sigmoid')]

        if any(dict_has_path(param, k) and
                dict_get_path_value(param, k) == v for k, v in coexist_defaults):
            for k, v in coexist_defaults:
                dict_set_path_if_not_exist(param, k, v)

        if any(param.get('data', '').startswith(p) for p in ['TaxOI5C',
                'OpenImageV5C']):
            dict_set_path_if_not_exist(param,
                    'MaskTSVDataset$remove_images_without_annotations',
                    False)

        # images_per_gpu
        all_condition_default = [
                (
                    {
                        'net': 'e2e_faster_rcnn_X_101_32x8d_FPN_1x_tb',
                        'INPUT$USE_FIXED_SIZE_AUGMENTATION': True,
                        'INPUT$FIXED_SIZE_AUG$INPUT_SIZE': 416,
                    },
                    {
                        'images_per_gpu': 16,
                    },
                ),
                (
                    {
                        'net': 'e2e_faster_rcnn_R_101_FPN_1x',
                    },
                    {
                        'images_per_gpu': 2,
                    },
                )
        ]
        self.update_by_condition_default(all_condition_default, param)
        set_if_not_exist(param, 'images_per_gpu', 4)

        all_condition_default = [
                (
                    {
                        'net': 'e2e_faster_rcnn_X_101_32x8d_FPN_1x_tb',
                        'INPUT$USE_FIXED_SIZE_AUGMENTATION': True,
                        'INPUT$FIXED_SIZE_AUG$INPUT_SIZE': 416,
                    },
                    {
                        'lr_factor': 0.5,
                    },
                )
        ]
        self.update_by_condition_default(all_condition_default, param)
        self.update_mask_opt(param, env)

        all_condition_default = [
                (
                    {
                        'net': 'e2e_faster_rcnn_X_152_32x8d_FPN_1x_tb',
                    },
                    {
                        'images_per_gpu': 2,
                    },
                ),
                (
                    {
                        'net': 'e2e_faster_rcnn_X_101_32x8d_FPN_1x_tb',
                        'MODEL$RESNETS$USE_SE': True,
                        'effective_batch_size': 128
                    },
                    {
                        # 0.16 is too large for openimage
                        'base_lr': 0.08
                    },
                ),
                ]
        self.update_by_condition_default(all_condition_default,
                param)

        # init parameters
        all_condition_default = load_from_yaml_file('./aux_data/auto_param/maskrcnn_benchmark_basemodel.yaml')
        self.update_by_condition_default(all_condition_default,
                param)

        dict_set_path_if_not_exist(env, 'num_gpu',
                self.get_default_num_gpu_for_maskrcnn(param, env))

        if param['effective_batch_size'] // env['num_gpu'] == 1:
            dict_set_path_if_not_exist(param,
                    'DATALOADER$ASPECT_RATIO_GROUPING', False)

        assert param['effective_batch_size']

        if env['run_type'] != 'debug':
            set_if_not_exist(param, 'log_step', 100)

        if env['run_type'] == 'debug':
            default_debug_param = [
                    ('force_train', True),
                    ('log_step', 1),
                    ('num_workers', 0),
                    ('DATALOADER$ASPECT_RATIO_GROUPING', False),
                    ('MaskTSVDataset$remove_images_without_annotations', False),
                    #('SOLVER$WARMUP_ITERS', 10),
                    #swap_params.append(('max_iter', [50000]))
                    #swap_params.append(('base_lr', [0.002]))
                    #swap_params.append(('SOLVER$CHECKPOINT_PERIOD', [1000]))
                    #swap_params.append(('device', ['cpu']))
                    ]
            for k, v in default_debug_param:
                set_if_not_exist(param, k, v)

    def infer_num_gpu(self, param):
        if dict_has_path(param, 'env$num_gpu'):
            return dict_get_path_value(param, 'env$num_gpu')
        if param['env']['run_type'] == 'debug':
            num_gpu = 1
        else:
            if 'num_gpu' in param:
                num_gpu = param['num_gpu']
            else:
                num_gpu = 4
        if 'num_gpu' in param:
            del param['num_gpu']
        return num_gpu

    def update_classification_by_mask(self, param,
            env):
        num_gpu = self.infer_num_gpu(param)
        if 'effective_batch_size' in param:
            eb = param['effective_batch_size']
        else:
            eb = 64 * num_gpu

        lr_scalar = eb / 256.
        default_param = {
                'data': 'imagenet2012',
                'net': 'resnet50',
                'effective_batch_size': eb,
                'max_iter': '120e',
                'base_lr': 0.1 * lr_scalar,
                'dataset_type': 'single',
                'expid_prefix': 'TByM', # Tagging by MaskRCNN
                'pipeline_type': 'classification_by_mask',
                'evaluate_method': 'top1',
                'bgr2rgb': True,
                }

        env['num_gpu'] = num_gpu

        for k, v in default_param.items():
            set_if_not_exist(param, k, v)

        if 'step_lr' not in param and 'scheduler_type' not in param:
            if type(param['max_iter']) is str:
                assert param['max_iter'].endswith('e')
                param['step_lr'] = '{}e'.format(int(param['max_iter'][:-1]) // 4)

    def update_pipeline_param_for_imagenet_classification(self, param, env):
        if env['run_type'] == 'debug':
            num_gpu = 1
        else:
            if 'num_gpu' in param:
                num_gpu = param['num_gpu']
            else:
                num_gpu = 4
        if 'num_gpu' in param:
            del param['num_gpu']
        eb = 64 * num_gpu
        default_param = {
                'data': 'imagenet2012',
                'net': 'resnet50',
                'effective_batch_size': eb,
                'max_iter': '120e',
                'base_lr': num_gpu * 0.1 / 4,
                'dataset_type': 'single',
                'expid_prefix': 'TagM', # Classification for MaskRCNN
                'pipeline_type': 'classification',
                'evaluate_method': 'top1',
                }

        env['num_gpu'] = num_gpu

        for k, v in default_param.items():
            set_if_not_exist(param, k, v)

        assert type(param['max_iter']) is str
        assert param['max_iter'].endswith('e')
        param['max_epoch'] = int(param['max_iter'][:-1])
        set_if_not_exist(param, 'step_lr', param['max_epoch'] // 4)

    def update_pipeline_param_for_classification_for_maskrcnn(self, param, env):
        bs_per_gpu = 64
        ref_base_lr = 0.1 # 256 batch size with 0.1 learning rate
        if 'net' in param:
            net = param['net']
            if net in ['e2e_faster_rcnn_X_101_32x8d_FPN_1x_tb',
                    'e2e_faster_rcnn_R_152_FPN_1x_tb',
                    'e2e_faster_rcnn_X_152_32x8d_FPN_1x_tb']:
                bs_per_gpu = 32
                ref_base_lr = 0.1 / 2.

        if env['run_type'] == 'debug':
            num_gpu = 1
            bs_per_gpu = 2
        else:
            if 'num_gpu' in param:
                num_gpu = param['num_gpu']
            else:
                num_gpu = 4
        if 'num_gpu' in param:
            del param['num_gpu']

        eb = bs_per_gpu * num_gpu

        env['num_gpu'] = num_gpu
        default_param = {
                'data': 'imagenet2012',
                'effective_batch_size': eb,
                'max_iter': '120e',
                'step_lr': '30e',
                'base_lr': num_gpu * ref_base_lr / 4,
                'dataset_type': 'single',
                'expid_prefix': 'CM', # Classification for MaskRCNN
                'pipeline_type': 'classification_for_mask',
                'evaluate_method': 'top1',
                'MODEL$BACKBONE$FREEZE_CONV_BODY_AT': 0,
                'bgr2rgb': True,
                }

        for k, v in default_param.items():
            set_if_not_exist(param, k, v)

    def get_yolo_effective_batch_size(self):
        ebs = 32 * 4
        return ebs

    def get_maskrcnn_effective_bath_size(self):
        if self.run_type == AutoParam.run_type_debug:
            ebs = 2
        else:
            ebs = 8
        return ebs

    def get_maskrcnn_params_basic(self, param):
        ebs = self.get_maskrcnn_effective_bath_size()
        swap_params = [
                ('net', [
                    'e2e_faster_rcnn_R_34_FPN_1x_tb',
                    'e2e_faster_rcnn_R_50_FPN_1x',
                    'e2e_faster_rcnn_R_101_FPN_1x',
                    'e2e_faster_rcnn_X_101_32x8d_FPN_1x',
                    'retinanet_R-50-FPN_1x',
                    'retinanet_R-50-FPN_P5_1x',
                    'retinanet_R-101-FPN_1x',
                    'retinanet_X_101_32x8d_FPN_1x',
                    ]),
                ('max_iter', ['20e', '40e']),
                ('effective_batch_size', [ebs]),
                ('expid_prefix', ['M']),
                ('log_step', [100]),
                ('base_lr', [0.01]),
                ('evaluate_method', ['map']),
                ('pipeline_type', ['MaskRCNNPipeline']),
                ]
        self.update_swap_params_by_param(swap_params, param)
        all_param = []
        for swap_param in iter_swap_param(swap_params):
            p = {}
            dict_update_nested_dict(p, swap_param)
            p = get_maskrcnn_param(p)
            all_param.append(p)
        return all_param

    def update_swap_params_by_param(self, swap_params, param):
        visited = set()
        for key, values in swap_params:
            if key in param:
                values.clear()
                values.append(param[key])
                visited.add(key)
        for key in param:
            if key not in visited:
                swap_params.append((key, [param[key]]))

    def get_maskrcnn_params_coco_init(self, param):
        ebs = self.get_maskrcnn_effective_bath_size()
        data = param['data']
        swap_params = [
                ('data', [data]),
                ('net', [
                    #'e2e_faster_rcnn_R_34_FPN_1x',
                    'e2e_faster_rcnn_R_34_FPN_1x_tb',
                    'e2e_faster_rcnn_R_50_FPN_1x',
                    'e2e_faster_rcnn_R_101_FPN_1x',
                    'e2e_faster_rcnn_X_101_32x8d_FPN_1x',
                    'retinanet_R-50-FPN_1x',
                    'retinanet_R-50-FPN_P5_1x',
                    'retinanet_R-101-FPN_1x',
                    # not ready
                    #'retinanet_X_101_32x8d_FPN_1x',
                    ]),
                ('max_iter', ['20e', '40e']),
                ('effective_batch_size', [ebs]),
                ('expid_prefix', ['M']),
                ('log_step', [100]),
                ('base_lr', [0.01]),
                ('evaluate_method', ['map']),
                ('pipeline_type', ['MaskRCNNPipeline']),
                ]

        self.update_swap_params_by_param(swap_params, param)

        all_param = []
        for swap_param in iter_swap_param(swap_params):
            p = {}
            dict_update_nested_dict(p, swap_param)
            if p['net'] in ['e2e_faster_rcnn_R_34_FPN_1x',
                    'e2e_faster_rcnn_R_34_FPN_1x_tb']:
                p['init_full_expid'] = 'coco2017Full_{}_M_BS16_MaxIter90000_LR0.02'.format(p['net'])
            elif p['net'] in ['e2e_faster_rcnn_R_50_FPN_1x',
                    'e2e_faster_rcnn_R_101_FPN_1x',
                    ]:
                p['init_full_expid'] = 'coco2017Full_{}_M_BS16_MaxIter90000'.format(p['net'])
            elif p['net'] in [
                    'e2e_faster_rcnn_X_101_32x8d_FPN_1x',
                    'retinanet_R-50-FPN_1x',
                    'retinanet_R-50-FPN_P5_1x',
                    'retinanet_R-101-FPN_1x',
                    'retinanet_X_101_32x8d_FPN_1x',
                    ]:
                p['init_full_expid'] = 'coco2017Full_{}_M_BS8_MaxIter180000'.format(p['net'])
            else:
                raise NotImplementedError
            p = get_maskrcnn_param(p)
            all_param.append(p)
        return all_param

    def get_maskrcnn_params_luxiyangperson(self, param):
        ebs = self.get_maskrcnn_effective_bath_size()
        data = param['data']
        swap_params = [
            ('data', [data]),
            ('net', ['e2e_faster_rcnn_R_34_FPN_fast',]),
            ('max_iter', ['20e', '40e']),
            ('effective_batch_size', [ebs]),
            ('expid_prefix', ['M']),
            ('log_step', [100]),
            ('base_lr', [0.01]),
            ('basemodel', ['output/LuXiyangHuman/snapshot/resnet34-v2-human-54.7.pth']),
            ('evaluate_method', ['map']),
            ('pipeline_type', ['MaskRCNNPipeline']),
        ]
        all_param = []
        for swap_param in iter_swap_param(swap_params):
            p = {}
            dict_update_nested_dict(p, swap_param)
            p = get_maskrcnn_param(p)
            all_param.append(p)
        return all_param

    def get_maskrcnn_param_basic_with_test(self, param):
        all_param = self.get_maskrcnn_params_basic(param)
        all_test_data = get_all_test_data(param['data'])
        result = []
        for p in all_param:
            result.append((p, all_test_data))
        return result

    def get_maskrcnn_params_with_test(self, param, template=None):
        if template is None:
            all_param = self.get_maskrcnn_param_basic_with_test(param)
        elif template == 'init_from_coco':
            raise NotImplementedError
            all_param = self.get_maskrcnn_params_coco_init(param)
        elif template == 'luxiyangperson':
            assert len(param) == 1 and 'data' in param
            all_param = self.get_maskrcnn_params_luxiyangperson(param)
            all_test_data = get_all_test_data(param['data'])
            result = []
            for p in all_param:
                result.append((p, all_test_data))
            all_param = result
        else:
            raise NotImplementedError
        return all_param

    def get_yolov2pt_param_with_test_auto_scale_by_close(self, param):
        closeness = get_gt_closeness(param['data'], 'train', 416)
        factor = 32. / closeness
        lower = factor - 1
        higher = factor + 1
        input_size = (int(factor * 416) + 31) // 32 * 32
        all_input_size = [input_size, input_size - 128, input_size + 128]
        all_lower_higher = []
        all_lower_higher.append((lower, higher))
        all_lower_higher.append((lower - 1, higher))
        all_lower_higher.append((lower + 1, higher))
        all_lower_higher.append((lower, higher - 1))
        all_lower_higher.append((lower, higher + 1))
        all_lower_higher = [(l, h) for l, h in all_lower_higher if l > 0]
        set_if_not_exist(param, 'max_iter', '20e')
        basic_params = self.get_yolov2pt_params_basic(param)
        all_test_data = get_all_test_data(param['data'])
        all_param = []
        for p in basic_params:
            for l, h in all_lower_higher:
                curr_p = copy.deepcopy(p)
                dict_update_nested_dict(curr_p,
                        {'yolo_train_session_param': {'data_augmentation': {'box_data_param': {
                                'random_scale_min': round(l, 1),
                                'random_scale_max': round(h, 1),
                                }}},
                        })
                curr_all_test_data = []
                for i in all_input_size:
                    for test_data in all_test_data:
                        curr_test_data = copy.deepcopy(test_data)
                        dict_update_nested_dict(curr_test_data,
                                {'yolo_predict_session_param':
                                    {'test_input_size': i}})
                        curr_all_test_data.append(curr_test_data)
                all_param.append((curr_p, curr_all_test_data))
        return all_param

    def get_from_swap_config(self, param):
        configs = load_from_yaml_file('./aux_data/auto_param/swap_param.yaml')
        result = []
        for config in configs:
            curr_param = copy.deepcopy(param)
            for k, v in config.items():
                if k not in curr_param:
                    curr_param[k] = v
            result.extend(iter_swap_param_simple(curr_param))
        from qd.qd_common import hash_sha1
        hash2result = {hash_sha1(r): r for r in result}
        valid_keys = list(set(hash2result.keys()))
        return [hash2result[k] for k in valid_keys]

    def get_yolov2pt_params_with_test(self, param, template=None):
        if template is None:
            all_param = self.get_yolov2pt_params_basic(param)
            all_test_data = get_all_test_data(param['data'])
            result = []
            for p in all_param:
                result.append((p, all_test_data))
            all_param = result
            return all_param
        elif template == 'auto_scale_by_closeness':
            return self.get_yolov2pt_param_with_test_auto_scale_by_close(param)
        else:
            raise NotImplementedError

    def get_yolov2pt_params_basic(self, param):
        # TODO: check if there is any tree structure
        ebs_yolo = self.get_yolo_effective_batch_size()
        swap_params = [
                ('net', ['darknet19_448',]),
                ('effective_batch_size', [ebs_yolo]),
                ('max_iter', ['20e', '40e']),
                ('use_treestructure', [False]),
                ('yolo_train_session_param', [
                    {'data_augmentation': {'box_data_param': {'max_trials': 50}},
                     'use_maskrcnn_sampler': True },
                    ]),
                ('yolo_predict_session_param', [
                    {'test_input_size': 416},
                    ]),
                ('expid_prefix', ['YoloV2Pt2']),
                ('basemodel',
                    ['./output/Tax1300V14.4_0.0_0.0_darknet19_448_C_Init.best_model6933_maxIter.10eEffectBatchSize128LR7580_bb_only_yolov2pt/snapshot/model_iter_139900.pt']),
                ('num_extra_convs', [2]),
                ('pipeline_type', ['YoloV2PtPipeline']),
                ]
        self.update_swap_params_by_param(swap_params, param)

        all_param = []
        for swap_param in iter_swap_param(swap_params):
            param = {}
            dict_update_nested_dict(param, swap_param)
            param = get_yolov2pt_param(param)
            all_param.append(param)
        return all_param

