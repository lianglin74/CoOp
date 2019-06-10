# import torch first because it can help resolve some symbolic issues: 
# Once your extension is built, you can simply import it in Python, using the name you specified in your setup.py script. Just be sure to import torch first, as this will resolve some symbols that the dynamic linker must see
# https://pytorch.org/tutorials/advanced/cpp_extension.html
import torch
import os.path as op
import logging
from pprint import pformat
import copy
import base64
import re
from future.utils import viewitems
from collections import OrderedDict
import copy

from qd.process_tsv import convertcomposite_to_standard
from qd.cloud_storage import create_cloud_storage
from qd.philly import create_philly_client
from qd.philly import convert_to_philly_extra_command
from qd.tsv_io import TSVDataset
from qd.qd_common import make_by_pattern_maker
from qd.qd_common import make_by_pattern_result
from qd.qd_common import load_from_yaml_file
from qd.qd_common import hash_sha1
from qd.qd_common import init_logging
from qd.qd_common import dict_has_path, dict_get_path_value, dict_get_all_path


def get_all_test_data(exp):
    pattern_to_test_datas = load_from_yaml_file('./aux_data/exp/pattern_to_test_datas.yaml')
    result = make_by_pattern_result(exp, pattern_to_test_datas)
    if result is None:
        result = [{'test_data': exp, 'test_split': 'test'}]
    return result

def get_all_full_expid_by_data(exp):
    pattern_to_test_datas = load_from_yaml_file('./aux_data/exp/pattern_to_full_expids.yaml')
    return make_by_pattern_result(exp, pattern_to_test_datas)

def get_all_related_data_for_philly_jobs(data):
    all_data = []
    all_data.append(data)
    if data.startswith('Tax') and not data.endswith('_with_bb'):
        all_data.append(data + '_with_bb')
        all_data.append(data + '_no_bb')
    all_test_data = get_all_test_data(data)
    all_data.extend([info['test_data'] for info in all_test_data])
    return all_data

def ensure_upload_data_for_philly_jobs(data, philly_client):
    all_data = get_all_related_data_for_philly_jobs(data)
    for d in all_data:
        # we need the hw for mask-rcnn
        logging.info('TODO: making sure hw is there')
    need_to_upload_data = copy.deepcopy(all_data)

    for d in all_data:
        dataset = TSVDataset(d)
        for split in ['train', 'test', 'trainval']:
            splitx = split + 'X'
            if op.isfile(dataset.get_data(splitx)) and d.endswith('_with_bb'):
                data_sources = dataset.load_composite_source_data_split(split)
                if len(data_sources) > 1:
                    if not philly_client.qd_data_exists('{}/{}.tsv'.format(d, split)):
                        convertcomposite_to_standard(d, split)
                elif len(data_sources) == 1:
                    need_to_upload_data.append(data_sources[0][0])

    for d in need_to_upload_data:
        # previously, we listed more than required datas to upload. some of
        # them might not be valid. For example, if data is TaxXV1_1_with_bb, it
        # might also include TaxXV1_1_with_bb_with_bb
        if op.isdir(TSVDataset(d)._data_root):
            philly_client.upload_qd_data(d)

def ensure_upload_init_model(param):
    if 'basemodel' not in param:
        return
    basemodel = param['basemodel']
    target_path = op.join('jianfw', 'work',
            basemodel.replace('output/', 'qd_output/'))
    c = create_cloud_storage('vig')
    if not c.exists(target_path):
        c.az_upload2(basemodel, target_path)

def philly_func_run(func, param, **submit_param):
    from qd.philly import create_multi_philly_client
    client = create_multi_philly_client(**submit_param)
    philly_client = client.select_client_for_submit()
    if 'data' in param.get('param', {}):
        ensure_upload_data_for_philly_jobs(param['param']['data'], philly_client)
        # TODO: the following only uploads data to blob. implement to upload to
        # hdfs also if philly still supports that
        ensure_upload_init_model(param)
    assert func.__module__ != '__main__'
    assert 'type' not in param
    param['type'] = func.__name__
    #param['gpus'] = list(range(client.num_gpu))
    if hasattr(func, 'func_code'):
        # py2
        code_file_name = func.func_code.co_filename
    else:
        # py3
        code_file_name = func.__code__.co_filename
    extra_param = convert_to_philly_extra_command(param,
            script=op.relpath(code_file_name))
    logging.info(extra_param)
    logging.info(philly_client.get_config_extra_param(extra_param))
    philly_client.submit_without_sync(extra_param)

def except_to_update_for_stageiter(param):
    if param.get('net', '').startswith('e2e'):
        stageiter = param.get('stageiter', (0, 0))
        max_iter = param.get('max_iter', 0)
        if len(stageiter) == 2 and \
                stageiter[0] == 6 * max_iter // 9 and \
                stageiter[1] == 8 * max_iter // 9:
            return True
    return False

def except_to_update_for_version(param):
    return param.get('MaskTSVDataset', {}).get('version') in [0, None]

def except_to_update_for_remove_bg_image(param):
    return param.get('MaskTSVDataset', {}).get('remove_images_without_annotations', True)

def except_to_update_for_data_augmentation(param):
    return type(param.get('yolo_train_session_param',
        {}).get('data_augmentation')) is not str

def except_to_update_for_dtype(param):
    return param.get('DTYPE', 'float32') == 'float32'

def except_to_update_classification_loss(param):
    if not dict_has_path(param, 'MODEL$ROI_BOX_HEAD$CLASSIFICATION_LOSS'):
        return True
    else:
        return dict_get_path_value(param,
                'MODEL$ROI_BOX_HEAD$CLASSIFICATION_LOSS') == 'CE'

def except_to_update_random_scale_min(param):
    if dict_has_path(param, 'INPUT$FIXED_SIZE_AUG$RANDOM_SCALE_MIN'):
        value = dict_get_path_value(param, 'INPUT$FIXED_SIZE_AUG$RANDOM_SCALE_MIN')
        return value == 1
    else:
        return True

def except_to_update_random_scale_max(param):
    if dict_has_path(param,
            'INPUT$FIXED_SIZE_AUG$RANDOM_SCALE_MAX'):
        value = dict_get_path_value(param,
            'INPUT$FIXED_SIZE_AUG$RANDOM_SCALE_MAX')
        return value == 1
    else:
        return True

def update_parameters(param):
    default_param = {
            'max_iter': 10000,
            'effective_batch_size': 64}

    direct_add_value_keys = [
            # first value is the key, the second is the name in the folder; the
            # third is the excpdetion condidtion
            ('MaskTSVDataset$version', 'V', except_to_update_for_version),
            ('MaskTSVDataset$remove_images_without_annotations', 'RemoveEmpty',
                except_to_update_for_remove_bg_image),
            ('MODEL$WEIGHT', 'init'),
            ('', ''),
            ('effective_batch_size', 'BS'),
            ('max_iter', 'MaxIter'),
            ('max_epoch', 'MaxEpoch'),
            ('last_fixed_param', 'LastFixed'),
            ('num_extra_convs', 'ExtraConv'),
            ('yolo_train_session_param$data_augmentation', 'Aug',
                except_to_update_for_data_augmentation),
            ('yolo_train_session_param$data_augmentation$box_data_param$max_trials',
                'AugTrials'),
            ('yolo_train_session_param$data_augmentation$box_data_param$random_scale_min',
                'ScaleMin'),
            ('yolo_train_session_param$data_augmentation$box_data_param$random_scale_max',
                'ScaleMax'),
            ('momentum', 'Momentum'),
            ('base_lr', 'LR'),
            ('stageiter', 'StageIter', except_to_update_for_stageiter),
            ('INPUT$MIN_SIZE_TRAIN', 'Min'),
            ('INPUT$MAX_SIZE_TRAIN', 'Max'),
            ('DTYPE', 'T', except_to_update_for_dtype),
            ('INPUT$FIXED_SIZE_AUG$RANDOM_SCALE_MIN', 'ScaleMin',
                except_to_update_random_scale_min),
            ('INPUT$FIXED_SIZE_AUG$RANDOM_SCALE_MAX', 'ScaleMax',
                except_to_update_random_scale_max),
            ('MODEL$ROI_BOX_HEAD$CLASSIFICATION_LOSS', '',
                except_to_update_classification_loss),
            ('min_size_range32', 'Min'),
            ('with_dcn', ('DCN', None)),
            ]

    non_expid_impact_keys = ['data', 'net', 'expid_prefix',
            'test_data', 'test_split', 'test_version',
            'dist_url_tcp_port', 'workers', 'force_train',
            'pipeline_type', 'test_batch_size',
            'yolo_train_session_param$debug_train',
            'yolo_predict_session_param',
            'evaluate_method', 'debug_train',
            'full_expid', 'log_step', 'MODEL$DEVICE',
            'MODEL$ROI_BOX_HEAD$CLASSIFICATION_ACTIVATE',
            'display',
            'apply_nms_gt',
            ]
    true_false_keys = OrderedDict([('use_treestructure', ('Tree', None)),
        ('MODEL$USE_TREESTRUCTURE', ('Tree', None)),
        ('MaskTSVDataset$multi_hot_label', ('MultiHot', None)),
        ('DATALOADER$ASPECT_RATIO_GROUPING', [None, 'NoARG']),
        ('INPUT$USE_FIXED_SIZE_AUGMENTATION', ['FSize', None]),
        ])

    if param['pipeline_type'] == 'MaskRCNNPipeline':
        non_expid_impact_keys.extend(['DATASETS', ''])

    for k, v in viewitems(default_param):
        assert k in param, 'set default outside'

    if 'expid' in param:
        return
    # we need to update expid so that the model folder contains the critical
    # param information
    infos = []
    need_hash_sha_params = ['basemodel']
    for k in need_hash_sha_params:
        if k in param:
            infos.append('{}{}'.format(k, hash_sha1(param[k])[:5]))
    for setting in direct_add_value_keys:
        k, v = setting[:2]
        if dict_has_path(param, k):
            pk = dict_get_path_value(param, k)
            if len(setting) == 3 and setting[2](param):
                continue
            if type(pk) is str:
                pk = pk.replace('/', '.')
            elif type(pk) in [list, tuple]:
                pk = '.'.join(map(str, pk))
            if type(v) is tuple:
                assert pk in [True, False]
                assert len(v) == 2
                if pk and v[0]:
                    infos.append('{}'.format(v[0]))
                elif not pk and v[1]:
                    infos.append('{}'.format(v[1]))
                continue
            else:
                infos.append('{}{}'.format(v, pk))
    for k in true_false_keys:
        if dict_has_path(param, k):
            if dict_get_path_value(param, k) and true_false_keys[k][0]:
                infos.append(true_false_keys[k][0])
            elif not dict_get_path_value(param, k) and true_false_keys[k][1]:
                infos.append(true_false_keys[k][1])

    known_keys = []
    known_keys.extend((k for k in need_hash_sha_params))
    known_keys.extend((k for k in non_expid_impact_keys))
    known_keys.extend((s[0] for s in direct_add_value_keys))
    known_keys.extend((k for k in true_false_keys))

    all_path = dict_get_all_path(param)

    invalid_keys = [k for k in all_path
        if all(k != n and not k.startswith(n + '$') for n in known_keys)]

    assert len(invalid_keys) == 0, pformat(invalid_keys)

    if 'expid_prefix' in param:
        infos.insert(0, param['expid_prefix'])
    param['expid'] = '_'.join(infos)

def create_pipeline(kwargs):
    pipeline_type = kwargs.get('pipeline_type', 'YoloV2PtPipeline')
    if pipeline_type == 'YoloV2PtPipeline':
        from qd.qd_pytorch import YoloV2PtPipeline
        return YoloV2PtPipeline(**kwargs)
    elif pipeline_type == 'MaskRCNNPipeline':
        from qd.qd_maskrcnn import MaskRCNNPipeline
        return MaskRCNNPipeline(**kwargs)
    elif pipeline_type == 'MMDetPipeline':
        from qd.qd_mmdetection import MMDetPipeline
        return MMDetPipeline(**kwargs)

def load_pipeline(**kwargs):
    from qd.qd_pytorch import load_latest_parameters
    kwargs_f = load_latest_parameters(op.join('output',
        kwargs['full_expid']))
    from qd.qd_common import dict_update_nested_dict
    dict_update_nested_dict(kwargs_f, kwargs)
    return create_pipeline(kwargs_f)

def pipeline_train_eval_multi(all_test_data, param, **kwargs):
    init_logging()
    curr_param = copy.deepcopy(param)
    if len(all_test_data) > 0:
        curr_param.update(all_test_data[0])
    pip = create_pipeline(curr_param)
    pip.ensure_train()
    param['full_expid'] = pip.full_expid
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(param)
        curr_param.update(test_data_info)
        pip = load_pipeline(**curr_param)
        pip.ensure_predict()
        pip.ensure_evaluate()

def pipeline_eval_multi(param, all_test_data, **kwargs):
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(param)
        curr_param.update(test_data_info)
        pip = load_pipeline(**curr_param)
        pip.ensure_predict()
        pip.ensure_evaluate()

def test_model_pipeline(param):
    '''
    run the script by

    mpirun -npernode 4 \
            python script_with_this_function_called.py
    '''
    init_logging()
    update_parameters(param)
    pip = create_pipeline(param)

    if param.get('monitor_train_only'):
        pip.monitor_train()
    else:
        pip.ensure_train()
        pip.ensure_predict()
        pip.ensure_evaluate()

if __name__ == '__main__':
    from qd.qd_common import parse_general_args
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
