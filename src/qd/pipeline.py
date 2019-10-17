# import torch first because it can help resolve some symbolic issues:
# Once your extension is built, you can simply import it in Python, using the name you specified in your setup.py script. Just be sure to import torch first, as this will resolve some symbols that the dynamic linker must see
# https://pytorch.org/tutorials/advanced/cpp_extension.html
import torch
from deprecated import deprecated
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
from qd.gpucluster import create_philly_client
from qd.gpucluster import convert_to_philly_extra_command
from qd.tsv_io import TSVDataset
from qd.qd_common import make_by_pattern_maker
from qd.qd_common import make_by_pattern_result
from qd.qd_common import load_from_yaml_file
from qd.qd_common import hash_sha1
from qd.qd_common import init_logging
from qd.qd_common import dict_has_path, dict_get_path_value, dict_get_all_path
from qd.qd_common import dict_update_nested_dict
from qd.qd_common import dict_ensure_path_key_converted


def get_all_test_data(exp):
    pattern_to_test_datas = load_from_yaml_file('./aux_data/exp/pattern_to_test_datas.yaml')
    result = make_by_pattern_result(exp, pattern_to_test_datas)
    if result is None:
        result = [{'test_data': exp, 'test_split': 'test'}]
    return result

def get_all_full_expid_by_data(exp):
    pattern_to_test_datas = load_from_yaml_file('./aux_data/exp/pattern_to_full_expids.yaml')
    return make_by_pattern_result(exp, pattern_to_test_datas)

@deprecated('use get_all_related_data_for_gpu_jobs')
def get_all_related_data_for_philly_jobs(data):
    return get_all_related_data_for_gpu_jobs(data)

def get_all_related_data_for_gpu_jobs(data):
    all_data = []
    all_data.append(data)
    if data.startswith('Tax') and not data.endswith('_with_bb'):
        all_data.append(data + '_with_bb')
        all_data.append(data + '_no_bb')
    all_test_data = get_all_test_data(data)
    all_data.extend([info['test_data'] for info in all_test_data])
    return all_data

@deprecated('use ensure_upload_data_for_gpu_jobs')
def ensure_upload_data_for_philly_jobs(data, philly_client):
    return ensure_upload_data_for_gpu_jobs(data, philly_client)

def ensure_upload_data_for_gpu_jobs(data, philly_client):
    all_data = get_all_related_data_for_gpu_jobs(data)
    ensure_upload_all_data(all_data, philly_client)

def ensure_upload_all_data(all_data, philly_client):
    for d in all_data:
        dataset = TSVDataset(d)
        if any(dataset.has('train') and not dataset.has('train', 'hw')
                for split in ['train', 'trainval', 'test']):
            from qd.process_tsv import populate_dataset_details
            populate_dataset_details(d, check_image_details=True)
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

def ensure_upload_init_model(param, philly_client):
    if 'basemodel' not in param:
        return
    basemodel = param['basemodel']
    if basemodel == '' or basemodel.startswith('http'):
        logging.info('No need to upload base model')
        return
    assert(op.isfile(basemodel))
    philly_client.upload_qd_model(basemodel)

def ensure_upload_trained_model(param, aml_client):
    if 'full_expid' in param:
        full_expid = param['full_expid']
        aml_client.sync_full_expid_from_local_by_exist(full_expid)

def aml_func_run(func, param, **submit_param):
    from qd.gpucluster import create_aml_client
    aml_client = create_aml_client(**submit_param)
    if 'data' in param.get('param', {}):
        from qd.pipeline import get_all_related_data_for_gpu_jobs
        data = param['param']['data']
        all_data = get_all_related_data_for_gpu_jobs(data)
        if 'all_test_data' in param:
            all_data.extend([test_data_info['test_data'] for test_data_info in
                param['all_test_data']])
        ensure_upload_all_data(all_data, aml_client)
    if 'basemodel' in param.get('param', {}):
        ensure_upload_init_model(param['param'], aml_client)
    if 'param' in param:
        ensure_upload_trained_model(param['param'], aml_client)
    assert func.__module__ != '__main__', \
            'the executed func should not be in the main module'
    assert 'type' not in param
    param['type'] = func.__name__
    #param['gpus'] = list(range(client.num_gpu))
    if hasattr(func, 'func_code'):
        # py2
        code_file_name = func.func_code.co_filename
    else:
        # py3
        code_file_name = func.__code__.co_filename
    from qd.qd_common import convert_to_command_line
    extra_param = convert_to_command_line(param,
            script=op.relpath(code_file_name))
    logging.info(extra_param)
    aml_client.submit(extra_param)

def philly_func_run(func, param, **submit_param):
    from qd.philly import create_multi_philly_client
    client = create_multi_philly_client(**submit_param)
    philly_client = client.select_client_for_submit()
    if 'data' in param.get('param', {}):
        ensure_upload_data_for_gpu_jobs(param['param']['data'], philly_client)
        # TODO: the following only uploads data to blob. implement to upload to
        # hdfs also if philly still supports that
    if 'basemodel' in param.get('param', {}):
        ensure_upload_init_model(param['param'], philly_client)
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
    if dict_has_path(param, 'MODEL$ROI_BOX_HEAD$CLASSIFICATION_LOSS'):
        return dict_get_path_value(param,
                'MODEL$ROI_BOX_HEAD$CLASSIFICATION_LOSS') == 'CE'
    elif 'classification_loss_type' in param:
        return param['classification_loss_type'] == 'CE'
    else:
        return True


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

def except_to_update_fixed_input_size(param):
    if dict_has_path(param, 'INPUT$FIXED_SIZE_AUG$INPUT_SIZE'):
        value = dict_get_path_value(param, 'INPUT$FIXED_SIZE_AUG$INPUT_SIZE')
        return value == 800
    else:
        return True

def except_to_update_fixed_jitter(param):
    if dict_has_path(param, 'INPUT$FIXED_SIZE_AUG$JITTER'):
        value = dict_get_path_value(param, 'INPUT$FIXED_SIZE_AUG$JITTER')
        return value == 0.2
    else:
        return True

def except_to_update_lr_policy(param):
    if dict_has_path(param, 'SOLVER$LR_POLICY'):
        value = dict_get_path_value(param, 'SOLVER$LR_POLICY')
        return value == 'multistep'
    else:
        return True

def except_to_update_bb_loss_type(param):
    if dict_has_path(param, 'MODEL$ROI_BOX_HEAD$BOUNDINGBOX_LOSS_TYPE'):
        value = dict_get_path_value(param, 'MODEL$ROI_BOX_HEAD$BOUNDINGBOX_LOSS_TYPE')
        return value == 'SL1'
    else:
        return True

def update_parameters(param):
    logging.info('use populate_expid')
    populate_expid(param)

def populate_expid(param):
    if 'expid' in param:
        return
    param['expid'] = generate_expid(param)

def generate_expid(param):
    dict_ensure_path_key_converted(param)
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
            ('yolo_train_session_param$data_augmentation$box_data_param$max_boxes',
                'MB'),
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
            ('INPUT$FIXED_SIZE_AUG$INPUT_SIZE', 'In',
                except_to_update_fixed_input_size),
            ('INPUT$FIXED_SIZE_AUG$JITTER', 'J',
                except_to_update_fixed_jitter),
            ('MODEL$ROI_BOX_HEAD$CLASSIFICATION_LOSS', '',
                except_to_update_classification_loss),
            ('classification_loss_type', '',
                except_to_update_classification_loss),
            ('min_size_range32', 'Min'),
            ('with_dcn', ('DCN', None)),
            ('opt_cls_only', ('ClsOnly', None)),
            ('MODEL$FPN$USE_GN', ('FpnGN', None)),
            ('bn_momentum', 'BNMoment'),
            ('use_hvd', ('HVD', None)),
            ('yolo_train_session_param$use_maskrcnn_sampler', ('MaskSam', None)),
            ('use_treestructure', ('Tree', None)),
            ('MODEL$USE_TREESTRUCTURE', ('Tree', None)),
            ('MaskTSVDataset$multi_hot_label', ('MultiHot', None)),
            ('multi_hot_label', ('MultiHot', None)),
            ('DATALOADER$ASPECT_RATIO_GROUPING', [None, 'NoARG']),
            ('INPUT$USE_FIXED_SIZE_AUGMENTATION', ['FSize', None]),
            ('dataset_type', ''),
            ('step_lr', 'StepLR'),
            ('MODEL$RPN$USE_BN', ('RpnBN', None)),
            ('MODEL$ROI_BOX_HEAD$USE_GN', ('HeadGN', None)),
            ('MODEL$FPN$USE_BN', ('FPNBN', None)),
            ('sync_bn', ('SyncBN', None)),
            ('SOLVER$LR_POLICY', 'LRP', except_to_update_lr_policy),
            ('SOLVER$WARMUP_ITERS', 'Warm'),
            ('MODEL$BACKBONE$FREEZE_CONV_BODY_AT', 'Freeze'),
            ('MODEL$ROI_BOX_HEAD$BOUNDINGBOX_LOSS_TYPE', '', except_to_update_bb_loss_type),
            ('MODEL$RESNETS$STEM_FUNC', '', {'default_value': 'StemWithFixedBatchNorm'}),
            ('MODEL$RESNETS$TRANS_FUNC', '', {'default_value': 'BottleneckWithFixedBatchNorm'}),
            ('MODEL$FPN$USE_RELU', ('FPNRelu', None)),
            ('init_model_only', (None, 'Continue')),
            ('MODEL$RESNETS$USE_SE', ('SE', None)),
            ('SOLVER$GAMMA', 'Gamma'),
            ('SOLVER$WEIGHT_DECAY', 'WD', lambda x: dict_get_path_value(x,
                'SOLVER$WEIGHT_DECAY') == 1e-4),
            ('use_apex_ddp', ('ADDP', None)),
            ('SOLVER$STEPS', 'S'),
            ('exclude_convert_gn', ('NoSyncGN', None)),
            ('ignore_filter_img', ('NoFilter', None)),
            ('pretrained', ('Pretrained', None)),
            ('bgr2rgb', ('RGB', None)),
            ('MODEL$FPN$INTERPOLATE_MODE', 'Inp'),
            ('use_ddp', (None, 'NoneDDP')),
            ('data_partition', 'P', lambda x: x['data_partition'] == 1),
            ('MODEL$ROI_BOX_HEAD$MLP_HEAD_DIM', 'Head', lambda x: dict_get_path_value(x, 'MODEL$ROI_BOX_HEAD$MLP_HEAD_DIM') == 1024),
            ('MODEL$RPN$MATCHER_TYPE', 'RpnM', lambda x: dict_get_path_value(x, 'MODEL$RPN$MATCHER_TYPE') == 'default'),
            ('MODEL$ROI_HEADS$MATCHER_TYPE', 'RoiM', lambda x: dict_get_path_value(x, 'MODEL$ROI_HEADS$MATCHER_TYPE') == 'default'),
            ('MODEL$ROI_HEADS$BG_IOU_THRESHOLD', 'RoiBG', lambda x: dict_get_path_value(x, 'MODEL$ROI_HEADS$BG_IOU_THRESHOLD') == 0.5),
            ('MODEL$RPN$BG_IOU_THRESHOLD', 'RpnBG', lambda x: dict_get_path_value(x, 'MODEL$RPN$BG_IOU_THRESHOLD') == 0.3),
            ('yolo_train_session_param$use_maskrcnn_trainer', ('MT', None)),
            ('rt_param$xy_scale', 'XY'),
            ('rt_param$wh_scale', 'WH'),
            ('rt_param$object_scale', 'O'),
            ('rt_param$valid_norm_xywhpos', ('VNXYWHPos', None)),
            ]

    non_expid_impact_keys = ['data', 'net', 'expid_prefix',
            'test_data', 'test_split', 'test_version',
            'dist_url_tcp_port', 'workers', 'force_train',
            'num_workers',
            'pipeline_type', 'test_batch_size',
            'yolo_train_session_param$debug_train',
            'yolo_predict_session_param',
            'evaluate_method', 'debug_train',
            'full_expid', 'log_step', 'device',
            'MODEL$ROI_BOX_HEAD$CLASSIFICATION_ACTIVATE',
            'display',
            'apply_nms_gt',
            'apply_nms_det',
            'expand_label_det',
            'SOLVER$CHECKPOINT_PERIOD',
            'yolo_train_session_param$display',
            'INPUT$MIN_SIZE_TEST',
            'ovthresh',
            'MODEL$ROI_HEADS$NMS_POLICY$TYPE',
            'images_per_gpu',
            'yolo_predict_session_param$nms_threshold',
            'env',
            'test_max_iter',
            'test_input_size',
            ]

    if param['pipeline_type'] == 'MaskRCNNPipeline':
        non_expid_impact_keys.extend(['DATASETS', ''])

    for k, v in viewitems(default_param):
        assert k in param, 'set default outside'

    # we need to update expid so that the model folder contains the critical
    # param information
    infos = []
    need_hash_sha_params = ['basemodel', 'lock_up_to',
            'yolo_train_session_param$anchors', 'stagelr']
    for k in need_hash_sha_params:
        if dict_has_path(param, k):
            v = dict_get_path_value(param, k)
            if len(v) > 5 or isinstance(v, list) or isinstance(v, tuple):
                infos.append('{}{}'.format(k.split('$')[-1],
                    hash_sha1(v)[:5]))
            else:
                infos.append('{}{}'.format(k.split('$')[-1],
                    v))

    for setting in direct_add_value_keys:
        k, v = setting[:2]
        if dict_has_path(param, k):
            pk = dict_get_path_value(param, k)
            if len(setting) == 3:
                if isinstance(setting[2], dict):
                    if setting[2]['default_value'] == pk:
                        continue
                elif setting[2](param):
                    continue
            if type(pk) is str:
                pk = pk.replace('/', '.')
            elif type(pk) in [list, tuple]:
                pk = '.'.join(map(str, pk))
            if type(v) is tuple or type(v) is list:
                assert pk in [True, False]
                assert len(v) == 2
                if pk and v[0]:
                    infos.append('{}'.format(v[0]))
                elif not pk and v[1]:
                    infos.append('{}'.format(v[1]))
                continue
            else:
                infos.append('{}{}'.format(v, pk))

    known_keys = []
    known_keys.extend((k for k in need_hash_sha_params))
    known_keys.extend((k for k in non_expid_impact_keys))
    known_keys.extend((s[0] for s in direct_add_value_keys))

    all_path = dict_get_all_path(param)

    invalid_keys = [k for k in all_path
        if all(k != n and not k.startswith(n + '$') for n in known_keys)]

    assert len(invalid_keys) == 0, pformat(invalid_keys)

    if 'expid_prefix' in param:
        infos.insert(0, param['expid_prefix'])
    return '_'.join(infos)

def create_pipeline(kwargs):
    pipeline_type = kwargs.get('pipeline_type')
    if pipeline_type == 'YoloV2PtPipeline':
        from qd.pipelines.yolov2_pt import YoloV2PtPipeline
        return YoloV2PtPipeline(**kwargs)
    elif pipeline_type == 'MaskRCNNPipeline':
        from qd.qd_maskrcnn import MaskRCNNPipeline
        return MaskRCNNPipeline(**kwargs)
    elif pipeline_type == 'MMDetPipeline':
        from qd.pipelines.qd_mmdetection import MMDetPipeline
        return MMDetPipeline(**kwargs)
    elif pipeline_type == 'classification':
        from qd.qd_pytorch import TorchTrain
        return TorchTrain(**kwargs)
    elif pipeline_type == 'classification_for_mask':
        from qd.pipelines.classification_for_maskrcnn import ClassificationForMaskRCNN
        return ClassificationForMaskRCNN(**kwargs)
    elif pipeline_type == 'classification_by_mask':
        from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
        return MaskClassificationPipeline(**kwargs)
    elif pipeline_type == 'YoloByMask':
        from qd.pipelines.yolo_by_mask import YoloByMask
        return YoloByMask(**kwargs)
    else:
        raise NotImplementedError()

def load_pipeline(**kwargs):
    from qd.qd_pytorch import load_latest_parameters
    kwargs = copy.deepcopy(kwargs)
    kwargs_f = load_latest_parameters(op.join('output',
        kwargs['full_expid']))
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
        dict_ensure_path_key_converted(test_data_info)
        dict_update_nested_dict(curr_param, test_data_info)
        pip = load_pipeline(**curr_param)
        pip.ensure_predict()
        pip.ensure_evaluate()

def pipeline_monitor_train(param, all_test_data, **kwargs):
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(param)
        dict_ensure_path_key_converted(test_data_info)
        dict_update_nested_dict(curr_param, test_data_info)
        pip = load_pipeline(**curr_param)
        pip.monitor_train()

def pipeline_continuous_train(param, all_test_data, **kwargs):
    init_logging()
    curr_param = copy.deepcopy(param)
    pip = load_pipeline(**curr_param)
    pip.ensure_train()

    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(param)
        dict_ensure_path_key_converted(test_data_info)
        dict_update_nested_dict(curr_param, test_data_info)
        pip = load_pipeline(**curr_param)
        pip.ensure_predict()
        pip.ensure_evaluate()

def pipeline_eval_multi(param, all_test_data, **kwargs):
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(param)
        dict_ensure_path_key_converted(test_data_info)
        dict_update_nested_dict(curr_param, test_data_info)
        pip = load_pipeline(**curr_param)
        # we should check here instead of before for-loop since we can alter
        # the value of max_iter to just evaluate the intermediate model or take
        # the intermediate model as the final model
        if not pip.is_train_finished():
            logging.info('the model specified by the following is not ready\n{}'.format(
                pformat(param)))
            return
        pip.ensure_predict()
        pip.ensure_evaluate()

def pipeline_demo(param, image_path):
    assert 'full_expid' in param
    pip = load_pipeline(**param)
    pip.demo(image_path)

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
