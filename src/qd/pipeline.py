# import torch first because it can help resolve some symbolic issues:
# Once your extension is built, you can simply import it in Python, using the name you specified in your setup.py script. Just be sure to import torch first, as this will resolve some symbols that the dynamic linker must see
# https://pytorch.org/tutorials/advanced/cpp_extension.html
import torch
from qd.qd_common import cmd_run
from qd.gpucluster.aml_client import AMLClient
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
from qd.qd_common import dict_has_path, dict_get_path_value, get_all_path
from qd.qd_common import dict_update_nested_dict
from qd.qd_common import dict_ensure_path_key_converted
from qd.qd_common import try_once
from qd.qd_common import iter_swap_param
from qd.qd_common import convert_to_command_line
from qd.qd_common import load_list_file
from qd.process_tsv import populate_dataset_hw
from qd.batch_process import BatchProcess
from qd.gpucluster import create_aml_client
from qd.db import create_annotation_db
from qd.qd_common import qd_tqdm as tqdm
from qd.qd_common import write_to_yaml_file
import datetime


def get_all_test_data(exp):
    fname = './aux_data/exp/pattern_to_test_datas.yaml'
    if op.isfile(fname):
        pattern_to_test_datas = load_from_yaml_file(fname)
    else:
        pattern_to_test_datas = []
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

def ensure_upload_all_data(all_data, client, from_cluster=None):
    all_data = list(set(all_data))
    need_to_upload_data = []
    for d in all_data:
        folder = TSVDataset(d)._data_root
        if op.isdir(folder):
            need_to_upload_data.append(folder)

    for d in all_data:
        dataset = TSVDataset(d)
        from qd.tsv_io import get_default_splits
        for split in get_default_splits():
            splitx = split + 'X'
            for t in [None, 'label', 'caption', 'hw']:
                if (op.isfile(dataset.get_data(splitx, t=t)) and
                    not op.isfile(dataset.get_data(split, t=t))):
                    tsv_files = load_list_file(dataset.get_data(splitx, t=t))
                    need_to_upload_data.extend(tsv_files)
                    from qd.tsv_io import get_tsv_lineidx
                    need_to_upload_data.extend([
                        get_tsv_lineidx(t) for t in tsv_files])
    need_to_upload_data = list(set(need_to_upload_data))
    if from_cluster is not None and not isinstance(from_cluster, list):
        from_cluster=[from_cluster]
    ensure_aml_data_available(need_to_upload_data, client, from_cluster)

def ensure_upload_init_model(param, client):
    if 'basemodel' not in param:
        return
    basemodel = param['basemodel']
    if basemodel == '' or basemodel.startswith('http'):
        logging.info('No need to upload base model')
        return
    if client.exists(basemodel):
        return
    assert(op.isfile(basemodel) or op.isdir(basemodel)), basemodel
    client.upload_file(basemodel)

def ensure_upload_trained_model(param, aml_client):
    if 'full_expid' in param:
        full_expid = param['full_expid']
        aml_client.sync_full_expid_from_local_by_exist(full_expid)

def platform_run_knn_classifier(env, param):
    from qd.pipelines.knn_classifier import knn_classifier
    func = knn_classifier
    return env_run(env, func, param)

def smart_create_aml_client(cluster, **kwargs):
    from qd.gpucluster import create_aml_client
    if ',' not in cluster:
        return create_aml_client(cluster=cluster, **kwargs)
    choices = [{'cluster': c} for c in cluster.split(',')]
    db = create_annotation_db()
    # attach free gpus
    for c in choices:
        info = next(db.iter_general('current_cluster', cluster=c['cluster']))
        c.update(info)
    choices = sorted(choices, key=lambda c: -c['total_free_gpu'])

    return create_aml_client(cluster=choices[0]['cluster'], **kwargs)

def ensure_pipeline_data_available(param, client, from_clients=None):
    if 'param' not in param:
        return
    param = param['param']
    if 'basemodel' in param:
        ensure_upload_init_model(param, client)
    # this is for predict pipeline
    ensure_upload_trained_model(param, client)
    needed = []
    if 'text_encoder_type' in param and \
            isinstance(param['text_encoder_type'], str) and \
            op.isdir(param['text_encoder_type']):
        needed.append(param['text_encoder_type'])

    ensure_aml_data_available(needed, client, from_clients)

def ensure_aml_job_data_available(param, **submit_param):
    from qd.qd_common import run_if_not_memory_cached
    aml_client = run_if_not_memory_cached(smart_create_aml_client, **submit_param)
    candidate_clusters = submit_param.get('candidate_clusters', [])
    from_amls = [create_aml_client(cluster=_c) for _c in candidate_clusters]
    all_data = []
    if 'data' in param.get('param', {}):
        from qd.pipeline import get_all_related_data_for_gpu_jobs
        data = param['param']['data']
        all_data.extend(get_all_related_data_for_gpu_jobs(data))
    if 'all_test_data' in param:
        all_data.extend([test_data_info['test_data'] for test_data_info in
            param['all_test_data']])
    run_if_not_memory_cached(
        ensure_upload_all_data,
        all_data,
        aml_client,
        from_amls,
    )
    ensure_pipeline_data_available(param, aml_client, from_clients=from_amls)

def aml_func_run(func, param, **submit_param):
    from qd.qd_common import run_if_not_memory_cached
    aml_client = run_if_not_memory_cached(smart_create_aml_client, **submit_param)
    #if not submit_param.get('skip_data_upload'):
        #ensure_aml_job_data_available(param, **submit_param)
    if isinstance(func, dict):
        param = {
            'info': {
                'from': func['from'],
                'import': func['import'],
                'param': param,
            }}
        from qd.qd_common import execute_func
        param['type'] = execute_func.__name__
        code_file_name = execute_func.__code__.co_filename
    else:
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
    extra_param = convert_to_command_line(param,
            script=op.relpath(code_file_name))
    logging.info(extra_param)
    job_id = aml_client.submit(extra_param)

    job_info = {'appID': job_id}
    job_info.update(submit_param)
    job_info.update(param)
    return job_id

@try_once
def try_inject_submit_info(job_info):
    db = create_annotation_db()
    db.insert_one('ongoingjob', **job_info)

def philly_func_run(func, param, **submit_param):
    philly_client = create_philly_client(**submit_param)
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
    return philly_client.submit_without_sync(extra_param)

def get_default_stageiter(param):
    if param.get('net', '').startswith('e2e'):
        max_iter = param.get('max_iter', 0)
        return (6 * max_iter // 9, 8 * max_iter // 9)

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

def get_pred_file(full_expid, test_data, **kwargs):
    param = {
            'full_expid': full_expid,
            'test_data': test_data,
            }
    param.update(kwargs)
    pip = load_pipeline(**param)
    result = pip._get_predict_file()
    return result

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
    config = load_from_yaml_file('./aux_data/configs/expid_generate.yaml')

    direct_add_value_keys = [
            # first value is the key, the second is the name in the folder; the
            # third is the excpdetion condidtion
            ('MaskTSVDataset$version', 'V', except_to_update_for_version),
            ('yolo_train_session_param$data_augmentation', 'Aug',
                except_to_update_for_data_augmentation),
            ('momentum', 'Momentum'),
            ('DTYPE', 'T', except_to_update_for_dtype),
            ('INPUT$FIXED_SIZE_AUG$INPUT_SIZE', 'In',
                except_to_update_fixed_input_size),
            ('INPUT$FIXED_SIZE_AUG$JITTER', 'J',
                except_to_update_fixed_jitter),
            ('MODEL$ROI_BOX_HEAD$CLASSIFICATION_LOSS', '',
                except_to_update_classification_loss),
            ('classification_loss_type', '',
                except_to_update_classification_loss),
            ('dataset_type', ''),
            ('SOLVER$WARMUP_ITERS', 'Warm'),
            ('MODEL$ROI_BOX_HEAD$BOUNDINGBOX_LOSS_TYPE', '', except_to_update_bb_loss_type),
            ('MODEL$RESNETS$STEM_FUNC', '', {'default_value': 'StemWithFixedBatchNorm'}),
            ('MODEL$RESNETS$TRANS_FUNC', '', {'default_value': 'BottleneckWithFixedBatchNorm'}),
            ('SOLVER$GAMMA', 'Gamma'),
            ('use_apex_ddp', ('ADDP', None)),
            ('exclude_convert_gn', ('NoSyncGN', None)),
            ('ignore_filter_img', ('NoFilter', None)),
            ('pretrained', ('Pretrained', None)),
            ('use_ddp', (None, 'NoneDDP')),
            ('data_partition', 'P', lambda x: x['data_partition'] == 1),
            ('MODEL$ROI_BOX_HEAD$MLP_HEAD_DIM', 'Head', lambda x: dict_get_path_value(x, 'MODEL$ROI_BOX_HEAD$MLP_HEAD_DIM') == 1024),
            ('MODEL$RPN$MATCHER_TYPE', 'RpnM', lambda x: dict_get_path_value(x, 'MODEL$RPN$MATCHER_TYPE') == 'default'),
            ('MODEL$ROI_HEADS$MATCHER_TYPE', 'RoiM', lambda x: dict_get_path_value(x, 'MODEL$ROI_HEADS$MATCHER_TYPE') == 'default'),
            ('MODEL$ROI_HEADS$BG_IOU_THRESHOLD', 'RoiBG', lambda x: dict_get_path_value(x, 'MODEL$ROI_HEADS$BG_IOU_THRESHOLD') == 0.5),
            ('yolo_train_session_param$use_maskrcnn_trainer', ('MT', None)),
            ('rt_param$valid_norm_xywhpos', ('VNXYWHPos', None)),
            ('rt_param$opt_anchor', ('OptA', None)),
            ('opt_anchor_lr_mult', 'AnchorLR', lambda x:
                    x['opt_anchor_lr_mult'] == 1),
            ('MODEL$RPN$NMS_POLICY$THRESH', 'RpnNMS', 0.7),
            ('MODEL$RPN$ASPECT_RATIOS', 'RpnAR'),
            ('MODEL$RPN$PRE_NMS_TOP_N_TRAIN', 'RpnPreNms', 2000),
            ('MODEL$RPN$POST_NMS_TOP_N_TRAIN', 'RpnPostNms', 2000),
            ('MODEL$RPN$FPN_POST_NMS_TOP_N_TRAIN', 'FPNPostNms', 2000),
            ]
    direct_add_value_keys.extend(config.get('direct_add_value_keys', []))

    non_expid_impact_keys = config['non_expid_impact_keys']

    def encode_nms_policy(param):
        t = dict_get_path_value(param, 'MODEL$RPN$NMS_POLICY$TYPE')
        if t != 'nms':
            return 'RpnNMS{}{}'.format(t,
                    hash_sha1(param['MODEL']['RPN']['NMS_POLICY'])[-5:])

    if param['pipeline_type'] == 'MaskRCNNPipeline':
        non_expid_impact_keys.extend(['DATASETS', ''])

    groups = config.get('group', [])

    grouped_keys = set()
    grouped_hints = []
    for g in groups:
        met = all([dict_has_path(param, k) and dict_get_path_value(param, k) == v
               for k, v in g['condition'].items()])
        if met:
            for k in g['condition']:
                grouped_keys.add(k)
            grouped_hints.append(g['value'])

    # we need to update expid so that the model folder contains the critical
    # param information
    infos = []
    need_hash_sha_params = config['need_hash_sha_params']
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
        if isinstance(setting, tuple):
            # will be deprecated. use dictionary for more general setting
            k, v = setting[:2]
            if dict_has_path(param, k):
                pk = dict_get_path_value(param, k)
                if len(setting) == 3:
                    if isinstance(setting[2], dict):
                        if setting[2]['default_value'] == pk:
                            continue
                    elif hasattr(setting[2], '__call__') and setting[2](param):
                        continue
                    elif setting[2] == pk:
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
        elif isinstance(setting, dict):
            k = setting['key']
            if k in grouped_keys:
                continue
            if dict_has_path(param, k):
                pk = dict_get_path_value(param, k)
                default_value = setting['default']
                if isinstance(default_value, dict) and \
                        'from' in default_value and \
                        'import' in default_value:
                    default_value['param'] = {'param': param}
                    from qd.qd_common import execute_func
                    default_value = execute_func(default_value)
                if pk == default_value:
                    continue
                action_type = setting['action_type']
                if action_type == 'keyword':
                    if isinstance(pk, tuple) or isinstance(pk, list):
                        infos.append('{}{}'.format(setting['non_default_hint'],
                            '.'.join(map(str, pk))))
                    else:
                        infos.append('{}{}'.format(setting['non_default_hint'], pk))
                elif action_type == 'bool':
                    infos.append(setting['non_default_hint'])
                elif action_type == 'dict_key':
                    infos.append('{}{}'.format(
                        setting['non_default_hint'],
                        '.'.join(pk)
                    ))
                elif action_type == 'hash':
                    #- key: basemodel
                      #default: null
                      #action_type: hash
                      #non_default_hint: base
                      #readable_hint:
                          #- condition:
                              #basemodel: models/captioning/bert-base-uncased-clean-rand
                            #default: BertRand
                    readable_hint = ''
                    if 'readable_hint' in setting:
                        for kv in setting['readable_hint']:
                            met = all([dict_has_path(param, _k) and dict_get_path_value(param, _k) == _v
                                       for _k, _v in kv['condition'].items()])
                            if met:
                                assert readable_hint == ''
                                readable_hint = kv['value']
                    infos.append('{}{}{}'.format(
                        setting['non_default_hint'],
                        readable_hint,
                        hash_sha1(pk)[-5:],
                    ))
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError

    all_group_key_encode = [
            {'keys': [
                'MODEL$RPN$NMS_POLICY$TYPE',
                'MODEL$RPN$NMS_POLICY$ALPHA',
                'MODEL$RPN$NMS_POLICY$GAMMA',
                'MODEL$RPN$NMS_POLICY$NUM',
                'MODEL$RPN$NMS_POLICY$NUM2',
                'MODEL$RPN$NMS_POLICY$COMPOSE_FINAL_RERANK'],
             'encode': encode_nms_policy}
            ]

    for group_key_encode in all_group_key_encode:
        keys, encode = group_key_encode['keys'], group_key_encode['encode']
        if any(dict_has_path(param, k) for k in keys):
            en = encode(param)
            if en is not None:
                infos.append(en)

    known_keys = []
    known_keys.extend((k for k in need_hash_sha_params))
    known_keys.extend((k for k in non_expid_impact_keys))
    known_keys.extend((s[0] for s in direct_add_value_keys if isinstance(s,
        tuple)))
    known_keys.extend((s['key'] for s in direct_add_value_keys if isinstance(s,
        dict)))
    known_keys.extend(k for x in all_group_key_encode for k in x['keys'])
    known_keys.extend(grouped_keys)

    all_path = get_all_path(param)

    invalid_keys = [k for k in all_path
        if all(k != n and not k.startswith(n + '$') for n in known_keys)]

    assert len(invalid_keys) == 0, pformat(invalid_keys)

    if 'expid_prefix' in param:
        infos.insert(0, param['expid_prefix'])
    infos.extend(grouped_hints)
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
    elif pipeline_type == 'Detectron2Pipeline':
        from qd.pipelines.detectron2 import Detectron2Pipeline
        return Detectron2Pipeline(**kwargs)
    elif pipeline_type == 'fb_moco':
        from qd.pipelines.fb_moco import MocoPipeline
        return MocoPipeline(**kwargs)
    elif pipeline_type == 'sim_clr':
        from qd.pipelines.sim_clr import SimCLRPipeline
        return SimCLRPipeline(**kwargs)
    elif isinstance(pipeline_type, dict):
        from qd.qd_common import execute_func
        info = copy.deepcopy(pipeline_type)
        assert 'param' not in info
        info['param'] = kwargs
        return execute_func(info)
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
    from qd.qd_common import print_frame_info
    print_frame_info()
    init_logging()
    curr_param = copy.deepcopy(param)
    if len(all_test_data) > 0:
        dict_update_nested_dict(curr_param, all_test_data[0])
    pip = create_pipeline(curr_param)
    pip.ensure_train()
    full_expid = pip.full_expid
    param['full_expid'] = full_expid
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(param)
        dict_ensure_path_key_converted(test_data_info)
        dict_update_nested_dict(curr_param, test_data_info)
        pip = load_pipeline(**curr_param)
        pip.ensure_predict()
        pip.ensure_evaluate()

    if param.get('monitor_after'):
        for test_data_info in all_test_data:
            curr_param = copy.deepcopy(param)
            dict_ensure_path_key_converted(test_data_info)
            dict_update_nested_dict(curr_param, test_data_info)
            pip = load_pipeline(**curr_param)
            pip.monitor_train()
    return full_expid

def pipeline_monitor_train(all_test_data, **kwargs):
    for test_data_info in all_test_data:
        dict_ensure_path_key_converted(test_data_info)
        pip = load_pipeline(**test_data_info)
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

def pipeline_pred_eval(all_test_data, **kwargs):
    for test_data_info in all_test_data:
        dict_ensure_path_key_converted(test_data_info)
        pip = load_pipeline(**test_data_info)
        # we should check here instead of before for-loop since we can alter
        # the value of max_iter to just evaluate the intermediate model or take
        # the intermediate model as the final model
        #if not pip.is_train_finished() and pip.model_file is None:
        if not pip.is_train_finished():
            logging.info('the model specified by the following is not ready\n{}'.format(
                pformat(test_data_info)))
            return
        pip.ensure_predict()
        pip.ensure_evaluate()

def calc_randomness(feature_fname):
    from qd.data_layer.loader import create_feature_loader
    loader = create_feature_loader(feature_fname)
    total, num = 0, 0
    cov = 0
    logging.info('calc randomness')
    for i, feature in tqdm(enumerate(loader)):
        feature = feature['feature'].squeeze(1)
        feature = torch.nn.functional.normalize(feature)
        cov += torch.matmul(feature.T, feature)
        total += feature.sum()
        num += feature.numel()
    values = torch.eig(cov)[0][:, 0]
    mean_value = values.mean()
    ratio = (values - mean_value).abs().mean() / mean_value
    result = {}
    result['feat_eig_value_ratio'] = float(ratio)
    result['feat_mean_value'] = float(total / num)
    result['feat_eig_max_value_ratio'] = float(
        (values.max()-mean_value)/mean_value)
    logging.info(pformat(result))
    out_file = feature_fname + '.randomness.report'
    write_to_yaml_file(result, out_file)

@deprecated('use pipeline_pred_eval for simplicity')
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

def env_run(env, func, func_param):
    submit_param = copy.deepcopy(env)
    run_type = submit_param.pop('run_type')
    if run_type in ['debug', 'local']:
        return func(**func_param)
    elif run_type == 'aml':
        result = aml_func_run(func, func_param, **submit_param)
        aml_client = create_aml_client(**submit_param)
        aml_client.inject(result)
        return result
    elif run_type == 'remote':
        from qd.batch_process import remote_run_func
        return remote_run_func(func,
                               is_mpi=submit_param.get('is_mpi', True),
                               availability_check=submit_param.get('availability_check', False),
                               **func_param,
                               )
    elif run_type == 'scheduler':
        from qd.gpucluster.job_scheduler import JobScheduler
        scheduler = JobScheduler()
        scheduler_id = scheduler.submit_to_scheduler(
            env=env,
            execute_func={
                'from': func.__module__,
                'import': func.__name__,
                'param': func_param
            }
        )
        if not env.get('ignore_cluster_submit'):
            cluster_job_id = scheduler.submit_to_cluster(scheduler_id)
            aml_client = create_aml_client(**submit_param)
            aml_client.inject(cluster_job_id)
        return scheduler_id
    else:
        raise NotImplementedError

@deprecated('gradually use env_run')
def platform_run(env, func, **kwargs):
    run_type, num_gpu = env['run_type'], env['num_gpu']
    param = kwargs.get('param', {})
    all_test_data = kwargs.get('all_test_data', [])
    if run_type in ['debug', 'local']:
        return func(**kwargs)
    elif run_type.startswith('philly'):
        env = copy.deepcopy(env)
        del env['run_type']
        del env['num_gpu']
        extra_philly_param = env
        dry_run = run_type == 'philly_dry'
        result = philly_func_run(func,
                {'all_test_data': all_test_data,
                 'param': param},
                dry_run=dry_run,
                num_gpu=num_gpu,
                multi_process=True,
                isDebug=False,
                **extra_philly_param
                )
        return result
    elif run_type == 'aml':
        submit_param = copy.deepcopy(env)
        submit_param.pop('run_type')
        result = aml_func_run(func, kwargs, **submit_param)
        aml_client = create_aml_client(**submit_param)
        aml_client.inject(result)
        return result
    elif run_type == 'remote':
        submit_param = copy.deepcopy(env)
        submit_param.pop('run_type')
        from qd.batch_process import remote_run_func
        return remote_run_func(func,
                is_mpi=submit_param.get('is_mpi', True),
                availability_check=submit_param.get('availability_check', False),
                all_test_data=all_test_data,
                param=param,
                )
    elif run_type == 'print':
        config = {'all_test_data': all_test_data,
                'param': param,
                'type': func.__name__}
        code_file_name = func.__code__.co_filename
        extra_param = convert_to_command_line(config, script=op.relpath(code_file_name))
        logging.info(pformat(config))
        logging.info(extra_param)
    elif run_type == 'save_config':
        config = {'all_test_data': all_test_data,
                'param': param,
                'type': func.__name__}
        if 'full_expid' in param:
            full_expid = param['full_expid']
        else:
            full_expid = '{}_{}_{}'.format(param['data'], param['net'], param['expid'])
        out_file = './aux_data/qd_pipeline_config/{}_{}.yaml'.format(
                full_expid,
                hash_sha1(config)[:5],
            )
        logging.info(out_file)
        assert not op.isfile(out_file)
        logging.info(pformat(config))
        write_to_yaml_file(config, out_file)
    else:
        env_run(env, func, kwargs)

def run_training_pipeline(swap_params):
    from qd.pipelines.auto_param import AutoParam
    auto = AutoParam()
    all_param_env = []

    for param in iter_swap_param(swap_params):
        param = copy.deepcopy(param)
        auto.update_pipeline_param(param, param['env'])
        all_param_env.append((param, param['env']))

        if param['env']['run_type'] == 'scheduler' and \
                not param['env'].get('skip_data_upload'):
            ensure_aml_job_data_available(param, **param['env'])

    logging.info(pformat(all_param_env))
    result = []

    for param, env in all_param_env:
        if 'all_test_data' in param:
            all_test_data = param['all_test_data']
            del param['all_test_data']
        elif 'test_data' in param:
            all_test_data = [{'test_data': param['test_data'],
                'test_split': param.get('test_split', 'test')}]
        else:
            all_test_data = get_all_test_data(param['data'])

        logging.info(pformat(all_test_data))
        r = pipeline_train_eval_platform(param, all_test_data, env)
        result.append(r)

    logging.info(pformat(result))
    if len(result) > 0:
        from qd.db import try_query_job_acc
        try_query_job_acc(result)
    return result

def pipeline_train_eval_platform(param, all_test_data, env, **kwargs):
    import random
    import time
    random.seed(time.time())

    if param['pipeline_type'] == 'MaskRCNNPipeline':
        populate_dataset_hw(param['data'], ['train'])
        for test_data in all_test_data:
            populate_dataset_hw(test_data['test_data'],
                    [test_data['test_split']])

    from qd.prep_dataset.build_tax_data import ensure_build_taxonomy
    ensure_build_taxonomy(param['data'])
    for t in all_test_data:
        ensure_build_taxonomy(t['test_data'])

    if env['run_type'] == 'scheduler':
        result = env_run(
            env,
            pipeline_train_eval_multi,
            func_param={'param': param, 'all_test_data': all_test_data},
        )
    else:
        result = platform_run(env, pipeline_train_eval_multi, param=param,
                all_test_data=all_test_data)
    return result

def model_predict_pipeline(all_test_data,
        env, func=pipeline_pred_eval):
    if env['run_type'] == 'scheduler':
        return env_run(env, func, {'all_test_data': all_test_data})
    else:
        return platform_run(env, func, all_test_data=all_test_data)

def run_prediction_pipeline(swap_params, time_cost_test):
    all_test_data = list(iter_swap_param(swap_params))
    return run_prediction_pipeline_for_list(all_test_data, time_cost_test)

def run_prediction_pipeline_for_list(all_test_data, time_cost_test):
    if any(t.get('monitor_train') for t in all_test_data):
        assert all(t.get('monitor_train') for t in all_test_data)
        monitor_train = True
        for t in all_test_data:
            del t['monitor_train']
    else:
        monitor_train = False
    if time_cost_test:
        for test_data in all_test_data:
            test_data['env']['num_gpu'] = 1
            test_data['env']['run_type'] = 'local'

    from qd.pipelines.auto_param import AutoParam
    auto = AutoParam()
    logging.info('# test = {}'.format(len(all_test_data)))

    for test_data in all_test_data:
        auto.update_pipeline_param(test_data, test_data['env'])

    logging.info(pformat(all_test_data))
    #return

    func = pipeline_pred_eval
    if monitor_train:
        func = pipeline_monitor_train
    if any(t['env']['run_type'] != 'remote' for t in all_test_data):
        result = []
        for test_data in all_test_data:
            r = model_predict_pipeline(
                    all_test_data=[test_data],
                    env=test_data['env'],
                    func=func,
                    )
            result.append(r)
        logging.info(pformat(result))
        return result
    else:
        all_task = []
        for test_data in all_test_data:
            all_task.append({'all_test_data': [test_data]})
            if dict_has_path(test_data, 'env$availability_check'):
                avail_check = dict_get_path_value(test_data,
                        'env$availability_check')
            else:
                avail_check = True
        from qd.batch_process import get_resources, mpi_task_processor, task_processor
        all_resource = get_resources()
        if True:
            l_task_processor = lambda resource, task: mpi_task_processor(resource, task,
                    func)
        else:
            l_task_processor = lambda resource, task: task_processor(resource, task,
                    func)
        logging.info(pformat(all_task))
        b = BatchProcess(all_resource, all_task, l_task_processor)
        b._availability_check = avail_check
        b.run()

def is_pipeline_trained(full_expid):
    try:
        pip = load_pipeline(full_expid=full_expid)
        if pip.is_train_finished():
            return True
        else:
            return False
    except Exception as ex:
        logging.info(ex)
        from qd.qd_common import print_trace
        print_trace()
        return False

def get_last_model(full_expid):
    pip = load_pipeline(full_expid=full_expid)
    return pip.get_checkpoint_file()

def update_current_cluster_status(clusters):
    aml_clients = [create_aml_client(cluster=c) for c in clusters]
    summary = list(map(lambda c: c.get_cluster_status(), aml_clients))
    c = create_annotation_db()
    for cluster, s in zip(clusters, summary):
        s['last_update_time'] = datetime.datetime.now()
        c.update_one('current_cluster', {'cluster': cluster},
                     update={'$set': s}, upsert=True)

def insert_cluster_status(clusters):
    aml_clients = [create_aml_client(cluster=c) for c in clusters]
    summary = list(map(lambda c: c.get_cluster_status(), aml_clients))
    c = create_annotation_db()
    for cluster, s in zip(clusters, summary):
        s['cluster'] = cluster
        c.insert_one('cluster_status_summary', **s)
    if len(summary) > 0:
        all_s = copy.deepcopy(summary[0])
        for s in summary[1:]:
            for k, v in s.items():
                if type(v) in [int, float]:
                    all_s[k] += v
        all_s['cluster'] = '_'.join(clusters)
        c.insert_one('cluster_status_summary', **all_s)

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

def evaluate_topk_pipeline(data, split, pred):
    from qd.qd_pytorch import evaluate_topk
    dataset = TSVDataset(data)
    iter_gt = dataset.iter_data(split, 'label')
    from qd.tsv_io import tsv_reader
    iter_pred = tsv_reader(pred)
    acc = evaluate_topk(iter_pred, iter_gt)
    evaluate_file = pred + '.top1.report'
    logging.info('top1 = {}'.format(acc))
    write_to_yaml_file({'top1': acc}, evaluate_file)

from qd.qd_common import run_if_not_memory_cached

def inject_aml_job_status(clusters, db_query=None):
    client_args = {'with_log': False,
                   }
    db = create_annotation_db()
    def iter_scheduler():
        for info in db.iter_general('scheduler', JobStatus={'$in': [
                'Submitted']}):
            c = info['env']['cluster']
            client = run_if_not_memory_cached(create_aml_client,
                                              cluster=c, **client_args)
            yield client, client.query_one(
                info['appID'],
                with_details=True,
                detect_error_if_failed=False
            )

    collection_name = 'phillyjob'
    def trim_info_in_aml_(job_in_aml):
        non_value_keys = [k for k, v in job_in_aml.items() if v is None]
        for k in non_value_keys:
            del job_in_aml[k]
        # logFiles are a dictionary. The key is the file name. However, mongodb
        # does not allow a key to be with a dot
        fs = job_in_aml.get('logFiles', {})
        job_in_aml['logFiles'] = [{'name': k, 'url': v} for k, v in fs.items()]

    def inject_one_job_status(job_in_aml, db):
        job_in_db = None
        try:
            job_in_db = next(db.iter_general(collection_name, appID=job_in_aml['appID']))
        except:
            pass

        if job_in_db is None:
            try:
                job_in_aml['need_attention'] = job_in_aml['status'] in [
                    AMLClient.status_completed,
                    AMLClient.status_failed]
                job_in_aml = client.query_one(
                    job_in_aml['appID'], with_details=True,
                    with_log=client_args['with_log'],
                    log_full=job_in_aml['need_attention'], detect_error_if_failed=True)
                trim_info_in_aml_(job_in_aml)
                db.insert_one(collection_name, **job_in_aml)
            except:
                # if two instances are running to inject to db, there might be
                # a chance that a new job is inserted here at the same time.
                # For the db, we make the appID unique, and one of the
                # instances will fail. Thus, we just ignore the error here
                from qd.qd_common import print_trace
                print_trace()
            return

        if job_in_db['status'] == job_in_aml['status']:
            s = job_in_db['status']
            if s in [AMLClient.status_failed,
                     AMLClient.status_canceled,
                     AMLClient.status_completed,
                     ]:
                return

        if job_in_aml['status'] in [
                AMLClient.status_completed,
                AMLClient.status_failed] and (job_in_db['status'] !=
                                job_in_aml['status']):
            job_in_aml['need_attention'] = True

        logging.info(job_in_aml['appID'])

        # re-query the job information
        if job_in_aml['status'] in [AMLClient.status_failed,
                                    AMLClient.status_completed]:
            job_in_aml.update(client.query_one(
                job_in_aml['appID'],
                with_details=True,
                with_log=client_args['with_log'],
                log_full=True,
                detect_error_if_failed=client_args['with_log'],
            ))
        if job_in_aml['status'] in [
                AMLClient.status_running,
                AMLClient.status_queued,
        ]:
            job_in_aml.update(client.query_one(
                job_in_aml['appID'],
                with_details=True,
                with_log=client_args['with_log'],
                log_full=False,
                detect_error_if_failed=client_args['with_log'],
            ))

        trim_info_in_aml_(job_in_aml)
        db.update_many(collection_name,
                query={'appID': job_in_aml['appID']},
                update={'$set': job_in_aml})

    visited_id = set()
    for client, job_in_aml in tqdm(iter_scheduler()):
        visited_id.add(job_in_aml['appID'])
        inject_one_job_status(job_in_aml, db)

    db_query_extra = {} if db_query is None else db_query
    for job_in_db in db.iter_general(collection_name, status={'$nin': [
            AMLClient.status_failed,
            AMLClient.status_completed,
            AMLClient.status_canceled,
    ]}, cluster={'$in': clusters}, **db_query_extra):
        if job_in_db['appID'] in visited_id:
            continue
        logging.info('re-check {}: {}'.format(job_in_db['appID'], job_in_db['status']))
        client = run_if_not_memory_cached(
            create_aml_client, cluster=job_in_db['cluster'], **client_args)
        job_in_aml = client.query_one(job_in_db['appID'])
        inject_one_job_status(job_in_aml, db)

    # we will not download the log or parse the log
    #identify_aml_running_job_issues_from_db(clusters, db_query)

def identify_aml_running_job_issues_from_db(clusters, db_query):
    # we will identify whether the running job has issues, and insert some
    # error code to it. We can re-submit it if needed. This function will not
    # re-submit it and leave the resubmission decision to other functions.
    db = create_annotation_db()
    collection_name = 'phillyjob'
    db_query_extra = {} if db_query is None else db_query
    for job_in_db in db.iter_general(
        collection_name,
        status={'$in': [AMLClient.status_running]},
        cluster={'$in': clusters},
        **db_query_extra,
        # if the update time is more than half an hour, we will not check it as
        update_time={'$gt': datetime.datetime.now() -
                     datetime.timedelta(minutes=60)},
    ):
        from qd.qd_common import read_lines
        started = False
        num_connection_failed = 0
        if 'master_log' not in job_in_db:
            continue
        if not op.isfile(job_in_db['master_log']):
            continue
        for line in read_lines(job_in_db['master_log']):
            if 'aml_server.py' in line:
                started = True
            if 'Connection refused' in line:
                num_connection_failed += 1
        code = list(job_in_db.get('result', '').split(','))
        if started and 'connection_failed' in code:
            code.remove('connection_failed')
        # 3600 is one hour here. last time, one job fails with 3601 times and
        # then it works. Here we set a larger number
        # jianfw_we3v32_1609560745_3843d6b7
        if num_connection_failed > 3700 and not started:
            code += ['connection_failed']
        code = ','.join(code)
        if code != job_in_db.get('result', ''):
            db.update_one(collection_name, query={'_id': job_in_db['_id']},
                          update={'$set': {'result': code}}
                          )
def get_vlp_key_idximg_idxcap_if_needed(swap_params):
    if 'pipeline_type' not in swap_params:
        return []
    pipeline_type = swap_params['pipeline_type']
    result = []
    for p in pipeline_type:
        if p['import'] == 'MMaskPretrainPipeline':
            for d in swap_params.get('data', []):
                for v in swap_params.get('caption_version', [None]):
                    result.append({
                        'data': d,
                        'split': 'train',
                        'type': 'key_idximage_idxcaption',
                        'version': v,
                    })
    return result

def verify_param(swap_params):
    run_type = None
    if 'env' in swap_params:
        assert len(swap_params['env']) == 1
        run_type = swap_params['env'][0]['run_type']
        cluster = swap_params['env'][0].get('cluster')
        candidate_clusters = swap_params['env'][0].get('candidate_clusters')
    assert run_type != 'aml'
    if run_type != 'scheduler':
        logging.info('ignore to check since run_type is not scheduler')
        return
    aml = create_aml_client(cluster=cluster)

    need_to_have_data = []
    for k in [
            'train_feature_version',
            'train_label_version',
            'teacher_train_feature_version',
            'teacher_train_label_version',
    ]:
        if k in swap_params:
            if 'data' in swap_params:
                for data in swap_params['data']:
                    data_type = 'feature' if 'feature' in k else 'label'
                    need_to_have_data.extend([
                        {
                            'data': data,
                            'split': 'train',
                            'type': data_type,
                            'version': version
                        } for version in swap_params[k]
                    ])
    need_to_have_data.extend(
        get_vlp_key_idximg_idxcap_if_needed(swap_params))

    need_to_have_file = []
    for info in need_to_have_data:
        dataset = TSVDataset(info['data'])
        tsv_x = dataset.get_data(info['split'] + 'X', info['type'], version=info['version'])
        if op.isfile(tsv_x):
            fs = load_list_file(tsv_x)
            need_to_have_file.extend(fs)
            need_to_have_file.append(tsv_x)
        else:
            tsv = dataset.get_data(info['split'], info['type'], version=info['version'])
            need_to_have_file.append(tsv)
        need_to_have_file.append(dataset.get_data(
            info['split'], 'caption_linelist'))
        need_to_have_file.append(dataset.get_data(
            info['split'], 'num_caption'))

    from_amls = [create_aml_client(cluster=_c) for _c in candidate_clusters]
    ensure_aml_data_available(need_to_have_file, aml, from_amls)

    not_exists = []
    for f in need_to_have_file:
        if not aml.file_exists(f):
            not_exists.append(f)
    logging.info(pformat(not_exists))
    assert len(not_exists) == 0

def ensure_aml_data_available(file_or_dirs, aml, from_amls):
    def find_in_other_cluster(f):
        src_aml = None
        for a in from_amls:
            if a.dir_exists(f):
                src_aml = a
                break
        if src_aml is None:
            for a in from_amls:
                if a.file_exists(f):
                    src_aml = a
                    break
        return src_aml
    for f in file_or_dirs:
        if aml.file_exists(f):
            continue
        src_aml = find_in_other_cluster(f)
        if src_aml is None:
            assert op.isfile(f) or op.isdir(f)
        if op.isdir(f):
            # if this is already a dir, just upload it
            aml.upload(f, from_cluster=src_aml)
        else:
            # as there could be tons of files, e.g. GoogleSplit64 to have 64
            # files, we upload the folder instead to leverage the parallel
            # uploading of azcopy
            assert f.startswith('./') or not f.startswith('/')
            parts = f.split('/')
            assert len(parts) >= 2
            path = '/'.join(parts[:2])
            aml.upload(path, from_cluster=src_aml)

def get_pipeline_waiting_full_expid(waiting_full_expid, basemodel):
    return {
        'execute_if_true_else_break': {
            'from': 'qd.pipeline',
            'import': 'is_pipeline_trained',
            'param': {
                'full_expid': waiting_full_expid,
            },
        },
        'execute': {
            'from': 'qd.pipeline',
            'import': 'get_last_model',
            'param': {
                'full_expid': waiting_full_expid
            }
        },
        'output': basemodel
    }

def assert_equal(a1, a2):
    # this is used in pipeline
    assert a1 == a2, (a1, a2)

def get_cfg(full_expid, key, default=None):
    pip = load_pipeline(full_expid=full_expid)
    from qd.pipelines.uni_pipeline import UniPipeline
    if isinstance(pip, UniPipeline):
        return pip.cfg.get_dict().get(key, default)
    else:
        return getattr(pip, key, default)

def download_aml_log_from_db():
    c = create_annotation_db()
    iter_job = c.iter_phillyjob(
        status={'$in': ['Running', 'Completed', 'Failed']},
        update_time={'$gt': datetime.datetime.now() - datetime.timedelta(days=7)}
    )
    for job in iter_job:
        if job.get('log_download_status') == 'Completed':
            continue
        download_all = job['status'] in ['Completed', 'Failed']
        from qd.gpucluster.aml_client import download_run_logs
        update_info = {}
        all_log = download_run_logs(job, full=download_all)
        update_info['all_log_path'] = all_log
        if all_log is None:
            all_log = []
        master_logs = [l for l in all_log if
                       l.endswith('70_driver_log_0.txt')
                       or l.endswith('00_stdout.txt')
                       or l.endswith('ps-0_stdout.txt')
                       ]
        if len(master_logs) > 0:
            logging.info('parsing the log for {}'.format(job['appID']))
            update_info['master_log'] = master_logs[0]
            job['master_log'] = master_logs[0]
            if op.isfile(job['master_log']):
                x = cmd_run(['tail', '-c', '2097152',
                    job['master_log']],
                    return_output=True)
            else:
                x =''
            job['latest_log'] = x
            from qd.qd_common import attach_log_parsing_result
            attach_log_parsing_result(job)
            keys = ['eta', 'left', 'log_time', 'speed']
            for k in keys:
                if k in job:
                    update_info[k] = job[k]
        if job['status'] in ['Failed', 'Running']:
            from qd.gpucluster.aml_client import detect_aml_error_message
            message = detect_aml_error_message(job['appID'])
            if message is not None:
                update_info['result'] = ','.join(message)
        if download_all:
            update_info['log_download_status'] = 'Completed'
        c.update_phillyjob(query={'_id': job['_id']}, update=update_info)

def blob_download_all_qdoutput(
    prefix, c=None, out_folder='output',
    latest_only=True,
    too_large_limit_in_gb=None,
    ignore_base_fname_patterns=None,
    from_job_scheduler=None,
):
    # e.g. prefix = jianfw/work/qd_output/TaxVehicleV1_1_with_bb_e2e_faster_rcnn_R_50_FPN_1x_M_BS8_MaxIter20e_LR0.01
    # out_folder = 'output'
    if c is None:
        c = 'vig'
    if isinstance(ignore_base_fname_patterns, str):
        from qd.qd_common import load_list_file
        ignore_base_fname_patterns = load_list_file(ignore_base_fname_patterns)
    c = create_cloud_storage(c)
    from datetime import datetime, timedelta
    creation_time_larger_than = datetime.utcnow() - timedelta(days=14)
    if from_job_scheduler:
        c = create_annotation_db()
        all_full_expid = []
        for info in c.iter_general('scheduler',
                                   **{'create_time': {'$gt': datetime.now() - timedelta(days=14)}}):
            from qd.qd_common import query_values_by_path_suffix
            for f in query_values_by_path_suffix(info, '$full_expid'):
                all_full_expid.append(f)
        root = prefix
    else:
        from qd.cloud_storage import get_root_all_full_expid
        all_blob_name = list(c.list_blob_names(
            prefix,
            creation_time_larger_than=creation_time_larger_than))
        root, all_full_expid = get_root_all_full_expid(prefix, all_blob_name)
    #target = 'TaxBingAltText10_B_CLIP_BS2048_MaxIter80e_LR0.01_WD0.0001_SGD_T0.1_AMP_TextMini12LRand41d8b_timm_vit_small_patch16_224_imPre3a612_Aavg_avg_n_bidirectional_MP0_Em1024_Seq100'
    #all_full_expid = [target]
    for full_expid in all_full_expid:
        logging.info(full_expid)
        src_path = op.join(root, full_expid)
        target_folder = op.join(out_folder, full_expid)
        c.blob_download_qdoutput(
            src_path,
            target_folder,
            latest_only=latest_only,
            creation_time_larger_than=creation_time_larger_than,
            too_large_limit_in_gb=too_large_limit_in_gb,
            ignore_base_fname_patterns=ignore_base_fname_patterns,
        )


if __name__ == '__main__':
    from qd.qd_common import parse_general_args
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

