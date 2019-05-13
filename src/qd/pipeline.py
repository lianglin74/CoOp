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
    if data.startswith('Tax'):
        all_data.append(data + '_with_bb')
        all_data.append(data + '_no_bb')
    all_test_data = get_all_test_data(data)
    all_data.extend([info['test_data'] for info in all_test_data])
    return all_data

def ensure_upload_data_for_philly_jobs(data):
    all_data = get_all_related_data_for_philly_jobs(data)

    c = create_cloud_storage('vig')
    philly_client = create_philly_client()
    data_folder = philly_client.get_data_folder_in_blob()
    for d in all_data:
        dataset = TSVDataset(d)
        need_upload = False
        for split in ['train', 'test']:
            splitx = split + 'X'
            if op.isfile(dataset.get_data(splitx)) and d.endswith('_with_bb'):
                if not c.exists('{}/{}/{}.tsv'.format(
                    data_folder,
                    d,
                    split)):
                    convertcomposite_to_standard(d, split)
                    need_upload = True
        if not c.exists('{}/{}/labelmap.txt'.format(data_folder, d)):
            need_upload = True
        if need_upload:
            if op.isdir(dataset._data_root):
                c.az_upload2(dataset._data_root, data_folder)

def philly_func_run(func, param, **submit_param):
    if 'data' in param:
        ensure_upload_data_for_philly_jobs(param['data'])
    assert func.__module__ != '__main__'
    assert 'type' not in param
    param['type'] = func.__name__
    client = create_philly_client(use_blob_as_input=True,
            **submit_param)
    param['gpus'] = list(range(client.num_gpu))
    if hasattr(func, 'func_code'):
        # py2
        code_file_name = func.func_code.co_filename
    else:
        # py3
        code_file_name = func.__code__.co_filename
    extra_param = convert_to_philly_extra_command(param,
            script=op.relpath(code_file_name))
    logging.info(extra_param)
    logging.info(client.get_config_extra_param(extra_param))
    client.submit_without_sync(extra_param)

def dict_get_all_path(d):
    all_path = []
    for k, v in viewitems(d):
        if not isinstance(v, dict):
            all_path.append(k)
        else:
            all_sub_path = dict_get_all_path(v)
            all_path.extend([k + '$' + p for p in all_sub_path])
    return all_path

def dict_has_path(d, p):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            if ps[0] in cur_dict:
                cur_dict = cur_dict[ps[0]]
                ps = ps[1:]
            else:
                return False
        else:
            return True

def dict_get_path_value(d, p):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            cur_dict = cur_dict[ps[0]]
            ps = ps[1:]
        else:
            return cur_dict

def except_to_update_for_stageiter(param):
    if param.get('net', '').startswith('e2e'):
        stageiter = param.get('stageiter', (0, 0))
        max_iter = param.get('max_iter', 0)
        if len(stageiter) == 2 and \
                stageiter[0] == 6 * max_iter // 9 and \
                stageiter[1] == 8 * max_iter // 9:
            return True
    return False

def update_parameters(param):
    default_param = {
            'max_iter': 10000,
            'effective_batch_size': 64}

    direct_add_value_keys = [
            # first value is the key, the second is the name in the folder; the
            # third is the excpdetion condidtion
            ('train_version', 'V'),
            ('effective_batch_size', 'BS'),
            ('max_iter', 'MaxIter'),
            ('max_epoch', 'MaxEpoch'),
            ('last_fixed_param', 'LastFixed'),
            ('num_extra_convs', 'ExtraConv'),
            ('yolo_train_session_param$data_augmentation', 'Aug'),
            ('momentum', 'Momentum'),
            ('base_lr', 'LR'),
            ('stageiter', 'StageIter', except_to_update_for_stageiter),
            ('INPUT$MIN_SIZE_TRAIN', 'Min'),
            ('INPUT$MAX_SIZE_TRAIN', 'Max'),
            ]

    non_expid_impact_keys = ['data', 'net', 'expid_prefix',
            'test_data', 'test_split', 'test_version',
            'dist_url_tcp_port', 'workers', 'force_train',
            'pipeline_type', 'test_batch_size',
            'yolo_train_session_param$debug_train',
            'evaluate_method', 'debug_train',
            'full_expid',
            'display']

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
            infos.append('{}{}'.format(v, pk))

    true_false_keys = OrderedDict([('use_treestructure', ('Tree', None))])
    for k in true_false_keys:
        if k in param:
            if param[k] and true_false_keys[k][0]:
                infos.append(true_false_keys[k][0])
            elif not param[k] and true_false_keys[k][1]:
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

def load_pipeline(kwargs):
    from qd.qd_pytorch import load_latest_parameters
    kwargs_f = load_latest_parameters(op.join('output',
        kwargs['full_expid']))
    for k in kwargs_f:
        if k not in kwargs:
            # we can overwrite the parameter in the parameter file
            kwargs[k] = kwargs_f[k]
    return create_pipeline(kwargs)

def pipeline_train_eval_multi(all_test_data, param, **kwargs):
    init_logging()
    pip = create_pipeline(param)
    pip.ensure_train()
    param['full_expid'] = pip.full_expid
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(param)
        curr_param.update(test_data_info)
        pip = load_pipeline(curr_param)
        pip.ensure_predict()
        pip.ensure_evaluate()

def pipeline_eval_multi(param, all_test_data, **kwargs):
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(param)
        curr_param.update(test_data_info)
        pip = load_pipeline(curr_param)
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
