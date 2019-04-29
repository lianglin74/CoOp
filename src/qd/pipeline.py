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
    return make_by_pattern_result(exp, pattern_to_test_datas)

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
    data_folder = philly_client.get_data_path_in_blob()
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

def philly_func_run(func, param, dry_run=False, **submit_param):
    if 'data' in param:
        ensure_upload_data_for_philly_jobs(param['data'])
    assert func.__module__ != '__main__'
    assert 'type' not in param
    param['type'] = func.__name__
    client = create_philly_client(use_blob_as_input=True, isDebug=False)
    param['gpus'] = list(range(client.num_gpu))
    extra_param = convert_to_philly_extra_command(param,
            script=op.relpath(func.func_code.co_filename))
    logging.info(extra_param)
    client.submit_without_sync(extra_param, dry_run=dry_run,
            **submit_param)

def update_parameters(param):
    default_param = {
            'max_iter': 10000,
            'effective_batch_size': 64}

    for k, v in viewitems(default_param):
        if k not in param:
            param[k] = v

    if 'expid' in param:
        return
    # we need to update expid so that the model folder contains the critical
    # param information
    infos = []
    need_hash_sha_params = ['basemodel']
    for k in need_hash_sha_params:
        if k in param:
            infos.append('{}{}'.format(k, hash_sha1(param[k])[:5]))

    direct_add_value_keys = OrderedDict([('effective_batch_size', 'BS'),
            ('max_iter', 'MaxIter'),
            ('max_epoch', 'MaxEpoch'),
            ('last_fixed_param', 'LastFixed'),
            ('num_extra_convs', 'ExtraConv')])
    for k, v in viewitems(direct_add_value_keys):
        if k in param:
            pk = param[k]
            if type(pk) is str:
                pk = pk.replace('/', '.')
            infos.append('{}{}'.format(v, pk))

    true_false_keys = OrderedDict([('use_treestructure', ('Tree', None))])
    for k in true_false_keys:
        if k in param:
            if param[k] and true_false_keys[k][0]:
                infos.append(true_false_keys[k][0])
            elif not param[k] and true_false_keys[k][1]:
                infos.append(true_false_keys[k][1])

    non_expid_impact_keys = ['data', 'net', 'expid_prefix',
            'test_data', 'test_split', 'test_version',
            'dist_url_tcp_port', 'workers', 'force_train']

    for k in param:
        assert k in need_hash_sha_params or \
                k in non_expid_impact_keys or \
                k in direct_add_value_keys or \
                k in true_false_keys, k

    if 'expid_prefix' in param:
        infos.insert(0, param['expid_prefix'])
    param['expid'] = '_'.join(infos)

def create_pipeline(kwargs):
    from qd.qd_pytorch import YoloV2PtPipeline
    return YoloV2PtPipeline(**kwargs)


def load_pipeline(curr_param):
    from qd.qd_pytorch import YoloV2PtPipeline
    return YoloV2PtPipeline(load_parameter=True, **curr_param)

def test_model_pipeline_eval_multi(all_test_data, param, **kwargs):
    init_logging()
    update_parameters(param)
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
