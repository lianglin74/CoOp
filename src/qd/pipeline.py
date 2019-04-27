import os.path as op
import logging
from pprint import pformat
import copy
import base64
import re

from qd.process_tsv import convertcomposite_to_standard
from qd.qd_common import load_from_yaml_file
from qd.cloud_storage import create_cloud_storage
from qd.philly import create_philly_client
from qd.tsv_io import TSVDataset
from qd.qd_common import make_by_pattern_maker
from qd.qd_common import make_by_pattern_result


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

def convert_to_philly_extra_command(param, script='scripts/tools.py'):
    logging.info(pformat(param))
    x = copy.deepcopy(param)
    from qd.qd_common import dump_to_yaml_str
    result = "python {} -bp {}".format(
            script,
            base64.b64encode(dump_to_yaml_str(x)).decode())
    return result

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
