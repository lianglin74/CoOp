from qd.qd_common import init_logging
from future.utils import viewitems
from qd.qd_common import hash_sha1
from collections import OrderedDict
import copy
import logging
from pprint import pformat


# you can all all the functions prefixed with test_ without any input
# parameter
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
            infos.append('{}{}'.format(v, param[k]))

    true_false_keys = OrderedDict([('use_treestructure', ('Tree', None))])
    for k in true_false_keys:
        if k in param:
            if param[k] and true_false_keys[k][0]:
                infos.append(true_false_keys[k][0])
            elif not param[k] and true_false_keys[k][1]:
                infos.append(true_false_keys[k][1])

    non_expid_impact_keys = ['data', 'net', 'expid_prefix',
            'test_data', 'test_split', 'test_version',
            'dist_url_tcp_port', 'workers']

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

def load_pipeline(curr_param):
    from qd.qd_pytorch import YoloV2PtPipeline
    return YoloV2PtPipeline(load_parameter=True, **curr_param)

def test_model_pipeline_eval_multi(all_test_data, param):
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

def test_uhrs_verify_db_merge_to_tsv():
    from qd.db import uhrs_verify_db_merge_to_tsv
    uhrs_verify_db_merge_to_tsv()

def test_create_new_image_tsv_if_exif_rotated():
    from qd.process_tsv import create_new_image_tsv_if_exif_rotated
    create_new_image_tsv_if_exif_rotated('voc20', 'train')

if __name__ == '__main__':
    from qd.qd_common import parse_general_args
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
