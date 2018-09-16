'''
If the function name is func, you can call the
function through command line by -p {type:func}
'''
from qd_common import init_logging
from yolotrain import CaffeWrapper
from process_tsv import build_taxonomy_impl
from tsv_io import TSVDataset
from qd_common import load_from_yaml_file
from pprint import pformat
from process_tsv import populate_dataset_details
from yolotrain import yolo_predict
import argparse
import logging
import os.path as op
import copy

def evaluate_tax_fullexpid(full_expid, **kwargs):
    '''
    does evaluation for the model in full_expid. Taxonomy based
    '''
    c = CaffeWrapper(full_expid=full_expid, load_parameter=True)
    data = c._data

    # evaluate on Tax1300V11_3_with_bb
    all_test_data = [{'test_data': data+'_with_bb', 'test_split':'test'}]

    # MIT and Instagram for human evaluation
    all_test_data.append({'test_data': 'MIT1K-GUID', 'test_split': 'test'})
    all_test_data.append({'test_data': 'Top100Instagram-GUID', 'test_split': 'test'})

    dataset = TSVDataset(data)
    # golden test set
    taxonomy_folder = op.join(dataset._data_root,
            'taxonomy_folder')
    idx = data.find('_')
    if idx >= 0:
        taxonomy_name = data[:idx]
    else:
        taxonomy_name = data
    for golden_data in ['voc0712', 'coco2017', 'OpenImageV4_448']:
        data_source_info = [{'data': golden_data, 'valid_splits': ['test']}]
        out_golden_data = '{}_{}Test'.format(taxonomy_name, golden_data)
        build_taxonomy_impl(
                taxonomy_folder,
                version=0, # for golden dataset, we only use the 0-th version
                max_image_per_label=10000000000,
                min_image_per_label=0,
                num_test=0, # it will only create train.tsv, which is real test
                data=out_golden_data,
                datas=data_source_info)
        out_with_bb = out_golden_data + '_with_bb'
        populate_dataset_details(out_with_bb)
        all_test_data.append({'test_data': out_with_bb, 'test_split': 'train'})
    
    if 'test_data' in kwargs:
        del kwargs['test_data']
    if 'test_split' in kwargs:
        del kwargs['test_split']
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(kwargs)
        for k in test_data_info:
            curr_param[k] = test_data_info[k]
        yolo_predict(full_expid=full_expid, 
                **curr_param)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Yolo network')
    parser.add_argument('-c', '--config_file', help='config file',
            type=str)
    parser.add_argument('-p', '--param', help='parameter string, yaml format',
            type=str)
    args = parser.parse_args()
    kwargs =  {}
    if args.config_file:
        logging.info('loading parameter from {}'.format(args.config_file))
        configs = load_from_yaml_file(args.config_file)
        for k in configs:
            kwargs[k] = configs[k]
    from qd_common import  load_from_yaml_str
    if args.param:
        configs = load_from_yaml_str(args.param)
        for k in configs:
            if k not in kwargs:
                kwargs[k] = configs[k]
            elif kwargs[k] == configs[k]:
                continue
            else:
                logging.info('overwriting {} to {} for {}'.format(kwargs[k], 
                    configs[k], k))
                kwargs[k] = configs[k]
    return kwargs

if __name__ == '__main__':
    init_logging()
    kwargs = parse_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    locals()[kwargs['type']](**kwargs)
    '''
    examples:
    1. evaluate the model on different test datasets
    python scripts/tools.py -p
    "{'type':'evaluate_tax_fullexpid','full_expid':'Tax1300V11_3_darknet19_448_B_noreorg_extraconv2_tree_Init.best_model1739_maxIter.30e_IndexLossWeight0_TsvBoxSamples50EffectBatchSize128NoColorDistortion_bb_only',
    'gpus': [0]}"
    '''

