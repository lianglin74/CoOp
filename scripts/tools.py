'''
If the function name is func, you can call the
function through command line by -p {type:func}
'''
from qd_common import init_logging
from yolotrain import CaffeWrapper
from process_tsv import build_taxonomy_impl
from tsv_io import TSVDataset
from qd_common import write_to_yaml_file, load_from_yaml_file
from pprint import pformat
from process_tsv import populate_dataset_details
from yolotrain import yolo_predict
import argparse
import logging
import os.path as op
import copy
from taxonomy import Taxonomy
from taxonomy import LabelToSynset
from process_tsv import TSVDatasetSource

def extract_full_taxonomy_to_vso_format(full_taxonomy_yaml, hier_tax_yaml,
        property_yaml):
    tax = Taxonomy(load_from_yaml_file(full_taxonomy_yaml))
    write_to_yaml_file(tax.dump(feature_name='name', for_train=True), 
            hier_tax_yaml)
    
    ps = []
    for n in tax.root.traverse('preorder'):
        if n == tax.root:
            continue
        c = {}
        c['name'] = n.name
        for f in n.features:
            if f in ['support', 'name', 'dist', 'sub_group']:
                continue
            c[f] = n.__getattribute__(f)
        if len(c) > 1:
            ps.append(c)
    write_to_yaml_file(ps, property_yaml)

def create_taxonomy_based_on_vso(hier_tax_yaml, property_yaml,
        full_taxonomy_yaml):
    tax = Taxonomy(load_from_yaml_file(hier_tax_yaml))
    property_tax = Taxonomy(load_from_yaml_file(property_yaml))
    from process_tsv import attach_properties
    attach_properties([n for n in property_tax.root.iter_search_nodes() 
                            if n != property_tax.root], tax.root)
    from qd_common import ensure_directory
    ensure_directory(op.dirname(full_taxonomy_yaml))
    write_to_yaml_file(tax.dump(for_train=True), full_taxonomy_yaml)

def check_coverage(root_yaml, golden_data):
    tax = Taxonomy(load_from_yaml_file(root_yaml))

    mapper = LabelToSynset()
    mapper.populate_noffset(tax.root)

    golden_source = TSVDatasetSource(golden_data, tax.root)
    golden_source._ensure_initialized()
    labelmap = golden_source.load_labelmap()
    result = []
    line = 'num labels in golden dataset = {}'.format(len(labelmap))
    result.append(line)

    mapped_labels = [l for l in labelmap if l in
            golden_source._sourcelabel_to_targetlabels]
    line = '#mapped label = {}'.format(len(mapped_labels))
    result.append(line)

    ignored_labels = [l for l in labelmap if l not in
            golden_source._sourcelabel_to_targetlabels]
    line = 'ignored_label = {}'.format(len(ignored_labels))
    label_to_count = {}
    for split in ['train', 'trainval', 'test']:
        for label, count in golden_source.iter_data(split,
                'inverted.label.count', 0):
            if label not in ignored_labels:
                continue
            if label not in label_to_count:
                label_to_count[label] = 0
            label_to_count[label] = label_to_count[label] + int(count)
    result.append(line)

    line = ', '.join(ignored_labels)
    result.append(line)

    ignored_but_has_many = [l for l in ignored_labels if label_to_count.get(l,
        0) > 50]
    line = 'ignored but has many'
    result.append(line)

    line = ', '.join(ignored_but_has_many)
    result.append(line)

    logging.info('\n'.join(result))

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
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
    '''
    examples:
    1. evaluate the model on different test datasets
    python scripts/tools.py -p
    "{'type':'evaluate_tax_fullexpid','full_expid':'Tax1300V11_3_darknet19_448_B_noreorg_extraconv2_tree_Init.best_model1739_maxIter.30e_IndexLossWeight0_TsvBoxSamples50EffectBatchSize128NoColorDistortion_bb_only',
    'gpus': [0]}"
    '''

