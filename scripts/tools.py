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
from yolotrain import yolotrain_main
from process_tsv import convert_pred_to_dataset_label
from qd_common import print_as_html
from qd_common import yolo_new_to_old

def convert_full_gpu_yolo_to_non_full_gpu_yolo(full_expid):
    c = CaffeWrapper(full_expid=full_expid, load_parameter=True)
    best_model = c.best_model()
    best_model_iter = best_model.model_iter
    new_proto = op.join('output', full_expid, 'train.prototxt')
    new_weight = op.join('output', full_expid, 'snapshot',
            'model_iter_{}.caffemodel'.format(best_model_iter))
    old_weight = op.join('output', full_expid, 'snapshot',
            'model_iter_{}_old.caffemodel'.format(best_model_iter))
    yolo_new_to_old(new_proto, new_weight, old_weight)

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
    data_with_bb = data + '_with_bb'

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
    
    all_eval_file = []
    for test_data_info in all_test_data:
        curr_param = copy.deepcopy(kwargs)
        for k in test_data_info:
            curr_param[k] = test_data_info[k]
        eval_file = yolo_predict(full_expid=full_expid, 
                **curr_param)
        all_eval_file.append(eval_file)
    
    report_table = {'mAP': {}, 'person': {}}
    for eval_file, test_data_info in zip(all_eval_file, all_test_data):
        test_data = test_data_info['test_data']
        if op.isfile(eval_file):
            overall_map_info = load_from_yaml_file(eval_file + '.map.json')
            per_class_map_info = load_from_yaml_file(eval_file +
                    '.class_ap.json')
            report_table['mAP'][test_data] = overall_map_info['overall']['0.5']['map']
            report_table['person'][test_data] = per_class_map_info['overall']['0.5']['class_ap'].get('person', -1)
        else:
            report_table['mAP'][test_data] = -1
            report_table['person'][test_data] = -1
    
    html_output = op.join('output', full_expid, 'snapshot', 'maps', 'index.html')
    print_as_html(report_table, html_output)

if __name__ == '__main__':
    init_logging()
    from qd_common import parse_general_args
    kwargs = parse_general_args()
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

