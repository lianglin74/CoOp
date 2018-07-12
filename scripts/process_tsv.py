from itertools import izip
from process_image import show_images
import shutil
import glob
import yaml
import matplotlib.pyplot as plt
from tsv_io import tsv_reader, tsv_writer
from pprint import pformat
import os
import os.path as op
import sys
import json
from pprint import pprint
import multiprocessing as mp
import random
import numpy as np
from qd_common import ensure_directory
from qd_common import default_data_path, load_net
import time
from multiprocessing import Queue
from shutil import copytree
from shutil import rmtree
from qd_common import img_from_base64
import cv2
import base64
from taxonomy import load_label_parent, labels2noffsets
from taxonomy import LabelToSynset, synset_to_noffset
from taxonomy import noffset_to_synset
from taxonomy import get_nick_name
from taxonomy import load_all_tax, merge_all_tax
from taxonomy import child_parent_print_tree2
from taxonomy import create_markdown_url
#from taxonomy import populate_noffset
from taxonomy import populate_cum_images
from taxonomy import gen_term_list
from taxonomy import gen_noffset
from taxonomy import populate_url_for_offset
from taxonomy import disambibuity_noffsets
from taxonomy import Taxonomy
from qd_common import read_to_buffer, load_list_file
from qd_common import write_to_yaml_file, load_from_yaml_file
from qd_common import encoded_from_img
from tsv_io import extract_label
from tsv_io import create_inverted_tsv
from tsv_io import create_inverted_list

from process_image import draw_bb, show_image, save_image
from qd_common import write_to_file
from tsv_io import TSVDataset, TSVFile
from qd_common import init_logging
import logging
from qd_common import is_cluster
from tsv_io import tsv_shuffle_reader
import argparse
from tsv_io import get_meta_file
import imghdr
from qd_common import calculate_iou
from qd_common import yolo_old_to_new
from qd_common import generate_lineidx
from qd_common import list_to_dict, list_to_dict_unique
from qd_common import dict_to_list
from qd_common import parse_test_data
from qd_common import worth_create
from tsv_io import load_labels
import unicodedata
from taxonomy import is_noffset
import copy
from tsv_io import create_inverted_list2
from qd_common import calculate_image_ap
from tqdm import tqdm

def get_class_count(data, splits):
    dataset = TSVDataset(data)
    result = {}
    for split in splits:
        result[split] = {row[0]: int(row[1])
                for row in dataset.iter_data(
                    split, 'inverted.label.count', -1)}
    return result

def update_yolo_test_proto(input_test, test_data, map_file, output_test):
    dataset = TSVDataset(test_data)
    if op.isfile(dataset.get_noffsets_file()):
        test_noffsets = dataset.load_noffsets()
    else:
        test_noffsets = labels2noffsets(dataset.load_labelmap())
    test_map_id = []
    net = load_net(input_test)
    for l in net.layer:
        if l.type == 'RegionOutput':
            tree_file = l.region_output_param.tree
            r = load_label_parent(tree_file)
            noffset_idx, noffset_parentidx, noffsets = r
            for noffset in test_noffsets:
                test_map_id.append(noffset_idx[noffset])
            write_to_file('\n'.join(map(str, test_map_id)), map_file)
            l.region_output_param.map = map_file 
            l.region_output_param.thresh = 0.005
    assert len(test_noffsets) == len(test_map_id)
    write_to_file(str(net), output_test)

def get_colored_name(node, colors):
    name = node.name
    if hasattr(node, 'sub_group'):
        sub_group = node.sub_group
    else:
        sub_group = -1
    idx = sub_group + 1
    while idx >= len(colors):
        colors.append('rgb({},{},{})'.format(
            int(random.random() * 255),
            int(random.random() * 255),
            int(random.random() * 255)))
    return "<span style='color:{}'>{}</span>".format(colors[idx],
            name)

def get_vis_url(data, l, split=None):
    if split is None:
        return '/detection/view_image?data={}&label={}&start_id=0'.format(
                data, l)
    else:
        return '/detection/view_image?data={}&split={}&label={}&start_id=0'.format(
                data, split, l)


def get_readable_label(l):
    if is_noffset(l):
        return '{}({})'.format(l, get_nick_name(noffset_to_synset(l)))
    else:
        return l

def get_node_key(node, k, default_value=-1):
    r = default_value
    if hasattr(node, k):
        r = node.__getattribute__(k)
    return r

def get_node_info(data, node, colors, all_data):
    ss = []
    if hasattr(node, 'ap'):
        ss.append('ap({0:.2f})'.format(node.ap))
    if hasattr(node, 'images_with_bb') and \
            hasattr(node, 'images_no_bb'):
        keys = ['with_bb', 'no_bb']
        for k in keys:
            ss.append("{}({},<span><a href='{}' target='_blank'>{}</a></span>,<span><a href='{}' target='_blank'>{}</a></span>)".format(
                k,
                get_node_key(node, 'images_' + k, -1),
                get_vis_url(data + '_' + k, node.name, split='train'),
                get_node_key(node, k + '_train', -1),
                get_vis_url(data + '_' + k, node.name, split='test'),
                get_node_key(node, k + '_test', -1)))
    ignore_keys = ['support', 'dist', 'name', 'url',
            'with_bb_train',
            'with_bb_test',
            'no_bb_train',
            'no_bb_test',
            'ap',
            'images_with_bb',
            'images_no_bb',
            'noffset', 
            'sub_group',
            'cum_images_with_bb', 
            'cum_images_no_bb']
    for key in node.features:
        if any(x == key for x in ignore_keys):
            continue
        if key.endswith('_readable') or \
                key.endswith('_toTrain') or \
                key.endswith('_toTest') or \
                key.endswith('_total'):
            continue
        value = node.__getattribute__(key)
        if key in all_data and value != None:
            total = get_node_key(node, '{}_total'.format(key))
            toTrain = get_node_key(node, '{}_toTrain'.format(key))
            toTest = get_node_key(node, '{}_toTest'.format(key))
            extra_info = '({},{},{})'.format(total, toTrain, toTest)
            labels = value.split(',')
            value = ','.join(["<a href='{}' target='_blank'>{}</a>".format(get_vis_url(key, l),
                get_readable_label(l)) for l in labels])
            value = '{}{}'.format(value, extra_info)
        ss.append('{}[{}]'.format(key, value))
    if len(ss) > 0:
        return "{}: {}".format(
                get_colored_name(node, colors),
                '; '.join(ss))
    else:
        return get_colored_name(node, colors)

def gen_html_tree_view(data, full_expid=None,
        predict_file=None):
    colors=['rgb(0,0,0)', 
                'rgb(255,0,0)', 
                'rgb(0,0,255)']
    dataset = TSVDataset(data)
    all_data = os.listdir('./data')
    file_name = op.join(dataset._data_root, 'root_enriched.yaml')
    logging.info('loading {}'.format(file_name))
    with open(file_name, 'r') as fp:
        config_tax = yaml.load(fp)
    tax = Taxonomy(config_tax)
    if full_expid is not None and \
            predict_file is not None:
        map_file = op.join('output',full_expid, 'snapshot', 
                op.splitext(predict_file)[0] + '.report.class_ap.json')
        if op.isfile(map_file):
            class_ap = json.loads(read_to_buffer(map_file))
            class_ap = class_ap['overall']['0.3']['class_ap']
            for node in tax.root.iter_search_nodes():
                node.add_feature('ap', 
                        class_ap.get(node.name, -1))

    def gen_html_tree_view_rec(root):
        '''
        include itself
        '''
        if len(root.children) == 0:
            s = u"<li data-jstree='{{\"icon\":\"glyphicon glyphicon-leaf\"}}'><span>{}</span></li>".format(
                    get_node_info(data, root, colors, all_data))
            return s
        else:
            result = []
            # we cannot remove span tag here
            result.append("<li data-jstree='{{\"opened\":true}}' ><span>{}</span>".format(
                get_node_info(data, root, colors, all_data)))
            result.append('<ul>')
            all_child = sorted(root.children, key=lambda c: c.sub_group)
            for c in all_child:
                r = gen_html_tree_view_rec(c)
                result.append(r)
            result.append('</ul>')
            result.append('</li>')
            return '\n'.join(result)
    s = gen_html_tree_view_rec(tax.root)
    return s


def gt_predict_images(predicts, gts, test_data, target_images, label, start_id, threshold,
        label_to_idx, image_aps, test_data_split='test'): 
    test_dataset = TSVDataset(test_data)
    test_tsv = TSVFile(test_dataset.get_data(test_data_split))
    for i in xrange(start_id, len(target_images)):
        key = target_images[i]
        logging.info('key = {}, ap = {}'.format(key, image_aps[i][1]))
        idx = label_to_idx[key]
        row = test_tsv.seek(idx)
        origin = img_from_base64(row[2])
        im_gt = np.copy(origin)
        draw_bb(im_gt, [g['rect'] for g in gts[key]],
                [g['class'] for g in gts[key]])
        im_gt_target = np.copy(origin)
        gts_target = [g for g in gts[key] if g['class'] == label]
        draw_bb(im_gt_target, [g['rect'] for g in gts_target],
                [g['class'] for g in gts_target])
        im_pred = np.copy(origin)
        rects = [p for p in predicts[key] if p['conf'] > threshold]
        draw_bb(im_pred, [r['rect'] for r in rects],
                [r['class'] for r in rects], 
                [r['conf'] for r in rects])
        im_pred_target = np.copy(origin)
        rects = [p for p in rects if p['class'] == label]
        draw_bb(im_pred_target, [r['rect'] for r in rects],
                [r['class'] for r in rects], 
                [r['conf'] for r in rects])
        yield key, origin, im_gt_target, im_pred_target, im_gt, im_pred, image_aps[i][1]

def get_confusion_matrix_by_predict_file_label(full_expid, 
        predict_file, label, threshold):
    '''
    get confusion matrix for specific label
    '''
    test_data, test_data_split = parse_test_data(predict_file)

    # load the gt
    logging.info('loading {} - {}'.format(test_data, test_data_split))
    test_dataset = TSVDataset(test_data)
    rows = test_dataset.iter_data(test_data_split, 'label', -1)
    gts = {}
    label_to_idx = {}
    keys_with_label = []
    for i, row in enumerate(rows):
        rects = json.loads(row[1])
        gts[row[0]] = rects
        label_to_idx[row[0]] = i
        if any(r['class'] == label for r in rects):
            keys_with_label.append(row[0])

    predict_full_path = op.join('output', full_expid, 'snapshot', predict_file)
    if not full_expid.startswith('Tax'):
        logging.info('loading {}'.format(predict_file))
        predicts, _ = load_labels(predict_full_path)
    else:
        # create the inverted index
        pred_label_file = '{}.labelmap.tsv'.format(predict_full_path)
        pred_inverted_file = '{}.inverted.tsv'.format(predict_full_path)
        pred_key_file = '{}.key.tsv'.format(predict_full_path)
        if not op.isfile(pred_label_file) or \
                not op.isfile(pred_inverted_file) or \
                not op.isfile(pred_key_file):
            pred_inverted = {}
            pred_keys = []
            rows = tsv_reader(predict_full_path)
            logging.info('loading data and creating index')
            inverted, pred_keys = create_inverted_list2(
                    tsv_reader(predict_full_path))
            logging.info('done loading data and creating index')
            pred_labels = inverted.keys()
            tsv_writer(([l] for l in pred_labels), pred_label_file)
            tsv_writer(((l, ' '.join(map(str, inverted[l]))) for l in
                    pred_labels), pred_inverted_file)
            tsv_writer([[k] for k in pred_keys], pred_key_file)
        # find out data whose prediction has the target label
        all_labels = load_list_file(pred_label_file)
        if label in all_labels:
            row = TSVFile(pred_inverted_file).seek(all_labels.index(label))
            assert row[0] == label
            idx = map(int, row[1].split(' '))
        else:
            idx = []
        # find out the index from the ground truth
        key_to_predidx = {k: i for i, k in enumerate(load_list_file(pred_key_file))}
        idx.extend([key_to_predidx[k] for k in keys_with_label])
        idx = set(idx)
        tsv = TSVFile(predict_full_path)
        predicts = {}
        logging.info('loading')
        for i in idx:
            row = tsv.seek(i)
            assert len(row) == 2
            predicts[row[0]] = json.loads(row[1])
        logging.info('done')
        # load data from the inverted index
    logging.info('done loading {}'.format(predict_file))

    return {'predicts': predicts, 
            'gts': gts, 
            'label_to_idx': label_to_idx}

def get_confusion_matrix_by_predict_file(full_expid, 
        predict_file, threshold):
    '''
    get confusion matrix for all classes
    '''

    test_data, test_data_split = parse_test_data(predict_file)

    # load the gt
    logging.info('loading {} - {}'.format(test_data, test_data_split))
    test_dataset = TSVDataset(test_data)
    rows = test_dataset.iter_data(test_data_split, 'label', -1)
    gts = {}
    label_to_idx = {}
    for i, row in enumerate(rows):
        gts[row[0]] = json.loads(row[1])
        label_to_idx[row[0]] = i

    logging.info('loading {}'.format(predict_file))
    predicts, _ = load_labels(op.join('output', full_expid, 'snapshot', predict_file))
    logging.info('done loading {}'.format(predict_file))

    # calculate the confusion matrix
    confusion_pred_gt = {}
    confusion_gt_pred = {}
    update_confusion_matrix(predicts, gts, threshold, 
            confusion_pred_gt, 
            confusion_gt_pred)

    return {'predicts': predicts, 
            'gts': gts, 
            'confusion_pred_gt': confusion_pred_gt, 
            'confusion_gt_pred': confusion_gt_pred,
            'label_to_idx': label_to_idx}

def inc_one_dic_dic(dic, c1, c2):
    if c1 not in dic:
        dic[c1] = {}
    if c2 not in dic[c1]:
        dic[c1][c2] = 0
    dic[c1][c2] = dic[c1][c2] + 1

def update_confusion_matrix(predicts, gts, threshold, 
            confusion_pred_gt, 
            confusion_gt_pred):
    for key in predicts:
        curr_pred = [p for p in predicts[key] if p['conf'] > threshold]
        if key not in gts:
            continue
        curr_gt = gts[key]
        if len(curr_pred) == 0 and len(curr_gt) > 0:
            for g in curr_gt:
                inc_one_dic_dic(confusion_gt_pred, g['class'], 'None')
            continue
        elif len(curr_pred) > 0 and len(curr_gt) == 0:
            for p in curr_pred:
                inc_one_dic_dic(confusion_pred_gt, p['class'], 'None')
            continue
        elif len(curr_pred) == 0 and len(curr_gt) == 0:
            continue
        ious = np.zeros((len(curr_pred), len(curr_gt)))
        for i, p in enumerate(curr_pred):
            for j, g in enumerate(curr_gt):
                iou = calculate_iou(p['rect'], g['rect'])
                ious[i, j] = iou
        gt_idx = np.argmax(ious, axis=1)
        for i, p in enumerate(curr_pred):
            j = gt_idx[i]
            predict_class = p['class']
            gt_class = curr_gt[j]['class']
            if ious[i, j] > 0.3:
                inc_one_dic_dic(confusion_pred_gt, 
                        predict_class, gt_class)
            else:
                inc_one_dic_dic(confusion_pred_gt, 
                        predict_class, 'None')
        pred_idx = np.argmax(ious, axis=0)
        for j, g in enumerate(curr_gt):
            i = pred_idx[j]
            predict_class = curr_pred[i]['class']
            gt_class = g['class']
            if ious[i, j] > 0.3:
                inc_one_dic_dic(confusion_gt_pred,
                        gt_class, predict_class)
            else:
                inc_one_dic_dic(confusion_gt_pred,
                        gt_class, 'None')


def normalize_to_str(s):
    return unicodedata.normalize('NFKD', s).encode('ascii','ignore')

def normalize_str_in_rects(data, out_data):
    '''
    normalize all the unicode to string
    '''
    dataset = TSVDataset(data)
    dest_dataset = TSVDataset(out_data)
    splits = ['train', 'test', 'trainval']
    for split in splits:
        if not op.isfile(dataset.get_data(split)):
            continue
        def gen_rows():
            for key, label_str, im_str in tsv_reader(dataset.get_data(split)):
                rects = json.loads(label_str)
                for rect in rects:
                    s = rect['class']
                    rect['class'] = normalize_to_str(s)
                    if s != rect['class']:
                        logging.info(u'{}->{}'.format(s, rect['class']))
                label_str = json.dumps(rects)
                yield key, label_str, im_str
        tsv_writer(gen_rows(), dest_dataset.get_data(split))

def tsv_details(row_hw, row_label, num_rows):
    label_count = {}
    sizes = []
    logging.info('tsv details...')
    for r_hw, r_label in tqdm(izip(row_hw, row_label), total=num_rows):
        if r_label[1] == 'd':
            # this is the deleted label
            rects = []
        else:
            rects = json.loads(r_label[1])
        assert r_hw[0] == r_label[0]
        height, width = map(int, r_hw[1].split(' '))
        if type(rects) is list:
            # this is the detection dataset
            # convert it to str. if it is unicode, in yaml, there will be some
            # special tags, which is annoying
            curr_labels = set(normalize_to_str(rect['class']) for rect in rects)
            for rect in rects:
                r = rect['rect']
                if all(x == 0 for x in r):
                    # it is image-level annotation
                    continue
                # this should be a valid bounding box
                cx, cy = (r[0] + r[2]) / 2., (r[1] + r[3]) / 2.
                rw, rh = r[2] - r[0], r[3] - r[1]
                #assert cx >= 0 and cx < width \
                        #and cy >= 0 and cy < height
                if rw < 1 or rh < 1:
                    logging.warn('rw or rh too small: {} - {}'.format(r_label[0], 
                        ','.join(map(str, r))))
        else:
            # this is classification dataset
            assert type(rects) is int
            curr_labels = [rects]
        for c in curr_labels:
            if c in label_count:
                label_count[c] = label_count[c] + 1
            else:
                label_count[c] = 1
        sizes.append((height, width))
    size_counts = [s[0] * s[1] for s in sizes]
    min_size = sizes[np.argmin(size_counts)]
    max_size = sizes[np.argmax(size_counts)]
    min_size = map(float, min_size)
    max_size = map(float, max_size)
    mean_size = (np.mean([s[0] for s in sizes]), 
            np.mean([s[1] for s in sizes]))
    mean_size = map(float, mean_size)
    
    return {'label_count': label_count, 
            'min_image_size': min_size, 
            'max_image_size': max_size, 
            'mean_image_size': mean_size}


def detect_duplicate_key(tsv, duplicate_tsv):
    rows = tsv_reader(tsv)
    key_to_idx = {}
    for i, row in enumerate(rows):
        key = row[0]
        if key in key_to_idx:
            key_to_idx[key].append(i)
        else:
            key_to_idx[key] = [i]
    found_error = False
    def gen_rows():
        for key in key_to_idx:
            idxs = key_to_idx[key]
            if len(idxs) > 1:
                logging.info('duplicate key: {}: {}'.format(key, ', '.join(map(str,
                    idxs))))
                yield key, ','.join(map(str, idxs))
    tsv_writer(gen_rows(), duplicate_tsv)
    return TSVFile(duplicate_tsv).num_rows()
            
def populate_all_dataset_details():
    all_data = os.listdir('data/')
    for data in all_data:
        try:
            populate_dataset_details(data)
        except:
            continue

def update_labelmap(rows, num_rows, labelmap):
    '''
    labelmap is a hashset and can be added
    '''
    logging.info('updating labelmap')
    for row in tqdm(rows, total=num_rows):
        assert len(row) == 2
        try:
            labelmap.update(set([rect['class'] for rect in
                json.loads(row[1])]))
        except:
            labelmap.add(row[1])
    logging.info('done')

def populate_dataset_details(data, check_image_details=False):
    logging.info(data)
    dataset = TSVDataset(data)

    splits = ['trainval', 'train', 'test']

    # populate the height and with
    for split in splits:
        if dataset.has(split) and not dataset.has(split, 'hw') \
                and check_image_details:
            multi_thread = True
            if not multi_thread:
                logging.info('generating hw')
                rows = dataset.iter_data(split, progress=True)
                dataset.write_data(((row[0], ' '.join(map(str,
                    img_from_base64(row[-1]).shape[:2]))) for 
                    row in rows), split, 'hw')
            else:
                from pathos.multiprocessing import ProcessingPool as Pool
                num_worker = 128
                num_tasks = num_worker * 3
                num_images = dataset.num_rows(split)
                num_image_per_worker = (num_images + num_tasks - 1) /num_tasks 
                assert num_image_per_worker > 0
                all_idx = []
                for i in range(num_tasks):
                    curr_idx_start = i * num_image_per_worker
                    if curr_idx_start >= num_images:
                        break
                    curr_idx_end = curr_idx_start + num_image_per_worker
                    curr_idx_end = min(curr_idx_end, num_images)
                    if curr_idx_end > curr_idx_start:
                        all_idx.append(range(curr_idx_start, curr_idx_end))
                logging.info('creating pool')
                m = Pool(num_worker)
                def get_hw(filter_idx):
                    dataset = TSVDataset(data)
                    rows = dataset.iter_data(split, progress=True,
                            filter_idx=filter_idx)
                    return [(row[0], ' '.join(map(str,
                        img_from_base64(row[-1]).shape[:2]))) 
                        for row in rows]
                all_result = m.map(get_hw, all_idx)
                x = []
                for r in all_result:
                    x.extend(r)
                dataset.write_data(x, split, 'hw')

    # for each data tsv, generate the label tsv and the inverted file
    for split in splits:
        full_tsv = dataset.get_data(split)
        label_tsv = dataset.get_data(split, 'label')
        if not op.isfile(label_tsv) and op.isfile(full_tsv):
            extract_label(full_tsv, label_tsv)

    for split in splits:
        tsv_file = dataset.get_data(split)
        out_file = get_meta_file(tsv_file)
        if not op.isfile(out_file) and \
                dataset.has(split, 'hw') and \
                dataset.has(split):
            row_hw = dataset.iter_data(split, 'hw')
            row_label = dataset.iter_data(split, 'label')
            num_rows = dataset.num_rows(split)
            if check_image_details:
                details = tsv_details(row_hw, row_label, num_rows)
                write_to_yaml_file(details, out_file)

    labelmap = []
    # generate the label map if there is no
    if not op.isfile(dataset.get_labelmap_file()) and \
            not op.islink(dataset.get_labelmap_file()):
        logging.info('no labelmap, generating...')
        labelmap = []
        for split in splits:
            label_tsv = dataset.get_data(split, 'label', version=-1)
            if not op.isfile(label_tsv):
                continue
            for row in tsv_reader(label_tsv):
                try:
                    labelmap.extend(set([rect['class'] for rect in
                        json.loads(row[1])]))
                except:
                    labelmap.append(row[1])
        if len(labelmap) == 0:
            logging.warning('there are no labels!')
        labelmap = list(set(labelmap))
        logging.info('find {} labels'.format(len(labelmap)))
        need_update = False
        if op.isfile(dataset.get_labelmap_file()):
            origin_labelmap = dataset.load_labelmap()
            if len(origin_labelmap) == len(labelmap):
                for o in origin_labelmap:
                    if o not in labelmap:
                        need_update = True
                        break
            else:
                need_update = True
        else:
            need_update = True
        if need_update:
            logging.info('updating {}'.format(dataset.get_labelmap_file()))
            write_to_file('\n'.join(labelmap), dataset.get_labelmap_file())


    # generate the rows with duplicate keys
    for split in splits: 
        label_tsv = dataset.get_data(split, 'label')
        duplicate_tsv = dataset.get_data(split, 'key_duplicate')
        if op.isfile(label_tsv) and not op.isfile(duplicate_tsv):
            num_duplicate = detect_duplicate_key(label_tsv, duplicate_tsv)
    
    # generate lineidx if it is not generated
    for split in splits:
        lineidx = dataset.get_lineidx(split)
        full_tsv = dataset.get_data(split)
        if not op.isfile(lineidx) and op.isfile(full_tsv):
            logging.info('no lineidx for {}. generating...'.format(split))
            generate_lineidx(full_tsv, lineidx)

    # for each data tsv, generate the inverted file, and the labelmap
    for split in splits:
        v = 0
        while True:
            if not dataset.has(split, 'label', v):
                break
            if not dataset.has(split, 'labelmap', v):
                curr_labelmap = set()
                update_labelmap(dataset.iter_data(split, 'label', v), 
                        dataset.num_rows(split),
                        curr_labelmap)
                curr_labelmap = sorted(list(curr_labelmap))
                dataset.write_data([[l] for l in curr_labelmap], split, 'labelmap', v)
            else:
                curr_labelmap = None
            if not dataset.has(split, 'inverted.label', v) or \
                    not dataset.has(split, 'inverted.label.with_bb', v) or \
                    not dataset.has(split, 'inverted.label.no_bb', v):
                if curr_labelmap is None:
                    curr_labelmap = []
                    for row in dataset.iter_data(split, 'labelmap', v):
                        assert len(row) == 1
                        curr_labelmap.append(row[0])
                def gen_inverted_rows(inv):
                    for label in inv:
                        assert label in curr_labelmap 
                    for label in curr_labelmap:
                        i = inv[label] if label in inv else []
                        yield label, ' '.join(map(str, i))
                inverted, inverted_with_bb, inverted_no_bb = create_inverted_list(
                        dataset.iter_data(split, 'label', v))
                dataset.write_data(gen_inverted_rows(inverted), 
                        split, 'inverted.label', v)
                dataset.write_data(gen_inverted_rows(inverted_with_bb), 
                        split, 'inverted.label.with_bb', v)
                dataset.write_data(gen_inverted_rows(inverted_no_bb), 
                        split, 'inverted.label.no_bb', v)
            v = v + 1
    # check if the number of rows from the label tsv are equal to the number of rows in image tsv
    for split in splits:
        v = 0
        num_rows = None
        while True:
            if not dataset.has(split, 'label', v):
                break
            if dataset.has(split, 'check_count', v):
                v = v + 1
                continue
            if num_rows is None:
                num_rows = dataset.num_rows(split)
            num_rows_in_label = dataset.num_rows(split, 'label', v)
            # we can remove the assert
            assert num_rows == num_rows_in_label
            if num_rows != num_rows_in_label:
                dataset.write_data([['num_rows', num_rows], ['num_rows_in_label', num_rows_in_label]],
                        split, 'check_count', v)
            else:
                dataset.write_data([], split, 'check_count', v)
            v = v + 1

    if not op.isfile(dataset.get_noffsets_file()):
        logging.info('no noffset file. generating...')
        labelmap = dataset.load_labelmap()
        mapper = LabelToSynset()
        ambigous = []
        ss = [mapper.convert(l) for l in labelmap]
        for l, s in zip(labelmap, ss):
            if type(s) is list and len(s) > 1:
                d = create_info_for_ambigous_noffset(l, [synset_to_noffset(s1)
                    for s1 in s])
                ambigous.append(d)
        if len(ambigous) > 0:
            logging.info('ambigous term which has no exact noffset: {}'.format(
                dataset.name))
            write_to_yaml_file(ambigous, dataset.get_noffsets_file() +
                    '.ambigous.yaml')
        noffsets = []
        for success, s in ss:
            if not success:
                noffsets.append('')
            else:
                noffsets.append(','.join([synset_to_noffset(o) for o in s]))
        write_to_file('\n'.join(noffsets), dataset.get_noffsets_file())

    if not op.isfile(dataset.get_labelmap_of_noffset_file()):
        noffsets = dataset.load_noffsets()
        all_line = []
        for noffset in noffsets:
            if len(noffset) == 0:
                all_line.append('unkown')
            else:
                ss = [noffset_to_synset(n) for n in noffset.split(',')]
                all_line.append(','.join([get_nick_name(s) for s in ss]))
        write_to_file('\n'.join(all_line),
                dataset.get_labelmap_of_noffset_file())

    # generate the label -> count tsv
    for split in splits:
        v = 0
        while True:
            label_idx_file = dataset.get_data(split, 'inverted.label', v)
            label_count_file = dataset.get_data(split, 'inverted.label.count', v)
            if op.isfile(label_idx_file):
                if not op.isfile(label_count_file):
                    label_idx = dataset.load_inverted_label_as_list(split, v)
                    tsv_writer(((l, str(len(i))) for l, i in label_idx), label_count_file)
            else:
                break
            v = v + 1
    
    if op.isfile(dataset.get_data('trainX')):
        # composite dataset. 
        create_index_composite_dataset(dataset)
    
    populate_num_images_composite(dataset)

    add_node_to_ancestors(dataset)

def add_node_to_ancestors(dataset):
    tree_file = op.join(dataset._data_root, 'root.yaml')
    out_file = op.join(dataset._data_root, 'treenode_to_ancestors.tsv') 
    if op.isfile(tree_file) and worth_create(tree_file, out_file):
        tax = Taxonomy(load_from_yaml_file(tree_file))
        tax.update()
        tsv_writer([[name, ','.join(tax.name_to_ancestors[name])] for name in
            tax.name_to_ancestors], out_file)

def populate_num_images_composite(dataset):
    data = dataset.name
    data_with_bb = dataset.name + '_with_bb'
    data_no_bb = dataset.name + '_no_bb'
    datas = [data_with_bb, data_no_bb]
    datasets = [TSVDataset(d) for d in datas]
    suffixes = ['with_bb', 'no_bb']
    splits = ['train', 'test']
    dest_tree_file = op.join(dataset._data_root, 'root_enriched.yaml')
    src_tree_file = op.join(dataset._data_root, 'root.yaml')
    if op.isfile(dest_tree_file) or not op.isfile(src_tree_file):
        return
    tax = Taxonomy(load_from_yaml_file(src_tree_file))
    for split in splits:
        for suffix, d in izip(suffixes, datasets):
            if not d.has(split, 'inverted.label.count'):
                continue
            label_to_count_with_bb = {label: int(count) for label, count in d.iter_data(split,
                    'inverted.label.count')}
            for label in label_to_count_with_bb:
                count = label_to_count_with_bb[label]
                nodes = tax.root.search_nodes(name=label)
                assert len(nodes) == 1, label
                node = nodes[0]
                node.add_feature('{}_{}'.format(suffix, split), count)
    write_to_yaml_file(tax.dump(), dest_tree_file)

def create_index_composite_dataset(dataset):
    fname_numImagesPerSource = op.join(dataset._data_root, 
            'trainX.numImagesPerSource.tsv')
    if op.isfile(fname_numImagesPerSource):
        return
    # how many images are contributed from each data source
    trainX_file = dataset.get_data('trainX')
    source_tsvs = load_list_file(trainX_file)
    num_images_per_datasource = [None] * len(source_tsvs)

    shuffle_file = dataset.get_shuffle_file('train')
    if not op.isfile(shuffle_file):
        return
    rows = tsv_reader(shuffle_file)
    all_idxSource_idxRow = []
    for idx_source, idx_row in rows:
        all_idxSource_idxRow.append((int(idx_source), int(idx_row)))
    # note, the data may be duplicated. 
    all_idxSource_idxRow = list(set(all_idxSource_idxRow))

    idxSource_to_idxRows = list_to_dict(all_idxSource_idxRow, 0)
    for idxSource in idxSource_to_idxRows:
        assert num_images_per_datasource[idxSource] is None
        num_images_per_datasource[idxSource] = len(idxSource_to_idxRows[idxSource])
    tsv_writer([(name, str(num)) for name, num in zip(source_tsvs,
        num_images_per_datasource)], fname_numImagesPerSource)

    # for each data source, how many labels are contributed and how many are
    # not
    source_dataset_names = [op.basename(op.dirname(t)) for t in source_tsvs]
    source_tsv_label_files = load_list_file(dataset.get_data('trainX',
        'origin.label'))
    source_tsv_labels = [TSVFile(t) for t in source_tsv_label_files]
    trainX_label_file = dataset.get_data('trainX', 'label')
    all_dest_label_file = load_list_file(trainX_label_file)
    dest_labels = [TSVFile(f) for f in all_dest_label_file]
    all_idxSource_sourceLabel_destLabel = []
    logging.info('each datasource and each idx row')
    for idx_source, idx_row in tqdm(all_idxSource_idxRow):
        source_rects = json.loads(source_tsv_labels[idx_source].seek(idx_row)[-1])
        dest_rects = json.loads(dest_labels[idx_source].seek(idx_row)[-1])
        for r in dest_rects:
            sr = None
            for s in source_rects:
                if all(x == y for (x, y) in zip(s['rect'], r['rect'])):
                    sr = s
                    break
            # move to the end
            source_rects.remove(sr)
            source_rects.append(sr)
            all_idxSource_sourceLabel_destLabel.append((idx_source,
                sr['class'], r['class']))

    idxSource_to_sourceLabel_destLabels = list_to_dict(
            all_idxSource_sourceLabel_destLabel, 0)
    source_numSourceLabels = [(s, 0) for s in source_tsvs]
    source_includedSourceLabels = [(s, []) for s in source_tsvs]
    for idxSource in idxSource_to_sourceLabel_destLabels:
        sourceLabel_destLabels = idxSource_to_sourceLabel_destLabels[idxSource]
        sourceLabel_to_destLabels = list_to_dict(sourceLabel_destLabels, 0)
        source_numSourceLabels[idxSource] = (source_tsvs[idxSource],
                len(sourceLabel_to_destLabels))
        source_includedSourceLabels[idxSource][1].extend(
                sourceLabel_to_destLabels.keys())
    
    tsv_writer([(n, str(i)) for (n, i) in source_numSourceLabels], 
            op.join(dataset._data_root, 'trainX.numCategoriesPerSource.tsv'))

    # save teh list of included labels
    sourceDataset_to_includedSourceLabels = {}
    for source, sourceLabels in source_includedSourceLabels:
        source_dataset_name = op.basename(op.dirname(source))
        if source_dataset_name not in sourceDataset_to_includedSourceLabels:
            sourceDataset_to_includedSourceLabels[source_dataset_name] = []
        sourceDataset_to_includedSourceLabels[source_dataset_name].extend(sourceLabels)
    for source_dataset_name in sourceDataset_to_includedSourceLabels:
        sourceDataset_to_includedSourceLabels[source_dataset_name] = \
                set(sourceDataset_to_includedSourceLabels[source_dataset_name])

    tsv_writer([(n, ','.join(v)) for (n, v) in
        sourceDataset_to_includedSourceLabels.iteritems()], op.join(dataset._data_root, 
            'trainX.includeCategoriesPerSourceDataset.tsv'))

    tsv_writer([(n, get_nick_name(noffset_to_synset(s)) if is_noffset(s) else
        s) for (n, v) in sourceDataset_to_includedSourceLabels.iteritems() 
          for s in v], op.join(dataset._data_root, 
            'trainX.includeCategoriesPerSourceDatasetReadable.tsv'))
    
    #sourceDataset_to_excludeSourceLabels = {}
    ## find out the excluded label list
    #for source_dataset_name in sourceDataset_to_includedSourceLabels:
        #source_dataset = TSVDataset(source_dataset_name)
        #full_label_names = set(source_dataset.load_labelmap())
        #included_labels = sourceDataset_to_includedSourceLabels[source_dataset_name]
        #for l in included_labels:
            ## if l is not in the full_label_names, it will throw exceptions
            #full_label_names.remove(l)
        #sourceDataset_to_excludeSourceLabels[source_dataset_name] = full_label_names

    #tsv_writer([(n, ','.join(v)) for (n, v) in
        #sourceDataset_to_excludeSourceLabels.iteritems()], op.join(dataset._data_root, 
            #'trainX.excludeCategoriesPerSourceDataset.tsv'))

    #tsv_writer([(n, get_nick_name(noffset_to_synset(s)) if is_noffset(s) else
        #s) for (n, v) in sourceDataset_to_excludeSourceLabels.iteritems() for s in v], 
        #op.join(dataset._data_root, 'trainX.excludeCategoriesPerSourceDatasetReadable.tsv'))

class TSVTransformer(object):
    def __init__(self):
        self._total_rows = 0
        self._row_processor = None

    def ReadProcess(self, source_tsv, row_processor):
        self._row_processor = row_processor
        self._total_rows = 0

        rows = tsv_reader(source_tsv)
        x = [self._over_row_processor(row) for row in rows]

        logging.info('total rows = {}; total_processed = {}'.format(self._total_rows, 
                len(x)))
        
    def Process(self, source_tsv, dst_tsv, row_processor):
        '''
        row_processor: a function whose input should be a list of tsv cols and
                      whose return is also a list of tsv colums (will be saved into dst_tsv)
        '''
        self._row_processor = row_processor
        self._total_rows = 0

        rows = tsv_reader(source_tsv)
        result = (self._over_row_processor(row) for row in rows)
        tsv_writer(result, dst_tsv)

        logging.info('total rows = {}'.format(self._total_rows))

    def _over_row_processor(self, row):
        out = self._row_processor(row)
        self._total_rows = self._total_rows + 1
        if self._total_rows % 500 == 0:
            logging.info('processed = {}'.format(self._total_rows))
        return out
def randomize_tsv_file(tsv_file):
    prefix = os.path.splitext(tsv_file)[0]
    shuffle_file = prefix + '.shuffle'
    if os.path.exists(shuffle_file):
        return shuffle_file
    idx_file = prefix + '.lineidx' 
    with open(idx_file, 'r') as fp:
        num = len([line for line in fp.readlines() if len(line.strip()) > 0])
    np.random.seed(777)
    nums = np.random.permutation(num)
    result = '\n'.join(map(str, nums))
    with open(shuffle_file, 'w') as fp:
        fp.write(result)
    return shuffle_file

def gen_tsv_from_labeling(input_folder, output_folder):
    fs = glob.glob(op.join(input_folder, '*'))
    labels = set()
    def gen_rows():
        for f in fs:
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            if im is None:
                continue
            yaml_file = op.splitext(f)[0] + '.yaml'
            if not op.isfile(yaml_file):
                logging.info('{} not exist'.format(yaml_file))
            with open(yaml_file, 'r') as fp:
                bb_labels = yaml.loads(fp.read())
            for bb_label in bb_labels:
                labels.add(bb_label['class'])
            with open(f, 'r') as fp:
                encoded_im = base64.b64encode(fp.read())
            yield op.basename(f), json.dumps(bb_labels), encoded_im

    tsv_writer(gen_rows(), op.join(output_folder, 'train.tsv'))
    write_to_file('\n'.join(labels), op.join(output_folder, 'labelmap.txt'))

def try_json_parse(s):
    try:
        return json.loads(s)
    except ValueError, e:
        return s

def visualize_predict_no_draw(full_expid, predict_file, label, start_id, threshold):
    test_data, test_data_split = parse_test_data(predict_file)
    pred_full_path = op.join('output', full_expid, 'snapshot', predict_file)
    pred_key_path = '{}.key.tsv'.format(pred_full_path)
    pred_label_path = '{}.labelmap.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_inverted_path = '{}.inverted.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_sorted_cache_path = '{}.key_idxGT_idxPred_ap.th{}.{}.tsv'.format(
            pred_full_path, threshold, label)
    
    test_dataset = TSVDataset(test_data)
    if not op.isfile(pred_sorted_cache_path):
        if not op.isfile(pred_key_path) or \
                not op.isfile(pred_label_path) or \
                not op.isfile(pred_inverted_path):
            logging.info('loading {}'.format(pred_full_path))
            inverted, pred_keys = create_inverted_list2(
                    tsv_reader(pred_full_path), threshold)
            pred_labels = inverted.keys()
            logging.info('writing {}'.format(pred_key_path))
            tsv_writer([[k] for k in pred_keys], pred_key_path)
            tsv_writer(([l] for l in pred_labels), pred_label_path)
            tsv_writer(((l, ' '.join(map(str, inverted[l]))) for l in
                    pred_labels), pred_inverted_path)
        keys_from_pred = []
        labelmap = load_list_file(pred_label_path)
        pred_keys = load_list_file(pred_key_path)
        if label in labelmap:
            inverted_row = TSVFile(pred_inverted_path).seek(labelmap.index(label))
            assert inverted_row[0] == label
            assert len(inverted_row) == 2
            idx_from_pred = map(int, inverted_row[1].split(' '))
            keys_from_pred = [pred_keys[i] for i in idx_from_pred]
        else:
            keys_from_pred = []
        inverted_test_split = test_dataset.load_inverted_label(test_data_split, version=-1,
                label=label)
        if label in inverted_test_split:
            idx_from_gt = inverted_test_split[label]
        else:
            idx_from_gt = []
        rows = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=idx_from_gt, unique=True)
        keys_from_gt = [row[0] for row in rows]
        target_keys = list(set(keys_from_pred + keys_from_gt))
        target_keys = [k for k in target_keys if k in pred_keys]
        target_idx_in_pred = [pred_keys.index(k) for k in target_keys]
        gt_keys = test_dataset.load_keys(test_data_split)
        target_idx_in_gt = [gt_keys.index(k) for k in target_keys]
        rows_in_gt = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=target_idx_in_gt)
        pred_tsv = TSVFile(pred_full_path)
        rows_in_pred = (pred_tsv.seek(i) for i in target_idx_in_pred)
        target_aps = []
        for row_in_gt, row_in_pred in izip(rows_in_gt, rows_in_pred):
            assert row_in_gt[0] == row_in_pred[0]
            assert len(row_in_gt) == 2
            assert len(row_in_pred) == 2
            rects_gt = json.loads(row_in_gt[1])
            rects_pred = json.loads(row_in_pred[1])
            rects_gt = [r for r in rects_gt if r['class'] == label]
            rects_pred = [r for r in rects_pred if r['class'] == label]
            ap = calculate_image_ap([r['rect'] for r in rects_gt], 
                    [r['rect'] for r in rects_pred])
            target_aps.append(ap)
        key_idxGT_idxPred_aps = zip(target_keys, target_idx_in_gt,
                target_idx_in_pred, target_aps)
        key_idxGT_idxPred_aps = sorted(key_idxGT_idxPred_aps, key=lambda x:
                x[-1])
        tsv_writer(key_idxGT_idxPred_aps, pred_sorted_cache_path)
    
    tsv = TSVFile(pred_sorted_cache_path)
    total_num = tsv.num_rows()
    if total_num == 0:
        return
    while start_id < 0:
        start_id = start_id + total_num
    while start_id >= total_num:
        start_id = start_id - total_num
    i = start_id
    tsv_pred = TSVFile(pred_full_path)
    for i in range(start_id, total_num):
        key, idx_gt, idx_pred, ap = tsv.seek(i)
        idx_gt, idx_pred, ap = int(idx_gt), int(idx_pred), float(ap)
        row_gt = next(test_dataset.iter_data(test_data_split,
            filter_idx=[idx_gt]))
        row_pred = tsv_pred.seek(idx_pred)
        assert row_gt[0] == row_pred[0], (row_gt[0], row_pred[0])

        rects_gt = json.loads(row_gt[1])
        rects_pred = json.loads(row_pred[1])
        rects_pred = [r for r in rects_pred if r['conf'] > threshold]
        im_origin = img_from_base64(row_gt[-1])
        yield key, im_origin, rects_gt, rects_pred, ap

def visualize_predict(full_expid, predict_file, label, start_id, threshold):
    test_data, test_data_split = parse_test_data(predict_file)
    pred_full_path = op.join('output', full_expid, 'snapshot', predict_file)
    pred_key_path = '{}.key.tsv'.format(pred_full_path)
    pred_label_path = '{}.labelmap.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_inverted_path = '{}.inverted.th{}.tsv'.format(pred_full_path,
            threshold)
    pred_sorted_cache_path = '{}.key_idxGT_idxPred_ap.th{}.{}.tsv'.format(
            pred_full_path, threshold, label)
    
    test_dataset = TSVDataset(test_data)
    if not op.isfile(pred_sorted_cache_path):
        if not op.isfile(pred_key_path) or \
                not op.isfile(pred_label_path) or \
                not op.isfile(pred_inverted_path):
            logging.info('loading {}'.format(pred_full_path))
            inverted, pred_keys = create_inverted_list2(
                    tsv_reader(pred_full_path), threshold)
            pred_labels = inverted.keys()
            logging.info('writing {}'.format(pred_key_path))
            tsv_writer([[k] for k in pred_keys], pred_key_path)
            tsv_writer(([l] for l in pred_labels), pred_label_path)
            tsv_writer(((l, ' '.join(map(str, inverted[l]))) for l in
                    pred_labels), pred_inverted_path)
        keys_from_pred = []
        labelmap = load_list_file(pred_label_path)
        pred_keys = load_list_file(pred_key_path)
        if label in labelmap:
            inverted_row = TSVFile(pred_inverted_path).seek(labelmap.index(label))
            assert inverted_row[0] == label
            assert len(inverted_row) == 2
            idx_from_pred = map(int, inverted_row[1].split(' '))
            keys_from_pred = [pred_keys[i] for i in idx_from_pred]
        else:
            keys_from_pred = []
        inverted_test_split = test_dataset.load_inverted_label(test_data_split, version=-1,
                label=label)
        if label in inverted_test_split:
            idx_from_gt = inverted_test_split[label]
        else:
            idx_from_gt = []
        rows = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=idx_from_gt, unique=True)
        keys_from_gt = [row[0] for row in rows]
        target_keys = list(set(keys_from_pred + keys_from_gt))
        target_keys = [k for k in target_keys if k in pred_keys]
        target_idx_in_pred = [pred_keys.index(k) for k in target_keys]
        gt_keys = test_dataset.load_keys(test_data_split)
        target_idx_in_gt = [gt_keys.index(k) for k in target_keys]
        rows_in_gt = test_dataset.iter_data(test_data_split, t='label', version=-1,
                filter_idx=target_idx_in_gt)
        pred_tsv = TSVFile(pred_full_path)
        rows_in_pred = (pred_tsv.seek(i) for i in target_idx_in_pred)
        target_aps = []
        for row_in_gt, row_in_pred in izip(rows_in_gt, rows_in_pred):
            assert row_in_gt[0] == row_in_pred[0]
            assert len(row_in_gt) == 2
            assert len(row_in_pred) == 2
            rects_gt = json.loads(row_in_gt[1])
            rects_pred = json.loads(row_in_pred[1])
            rects_gt = [r for r in rects_gt if r['class'] == label]
            rects_pred = [r for r in rects_pred if r['class'] == label]
            ap = calculate_image_ap([r['rect'] for r in rects_gt], 
                    [r['rect'] for r in rects_pred])
            target_aps.append(ap)
        key_idxGT_idxPred_aps = zip(target_keys, target_idx_in_gt,
                target_idx_in_pred, target_aps)
        key_idxGT_idxPred_aps = sorted(key_idxGT_idxPred_aps, key=lambda x:
                x[-1])
        tsv_writer(key_idxGT_idxPred_aps, pred_sorted_cache_path)
    
    tsv = TSVFile(pred_sorted_cache_path)
    total_num = tsv.num_rows()
    if total_num == 0:
        return
    while start_id < 0:
        start_id = start_id + total_num
    while start_id >= total_num:
        start_id = start_id - total_num
    i = start_id
    tsv_pred = TSVFile(pred_full_path)
    for i in range(start_id, total_num):
        key, idx_gt, idx_pred, ap = tsv.seek(i)
        idx_gt, idx_pred, ap = int(idx_gt), int(idx_pred), float(ap)
        row_gt = next(test_dataset.iter_data(test_data_split,
            filter_idx=[idx_gt]))
        row_pred = tsv_pred.seek(idx_pred)
        assert row_gt[0] == row_pred[0], (row_gt[0], row_pred[0])

        rects_gt = json.loads(row_gt[1])
        rects_pred = json.loads(row_pred[1])
        rects_pred = [r for r in rects_pred if r['conf'] > threshold]
        rects_gt_target = [r for r in rects_gt if r['class'] == label]
        rects_pred_target = [r for r in rects_pred if r['class'] == label]
        if len(rects_gt_target) == 0 and len(rects_pred_target) == 0:
            logging.info('skipping to next')
            continue
        im_origin = img_from_base64(row_gt[-1])
        im_gt_target = np.copy(im_origin)
        draw_bb(im_gt_target, [r['rect'] for r in rects_gt_target],
                [r['class'] for r in rects_gt_target])
        im_pred_target = np.copy(im_origin)
        draw_bb(im_pred_target, [r['rect'] for r in rects_pred_target],
                [r['class'] for r in rects_pred_target])
        im_gt = np.copy(im_origin)
        draw_bb(im_gt, [r['rect'] for r in rects_gt],
                [r['class'] for r in rects_gt])
        im_pred = np.copy(im_origin)
        draw_bb(im_pred, [r['rect'] for r in rects_pred],
                [r['class'] for r in rects_pred])
        yield key, im_origin, im_gt_target, im_pred_target, im_gt, im_pred, ap

def visualize_box_no_draw(data, split, version, label, start_id, color_map={}):
    dataset = TSVDataset(data)
    logging.info('loading inverted label')
    if split is None:
        # guess which split should be used. only support non-composite tsv
        candidate_split = ['train', 'trainval', 'test']
        for c in candidate_split:
            if not op.isfile(dataset.get_data(c)):
                continue
            inverted = dataset.load_inverted_label(c, version, label)
            if label not in inverted:
                continue
            n = len(inverted[label])
            if n <= start_id:
                start_id = start_id - n
            else:
                logging.info('split = {}'.format(split))
                split = c
                break
        if not split:
            logging.info('cannot find the valid')
            return
    else:
        if label is not None:
            idx = dataset.load_inverted_label(split, version, label)[label]
        else:
            idx = range(dataset.num_rows(split, t='label', version=version))
    if len(idx) == 0:
        return
    while start_id > len(idx):
        start_id = start_id - len(idx)
    while start_id < 0:
        start_id = start_id + len(idx)
    logging.info('start to read')
    rows_image = dataset.iter_data(split, filter_idx=idx[start_id:])
    rows_label = dataset.iter_data(split, 'label', version=version,
            filter_idx=idx[start_id:])
    for row_image, row_label in izip(rows_image, rows_label):
        key = row_image[0]
        assert key == row_label[0]
        assert len(row_image) == 3
        assert len(row_label) == 2
        label_str = row_label[-1]
        img_str = row_image[-1]
        im = img_from_base64(img_str)
        origin = np.copy(im)
        rects = try_json_parse(label_str)
        new_name = key.replace('/', '_').replace(':', '')
        if type(rects) is list:
            rects = [l for l in rects if 'conf' not in l or l['conf'] > 0.3]
            def get_rect_class(rects):
                all_class = []
                all_rect = []
                for rect in rects:
                    label_class = rect['class']
                    rect = rect['rect']
                    all_class.append(label_class)
                    if not (rect[0] == 0 and rect[1] == 0
                            and rect[2] == 0 and rect[3] == 0):
                        all_rect.append(rect)
                    else:
                        all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
                return all_rect, all_class
            all_rect, all_class = get_rect_class(rects)

            all_rec_label, all_class_label = get_rect_class([l for l in rects if l['class']
                    == label])

            yield (new_name, origin, {'rect': all_rect, 'class': all_class},
                   {'rect': all_rec_label, 'class': all_class_label})
        else:
            yield new_name, origin, [{'class': label_str, 'rect': [0, 0, 0, 0]}]

def visualize_box(data, split, version, label, start_id, color_map={}):
    dataset = TSVDataset(data)
    logging.info('loading inverted label')
    if split is None:
        # guess which split should be used. only support non-composite tsv
        candidate_split = ['train', 'trainval', 'test']
        for c in candidate_split:
            if not op.isfile(dataset.get_data(c)):
                continue
            inverted = dataset.load_inverted_label(c, version, label)
            if label not in inverted:
                continue
            n = len(inverted[label])
            if n <= start_id:
                start_id = start_id - n
            else:
                logging.info('split = {}'.format(split))
                split = c
                break
        if not split:
            logging.info('cannot find the valid')
            return
    else:
        inverted = dataset.load_inverted_label(split, version, label)
    logging.info('inverted label loaded')
    logging.info('keys: {}'.format(inverted.keys()))
    if label not in inverted:
        return
    idx = inverted[label]
    if len(idx) == 0:
        return
    while start_id > len(idx):
        start_id = start_id - len(idx)
    while start_id < 0:
        start_id = start_id + len(idx)
    logging.info('start to read')
    rows_image = dataset.iter_data(split, filter_idx=idx[start_id:])
    rows_label = dataset.iter_data(split, 'label', version=version,
            filter_idx=idx[start_id:])
    for row_image, row_label in izip(rows_image, rows_label):
        key = row_image[0]
        assert key == row_label[0]
        assert len(row_image) == 3
        assert len(row_label) == 2
        label_str = row_label[-1]
        img_str = row_image[-1]
        im = img_from_base64(img_str)
        origin = np.copy(im)
        rects = try_json_parse(label_str)
        new_name = key.replace('/', '_').replace(':', '')
        if type(rects) is list:
            #rects = [l for l in rects if 'conf' not in l or l['conf'] > 0.3]
            def get_rect_class(rects):
                all_class = []
                all_rect = []
                for rect in rects:
                    label_class = rect['class']
                    rect = rect['rect']
                    all_class.append(label_class)
                    if not (rect[0] == 0 and rect[1] == 0 
                            and rect[2] == 0 and rect[3] == 0):
                        all_rect.append(rect)
                    else:
                        all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
                return all_rect, all_class
            all_rect, all_class = get_rect_class(rects)
            draw_bb(im, all_rect, all_class)
            target_rects = [l for l in rects if l['class']
                    == label]
            all_rect, all_class = get_rect_class(target_rects)
            im_label = np.copy(origin)
            draw_bb(im_label, all_rect, all_class)
            yield new_name, origin, im_label, im, pformat(target_rects) 
        else:
            yield new_name, origin, im, im, ''


def visualize_tsv2(data, split, label):
    '''
    by default, pass split as 'train'
    TODO: try to refactor it with visualize_box
    '''
    dataset = TSVDataset(data)
    logging.info('loading inverted label')
    inverted = dataset.load_inverted_label(split)
    logging.info('inverted label loaded')
    logging.info('keys: {}'.format(inverted.keys()))
    assert label in inverted
    idx = inverted[label]
    is_composite = False
    if split == 'train' and not op.isfile(dataset.get_data(split)):
        is_composite = True
        tsvs = [TSVFile(f) for f in dataset.get_train_tsvs()]
        shuffle_tsv_rows = tsv_reader(dataset.get_shuffle_file(split))
        shuffle = []
        for row in shuffle_tsv_rows:
            shuffle.append([int(row[0]), int(row[1])])
    else:
        tsv = TSVFile(dataset.get_data(split))
    num_image = 0
    num_rows = 2
    num_cols = 2
    num_image_one_fig = num_rows * num_cols
    idx.extend([0] * (num_image_one_fig - len(idx) % num_image_one_fig))
    idx = np.asarray(idx)
    idx = idx.reshape((-1, num_image_one_fig))
    logging.info('start to read')
    color_map = {}
    for i in idx:
        all_image = []
        for j in i:
            logging.info(j)
            if is_composite:
                row_image = tsvs[shuffle[j][0]].seek(shuffle[j][1])
            else:
                row_image = tsv.seek(j)
            im = img_from_base64(row_image[-1])
            labels = try_json_parse(row_image[1])
            num_image = num_image + 1
            if type(labels) is list:
                labels = [l for l in labels if 'conf' not in l or l['conf'] > 0.3]
                all_class = []
                all_rect = []
                for label in labels:
                    label_class = label['class']
                    rect = label['rect']
                    all_class.append(label_class)
                    if not (rect[0] == 0 and rect[1] == 0 
                            and rect[2] == 0 and rect[3] == 0):
                        all_rect.append(rect)
                    else:
                        all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
                new_name = row_image[0].replace('/', '_').replace(':', '')
                draw_bb(im, all_rect, all_class, color=color_map)
            all_image.append(im)
        logging.info('start to show')
        show_images(all_image, num_rows, num_cols)

    logging.info('#image: {}'.format(num_image))

def visualize_tsv(tsv_image, tsv_label, out_folder=None, label_idx=1):
    '''
    deprecated, use visualize_tsv2
    '''
    rows_image = tsv_reader(tsv_image)
    rows_label = tsv_reader(tsv_label)
    assert out_folder == None or not op.exists(out_folder)
    source_folder = op.dirname(tsv_image)
    num_image = 0
    for row_image, row_label in izip(rows_image, rows_label):
        assert row_image[0] == row_label[0]
        im = img_from_base64(row_image[-1])
        labels = try_json_parse(row_label[label_idx])
        num_image = num_image + 1
        if type(labels) is list:
            labels = [l for l in labels if 'conf' not in l or l['conf'] > 0.3]
            all_class = []
            all_rect = []
            for label in labels:
                label_class = label['class']
                rect = label['rect']
                all_class.append(label_class)
                if not (rect[0] == 0 and rect[1] == 0 
                        and rect[2] == 0 and rect[3] == 0):
                    all_rect.append(rect)
                else:
                    all_rect.append((0, 0, im.shape[1] - 1, im.shape[0] - 1))
            new_name = row_image[0].replace('/', '_').replace(':', '')
            draw_bb(im, all_rect, all_class)
            if out_folder:
                fname = os.path.join(out_folder, 
                        '_'.join(set(c.replace(' ', '_') for c in all_class)),
                        new_name +'.png')
        else:
            fname = op.join(out_folder, row_image[0],
                    '{}_{}_{}.png'.format(num_image, labels, row_image[0]))
        if out_folder:
            save_image(im, fname)
        else:
            show_image(im)

    logging.info('#image: {}'.format(num_image))

def iou(wh1, wh2):
    w1, h1 = wh1
    w2, h2 = wh2
    return min(w1, w2) * min(h1, h2) / max(w1, w2) / max(h1, h2)

class ImageTypeParser(object):
    def __init__(self):
        self.m = None
        pass

    def parse_type(self, im_binary):
        return imghdr.what('', im_binary)
        self._ensure_init()
        mime_type = self.m.buffer(im_binary)
        t = op.basename(mime_type)
        return t

    def _ensure_init(self):
        if self.m is None:
            import magic
            #m = magic.open(magic.MAGIC_MIME_TYPE)
            m = magic.from_file(magic.MAGIC_MIME_TYPE)
            m.load()
            self.m = m

def collect_label(row, stat, **kwargs):
    labels = json.loads(row[1])
    labels = [label['class'] for label in labels]
    remove_labels = kwargs['remove_image'].split(',')
    is_remove = False
    for label in labels:
        if label in remove_labels or remove_labels == 'all':
            if random.random() <= kwargs['remove_image_prob']:
                is_remove = True
                break

    stat.append((is_remove, labels))

class DatasetSource(object):
    def __init__(self):
        pass

    def populate_info(self, root):
        pass

    def gen_tsv_rows(self, root):
        pass

class TSVDatasetSource(TSVDataset, DatasetSource):
    def __init__(self, name, root=None):
        super(TSVDatasetSource, self).__init__(name)
        self._noffset_count = {}
        self._type = None
        self._root = root
        # the list of <datasetlabel, rootlabel>
        self._sourcelabel_to_targetlabels = None
        self._targetlabel_to_sourcelabels = None
        self._initialized = False
        self._type_to_datasetlabel_to_split_idx = None
        self._type_to_datasetlabel_to_count = None
        self._type_to_split_label_idx = None

    def populate_info(self, root):
        self._ensure_initialized()
        types = ['with_bb', 'no_bb']
        for node in root.iter_search_nodes():
            if root == node:
                continue
            if node.name in self._targetlabel_to_sourcelabels:
                sourcelabels = self._targetlabel_to_sourcelabels[node.name]
                node.add_feature(self.name, ','.join(sourcelabels))
                if any(is_noffset(l) for l in sourcelabels):
                    node.add_feature(self.name + '_readable', 
                            ','.join(get_nick_name(noffset_to_synset(l)) if
                                is_noffset(l) else l for l in sourcelabels))
                total = 0
                for sourcelabel in sourcelabels:
                    for t in self._type_to_datasetlabel_to_count:
                        datasetlabel_to_count = self._type_to_datasetlabel_to_count[t]
                        c = datasetlabel_to_count.get(sourcelabel, 0)
                        total = total + c
                        key = 'images_{}'.format(t)
                        node.add_feature(key,
                                node.__getattribute__(key) + c)
                node.add_feature('{}_total'.format(self.name), total)

    def _ensure_initialized(self):
        if self._initialized:
            return
        populate_dataset_details(self.name)
        splits = ['trainval', 'train', 'test']
        types = ['with_bb', 'no_bb']
        # check the type of the dataset
        #for split in splits:
            #label_tsv = self.get_data(split, 'label', version=-1)
            #if not op.isfile(label_tsv):
                #continue
            #for row in tsv_reader(label_tsv):
                #rects = json.loads(row[1])
                #if any(np.sum(r['rect']) > 1 for r in rects):
                    #self._type = 'with_bb'
                #else:
                    #self._type = 'no_bb'
                #break
        #assert self._type is not None, "{} is bad".format(self.name)
        #logging.info('identify {} as {}'.format(self.name, self._type))
        
        # list of <split, label, idx>
        self._type_split_label_idx = []
        for split in splits:
            logging.info('loading the inverted file: {}-{}'.format(self.name,
                split))
            if not op.isfile(self.get_data(split, 'label', version=-1)):
                continue
            for t in types:
                rows = self.iter_data(split, 'inverted.label.{}'.format(t), -1)
                inverted = {r[0]: (map(int, r[1].split(' ')) if len(r[1]) > 0
                    else []) for r in rows}
                label_idx = dict_to_list(inverted, 0)
                for label, idx in label_idx:
                    self._type_split_label_idx.append((t, split, label, idx))
        self._type_to_split_label_idx = list_to_dict(
                self._type_split_label_idx, 0)
        self._type_to_datasetlabel_to_split_idx = {}
        for t in self._type_to_split_label_idx:
            split_label_idx = self._type_to_split_label_idx[t]
            label_to_split_idx = list_to_dict(split_label_idx, 1)
            self._type_to_datasetlabel_to_split_idx[t] = label_to_split_idx
        self._type_to_datasetlabel_to_count = {}
        for t in self._type_to_datasetlabel_to_split_idx:
            datasetlabel_to_split_idx = self._type_to_datasetlabel_to_split_idx[t]
            datasetlabel_to_count = {l: len(datasetlabel_to_split_idx[l]) for l in
                    datasetlabel_to_split_idx}
            self._type_to_datasetlabel_to_count[t] = datasetlabel_to_count

        self.update_label_mapper()
        self._initialized = True

    def update_label_mapper(self):
        root = self._root
        # load the labelmap for all splits, self.load_labelmap is not correct,
        # since we will update the label and will not update the labelmap
        labelmap = []
        for split in ['train', 'test', 'trainval']:
            if self.has(split, 'labelmap', -1):
                for row in self.iter_data(split, 'labelmap', -1):
                    labelmap.append(row[0])
        labelmap = list(set(labelmap))
        hash_labelmap = set(labelmap)

        tree_noffsets = {}
        for node in root.iter_search_nodes():
            if node == root or not node.noffset:
                continue
            for s in node.noffset.split(','):
                tree_noffsets[s] = node.name
        name_to_targetlabels = {}
        targetlabel_has_whitelist = set()
        invalid_list = []
        for node in root.iter_search_nodes():
            if node == root:
                continue
            if hasattr(node, self.name):
                # this is like a white-list
                values = node.__getattribute__(self.name)
                if values is not None:
                    source_terms = values.split(',')
                    for t in source_terms:
                        t = t.strip()
                        if t not in name_to_targetlabels:
                            name_to_targetlabels[t] = set()
                        if t not in hash_labelmap:
                            invalid_list.append((t, self.name, node.name))
                        name_to_targetlabels[t].add(node.name)
                # even if it is None, we will also add it to white-list so that
                # we will not automatically match the term.
                targetlabel_has_whitelist.add(node.name)
            else:
                # we will keep the lower case always for case-insensitive
                # comparison
                t = node.name.lower()
                if t not in name_to_targetlabels:
                    name_to_targetlabels[t] = set()
                name_to_targetlabels[t].add(node.name)

        sourcelabel_targetlabel = [] 
        assert len(invalid_list) == 0, pformat(invalid_list)

        #result = {}
        label_to_synset = LabelToSynset()
        for l in labelmap:
            matched = False
            if l.lower() in name_to_targetlabels:
                matched = True
                for t in name_to_targetlabels[l.lower()]:
                    sourcelabel_targetlabel.append((l, t))
            if l in name_to_targetlabels:
                for t in name_to_targetlabels[l]:
                    sourcelabel_targetlabel.append((l, t))
                matched = True
            if not matched:
                succeed, ns = label_to_synset.convert(l)
                if not succeed:
                    continue
                for n in ns:
                    n = synset_to_noffset(n)
                    if n in tree_noffsets:
                        t = tree_noffsets[n]
                        if t in targetlabel_has_whitelist:
                            # if it has white list, we will not respect the
                            # noffset to do the autmatic matching
                            continue
                        sourcelabel_targetlabel.append((l, t))
                        #result[l] = t 

        self._sourcelabel_to_targetlabels = list_to_dict_unique(sourcelabel_targetlabel,
                0)
        self._targetlabel_to_sourcelabels = list_to_dict_unique(sourcelabel_targetlabel,
                1)
        
        name_to_node = None
        for sourcelabel in self._sourcelabel_to_targetlabels:
            targetlabels = self._sourcelabel_to_targetlabels[sourcelabel]
            # for all these target labels, if A is the parent of B, remove A.
            if len(targetlabels) > 1:
                if name_to_node is None:
                    name_to_node = {node.name: node for node in root.iter_search_nodes() if
                            node != root}
                targetnodes = [name_to_node[l] for l in targetlabels]
                validnodes = []
                for n in targetnodes:
                    ancestors = node.get_ancestors()[:-1]
                    to_remove = []
                    for v in validnodes:
                        if v in ancestors:
                            to_remove.append(v)
                    for t in to_remove:
                        validnodes.remove(t)
                    can_add = True
                    for v in validnodes:
                        if n in v.get_ancestors()[:-1]:
                            can_add = False
                            break
                    if can_add:
                        validnodes.append(n)
                targetlabels[:] = []
                targetlabels.extend(v.name for v in validnodes)

        return self._sourcelabel_to_targetlabels

    def select_tsv_rows(self, label_type):
        self._ensure_initialized()
        if label_type not in self._type_to_split_label_idx:
            return []
        split_label_idx = self._type_to_split_label_idx[label_type]
        datasetlabel_to_splitidx = list_to_dict(split_label_idx, 1) 
        result = []
        for datasetlabel in datasetlabel_to_splitidx:
            if datasetlabel in self._sourcelabel_to_targetlabels:
                split_idxes = datasetlabel_to_splitidx[datasetlabel]
                targetlabels = self._sourcelabel_to_targetlabels[datasetlabel]
                for rootlabel in targetlabels:
                    result.extend([(rootlabel, split, idx) for split, idx in
                        split_idxes])
        return result

    def gen_tsv_rows(self, root, label_type):
        selected_info = self.select_tsv_rows(root, label_type)
        mapper = self._sourcelabel_to_targetlabels
        for split, idx in selected_info:
            data_tsv = TSVFile(self.get_data(split))
            for i in idx:
                data_row = data_tsv.seek(i)
                rects = json.loads(data_row[1])
                convert_one_label(rects, mapper)
                assert len(rects) > 0
                data_row[1] = json.dumps(rects)
                yield data_row
        return


def initialize_images_count(root):
    for node in root.iter_search_nodes():
        node.add_feature('images_with_bb', 0)
        node.add_feature('images_no_bb', 0)

def trainval_split(dataset, num_test_each_label):
    if op.isfile(dataset.get_train_tsv()):
        logging.info('skip to run trainval split for {} because it has been done'.format(
            dataset.name))
        return
    random.seed(777)
    label_to_idx = {}
    for i, row in enumerate(tsv_reader(dataset.get_trainval_tsv('label'))):
        rects = json.loads(row[1])
        if len(rects) == 0:
            logging.info('{} has empty label for {}'.format(dataset.name, i))
            continue
        random.shuffle(rects)
        label = rects[0]['class']
        if label in label_to_idx:
            label_to_idx[label].append(i)
        else:
            label_to_idx[label] = [i]

    test_idx = []
    train_idx = []
    for label in label_to_idx:
        if len(label_to_idx[label]) < num_test_each_label:
            logging.fatal('dataset {} has less than {} images for label {}'.
                    format(dataset.name, num_test_each_label, label))
        random.shuffle(label_to_idx[label])
        test_idx.extend(label_to_idx[label][: num_test_each_label])
        train_idx.extend(label_to_idx[label][num_test_each_label: ])

    trainval = TSVFile(dataset.get_trainval_tsv())

    def gen_train():
        for i in train_idx:
            row = trainval.seek(i)
            yield row
    tsv_writer(gen_train(), dataset.get_train_tsv())

    def gen_test():
        for i in test_idx:
            yield trainval.seek(i)
    tsv_writer(gen_test(), dataset.get_test_tsv_file())

def convert_one_label(rects, label_mapper):
    to_remove = []
    rects2 = []
    for rect in rects:
        if rect['class'] in label_mapper:
            for t in label_mapper[rect['class']]:
                r2 = copy.deepcopy(rect)
                assert type(t) is str or type(t) is unicode
                r2['class'] = t 
                rects2.append(r2)
    rects[:] = []
    rects.extend(rects2)

def convert_label(label_tsv, idx, label_mapper, with_bb):
    '''
    '''
    tsv = TSVFile(label_tsv)
    result = None
    for i in tqdm(idx):
        row = tsv.seek(i)
        assert len(row) == 2
        rects = json.loads(row[1])
        if with_bb:
            rects = [r for r in rects if any(x != 0 for x in r['rect'])]
        else:
            rects = [r for r in rects if all(x == 0 for x in r['rect'])]
        if result is None:
            # don't use this, because all list will be shared
            #result = [len(row) * ['d']] * tsv.num_rows()
            result = [None] * tsv.num_rows()
            for _ in range(len(result)):
                result[_] = ['d'] * len(row)
        rects2 = []
        # the following code should use convert_one_label
        for rect in rects:
            if rect['class'] in label_mapper:
                for t in label_mapper[rect['class']]:
                    r2 = copy.deepcopy(rect)
                    r2['class'] = t 
                    if rect['class'] != t:
                        # keep this for logging
                        r2['class_from'] = rect['class']
                    rects2.append(r2)
        assert len(rects2) > 0
        row[1] = rects2
        result[i] = row
    return result

def create_info_for_ambigous_noffset(name, noffsets):
    definitions = [str(noffset_to_synset(n).definition()) for n in noffsets]
    de = [{'noffset': n, 'definition': d.replace("`", '').replace("'", '')}
            for n, d in zip(noffsets, definitions)]
    d = {'name': name,
            'definitions': de,
            'noffset': None,
            'markdown_url': create_markdown_url(noffsets)}
    return d

def node_should_have_images(root, th, fname):
    enough = True
    few_training = []
    for node in root.iter_search_nodes():
        if node.cum_images_no_bb + node.cum_images_with_bb < th:
            few_training.append({'name': node.name, 
                'cum_images_no_bb': node.cum_images_no_bb,
                'cum_images_with_bb': node.cum_images_with_bb,
                'parent list': [p.name for p in node.get_ancestors()[:-1]]})
            enough = False
            logging.warn('less images: {} ({}, {})'.format(
                node.name.encode('utf-8'),
                node.cum_images_with_bb,
                node.cum_images_no_bb))
    if enough:
        logging.info('con. every node has at least {} images'.format(th))
    else:
        write_to_yaml_file(few_training, fname)

def clean_dataset(source_dataset_name, dest_dataset_name):
    source_dataset = TSVDataset(source_dataset_name)
    dest_dataset = TSVDataset(dest_dataset_name)
    splits = ['train', 'trainval', 'test']
    for split in splits:
        src_tsv = source_dataset.get_data(split)
        if op.isfile(src_tsv):
            dest_tsv = dest_dataset.get_data(split)
            def gen_rows():
                rows = tsv_reader(src_tsv)
                num_removed_images = 0
                num_removed_rects = 0
                for i, row in enumerate(rows):
                    if (i % 1000) == 0:
                        logging.info('{} - #removedImages={};#removedRects={}'.format(
                            i, num_removed_images, num_removed_rects))
                    im = img_from_base64(row[-1])
                    height, width = im.shape[:2]
                    rects = json.loads(row[1])
                    invalid = False
                    to_remove = []
                    for rect in rects:
                        r = rect['rect']
                        if all(s == 0 for s in r):
                            continue
                        cx, cy = (r[0] + r[2]) / 2., (r[1] + r[3]) / 2.
                        w, h = r[2] - r[0], r[3] - r[1]
                        if cx < 0 or cy < 0 or cx >= width or \
                                cy >= height or w <= 1 or h <= 1 or \
                                w >= width or h >= height:
                            to_remove.append(rect)
                    if len(to_remove) > 0:
                        logging.info('before removing {}'.format(len(rects)))
                        num_removed_rects = num_removed_rects + len(to_remove)
                    for rect in to_remove:
                        rects.remove(rect)
                        r = rect['rect']
                        logging.info('removing {}'.format(','.join(map(str,
                                r))))
                    if len(to_remove) > 0:
                        logging.info('after removing {}'.format(len(rects)))
                        if len(rects) > 0:
                            yield row[0], json.dumps(rects), row[2]
                        else:
                            num_removed_images = num_removed_images + 1
                            logging.info('removing image {}'.format(row[0]))
                    else:
                        yield row
            tsv_writer(gen_rows(), dest_tsv)

def parallel_convert_label(func, all_task, num_worker=128):
    num_split = num_worker
    num_task_each_split = (len(all_task) + num_split - 1) / num_split
    all_sub_tasks = []
    for i in range(num_split):
        start_idx = i * num_task_each_split
        if start_idx >= len(all_task):
            break
        end_idx = start_idx + num_task_each_split
        if end_idx > len(all_task):
            end_idx = len(all_task)
        all_sub_tasks.append(all_task[start_idx:end_idx])
    from pathos.multiprocessing import ProcessingPool as Pool
    m = Pool(num_worker)
    all_sub_results = m.map(func, all_sub_tasks)
    result = all_sub_results[0]
    for s in all_sub_results[1:]:
        assert len(result) == len(s)
        for i in range(len(result)):
            assert len(result[i]) == 2
            assert len(s[i]) == 2
            if result[i][1] == 'd' and s[i][1] != 'd':
                result[i][0] = s[i][0]
                result[i][1] = s[i][1]
            else:
                assert not (result[i][1] != 'd' and s[i][1] != 'd')
    return result

def parallel_map_to_array(func, all_task, num_worker=128):
    num_split = num_worker * 2
    num_task_each_split = (len(all_task) + num_split - 1) / num_split
    all_sub_tasks = []
    for i in range(num_split):
        start_idx = i * num_task_each_split
        if start_idx > len(all_task):
            break
        end_idx = start_idx + num_task_each_split
        if end_idx > len(all_task):
            end_idx = len(all_task)
        all_sub_tasks.append(all_task[start_idx:end_idx])
    from pathos.multiprocessing import ProcessingPool as Pool
    m = Pool(num_worker)
    all_sub_results = m.map(func, all_sub_tasks)
    result = []
    for s in all_sub_results:
        result.extend(s)
    return result

def create_trainX(train_ldtsi, extra_dtsi, tax, out_dataset):
    t_to_ldsi = list_to_dict(train_ldtsi, 2)
    extra_t_to_ldsi = list_to_dict(extra_dtsi, 1)
    train_ldtsik = []
    extra_dtsik = []
    shuffle_idx = []
    for label_type in t_to_ldsi:
        ldsi = t_to_ldsi[label_type]
        extra_ldsi = extra_t_to_ldsi.get(label_type, [])
        d_to_lsi = list_to_dict(ldsi, 1)
        extra_d_to_lsi = list_to_dict(extra_ldsi, 1)
        k = 0
        sources = []
        sources_origin_label = []
        sources_label = []
        with_bb = label_type == 'with_bb'
        for dataset in d_to_lsi:
            lsi = d_to_lsi[dataset]
            extra_lsi = extra_d_to_lsi.get(dataset, [])
            s_li = list_to_dict(lsi, 1)
            extra_s_li = list_to_dict(extra_lsi, 1)
            for split in s_li:
                li = s_li[split]
                idx_to_l = list_to_dict(li, 1)
                idx = idx_to_l.keys()
                extra_li = extra_s_li.get(split, [])
                # link the data tsv
                source = dataset.get_data(split)
                out_split = 'train{}'.format(k)
                train_ldtsik.extend([(l, dataset, label_type, split, i,
                    k) for l, i in li])
                extra_dtsik.extend([(dataset, label_type, split, i, k) 
                    for l, i in extra_li])
                k = k + 1
                dest = out_dataset[label_type].get_data(
                        out_split)
                sources.append(source)
                logging.info('converting labels: {}-{}'.format(
                    dataset.name, split))
                source_origin_label = dataset.get_data(split,
                    'label', version=-1)
                
                converted_label = convert_label(source_origin_label,
                        idx, dataset._sourcelabel_to_targetlabels,
                        with_bb=with_bb)
                sources_origin_label.append(source_origin_label)
                # convert the file name
                logging.info('delifting the labels')
                for i in tqdm(idx):
                    l = converted_label[i]
                    l[1] = json.dumps(delift_one_image(l[1], tax))
                    l[0] = '{}_{}_{}'.format(dataset.name, split, l[0])
                label_file = out_dataset[label_type].get_data(out_split, 'label')
                logging.info('writing the label file {}'.format(label_file))
                tsv_writer(converted_label, label_file)
                sources_label.append(label_file)
        write_to_file('\n'.join(sources),
                out_dataset[label_type].get_data('trainX'))
        write_to_file('\n'.join(sources_label), 
                out_dataset[label_type].get_data('trainX', 'label'))
        write_to_file('\n'.join(sources_origin_label),
                out_dataset[label_type].get_data('trainX', 'origin.label'))

    logging.info('saving the shuffle file')
    type_to_ldsik = list_to_dict(train_ldtsik, 2)
    extra_type_to_dsik = list_to_dict(extra_dtsik, 1)
    for label_type in type_to_ldsik:
        ldsik = type_to_ldsik[label_type]
        shuffle_info = [(str(k), str(i)) for l, d, s, i, k in ldsik]
        shuffle_info = list(set(shuffle_info))
        if label_type in extra_type_to_dsik:
            dsik = extra_type_to_dsik[label_type]
            # we should not de-duplicate it because it comes from the duplicate
            # policy
            extra_shuffle_info = [(str(k), str(i) ) for d, s, i, k in dsik]
            shuffle_info.extend(extra_shuffle_info)
        random.shuffle(shuffle_info)
        tsv_writer(shuffle_info,
                out_dataset[label_type].get_shuffle_file('train'))

    populate_output_num_images(train_ldtsik, 'toTrain', tax.root)

def create_testX(test_ldtsi, tax, out_dataset):
    t_to_ldsi = list_to_dict(test_ldtsi, 2)
    for label_type in t_to_ldsi:
        sources = []
        sources_origin_label = []
        sources_label = []
        ldsi = t_to_ldsi[label_type]
        d_to_lsi = list_to_dict(ldsi, 1)
        k = 0
        all_ki = []
        with_bb = label_type == 'with_bb'
        for dataset in d_to_lsi:
            lsi = d_to_lsi[dataset]
            s_to_li = list_to_dict(lsi, 1)
            for split in s_to_li:
                li = s_to_li[split]
                idx = list_to_dict(li, 1).keys()
                out_split = 'test{}'.format(k)
                s_file = dataset.get_data(split)
                sources.append(s_file)
                src_img_tsv = TSVFile(s_file)
                source_origin_label = dataset.get_data(split, 'label',
                        version=-1)
                sources_origin_label.append(source_origin_label)
                src_label_tsv = TSVFile(source_origin_label)
                converted_label = convert_label(source_origin_label,
                        idx, dataset._sourcelabel_to_targetlabels,
                        with_bb=with_bb)
                for i in tqdm(idx):
                    l = converted_label[i]
                    l[1] = json.dumps(lift_one_image(l[1], tax))
                    l[0] = '{}_{}_{}'.format(dataset.name, split, l[0])
                all_ki.extend([(str(k), str(i)) for i in idx])
                label_file = out_dataset[label_type].get_data(out_split, 'label')
                tsv_writer(converted_label, label_file)
                sources_label.append(label_file)
                k = k + 1
        write_to_file('\n'.join(sources),
                out_dataset[label_type].get_data('testX'))
        write_to_file('\n'.join(sources_label), 
                out_dataset[label_type].get_data('testX', 'label'))
        write_to_file('\n'.join(sources_origin_label),
                out_dataset[label_type].get_data('testX', 'origin.label'))
        tsv_writer(all_ki, out_dataset[label_type].get_shuffle_file('test'))

def remove_or_duplicate(train_ldtsi, min_image, max_image):
    label_to_dtsi = list_to_dict(train_ldtsi, 0)
    extra_dtsi = []
    for label in label_to_dtsi:
        dtsi = label_to_dtsi[label]
        if len(dtsi) > max_image:
            # first remove the images with no bounding box
            num_remove = len(dtsi) - max_image
            type_to_dsi = list_to_dict(dtsi, 1)
            if 'no_bb' in type_to_dsi:
                dsi = type_to_dsi['no_bb']
                if num_remove >= len(dsi):
                    # remove all this images
                    del type_to_dsi['no_bb']
                    num_remove = num_remove - len(dsi)
                else:
                    random.shuffle(dsi)
                    type_to_dsi['no_bb'] = dsi[: len(dsi) - num_remove]
                    num_remove = 0
            if num_remove > 0:
                assert 'with_bb' in type_to_dsi
                dsi = type_to_dsi['with_bb']
                random.shuffle(dsi)
                dsi = sorted(dsi, key=lambda x: -x[0].cleaness)
                assert len(dsi) > num_remove
                type_to_dsi['with_bb'] = dsi[: len(dsi) - num_remove]
                num_remove = 0
            dtsi = dict_to_list(type_to_dsi, 1)
        elif len(dtsi) < min_image:
            num_duplicate = int(np.ceil(float(min_image) / len(dtsi)))
            logging.info('duplicate images for label of {}: {}->{}, {}'.format(
                label, len(dtsi), min_image, num_duplicate))
            extra_dtsi = (num_duplicate - 1) * dtsi
        label_to_dtsi[label] = dtsi
    logging.info('# train instances before duplication: {}'.format(len(train_ldtsi)))
    train_ldtsi = dict_to_list(label_to_dtsi, 0)
    logging.info('# train instances after duplication: {}'.format(len(train_ldtsi)))
    return train_ldtsi, extra_dtsi

def remove_test_in_train(train_ldtsi, test_ldtsi):
    logging.info('before len(train_ldtsi) = {}'.format(len(train_ldtsi)))
    set_train_dtsi = set((d, t, s, i) for l, d, t, s, i in train_ldtsi)
    set_train_ldtsi = set(train_ldtsi)
    #assert len(set_train_ldtsi) == len(train_ldtsi)
    set_test_dtsi = set((d, t, s, i) for l, d, t, s, i in test_ldtsi)
    result = [(l, d, t, s, i) for l, d, t, s, i in train_ldtsi 
            if (d, t, s, i) not in set_test_dtsi]
    logging.info('after len(train_ldtsi) = {}'.format(len(result)))
    return result

def split_train_test(ldtsi, num_test):
    # group by label_type
    t_to_ldsi = list_to_dict(ldtsi, 2)
    train_ldtsi = [] 
    test_ldtsi = []
    for label_type in sorted(t_to_ldsi.keys()):
        ldsi= t_to_ldsi[label_type]
        l_to_dsi = list_to_dict(ldsi, 0)
        for rootlabel in sorted(l_to_dsi.keys()):
            dsi = l_to_dsi[rootlabel]
            if len(dsi) < num_test:
                logging.info('rootlabel={}; label_type={}->less than {} images'.format(
                    rootlabel, label_type, len(dsi)))
            curr_num_test = min(num_test, int(len(dsi) / 2))
            random.shuffle(dsi)
            test_ldtsi.extend([(rootlabel, d, label_type, s, i) for d, s, i
                in dsi[:curr_num_test]])
            train_ldtsi.extend([(rootlabel, d, label_type, s, i) for d, s, i 
                in dsi[curr_num_test:]])
    return train_ldtsi, test_ldtsi

def build_taxonomy_impl(taxonomy_folder, **kwargs):
    random.seed(777)
    dataset_name = kwargs.get('data', 
            op.basename(taxonomy_folder))
    overall_dataset = TSVDataset(dataset_name)
    if op.isfile(overall_dataset.get_labelmap_file()):
        logging.info('ignore to build taxonomy since {} exists'.format(
            overall_dataset.get_labelmap_file()))
        return
    init_logging()
    all_tax = load_all_tax(taxonomy_folder)
    tax = merge_all_tax(all_tax)
    tax.update()
    initialize_images_count(tax.root)
    mapper = LabelToSynset()
    mapper.populate_noffset(tax.root)
    imagenet22k = TSVDatasetSource('imagenet22k_448', tax.root)
    if op.isfile(imagenet22k.get_labelmap_file()):
        disambibuity_noffsets(tax.root, imagenet22k.load_noffsets())
    else:
        logging.info('there is no imagenet22k_448 dataset to help identify the noffset')
    populate_url_for_offset(tax.root)

    ambigous_noffset_file = op.join(overall_dataset._data_root,
            'ambigous_noffsets.yaml')
    output_ambigous_noffsets(tax.root, ambigous_noffset_file)
    
    data_sources = []
    
    datas = kwargs['datas']
    cleaness = [d[1] if type(d) is tuple else 10 for d in datas]
    datas = [d[0] if type(d) is tuple else d for d in datas]
    
    logging.info('extract the images from: {}'.format(','.join(datas)))

    for d, c in izip(datas, cleaness):
        dataset = TSVDatasetSource(d, tax.root)
        dataset.cleaness = c
        data_sources.append(dataset)
    
    for s in data_sources:
        s.populate_info(tax.root)

    populate_cum_images(tax.root)

    labels, child_parent_sgs = child_parent_print_tree2(tax.root, 'name')

    label_map_file = overall_dataset.get_labelmap_file() 
    write_to_file('\n'.join(map(lambda l: l.encode('utf-8'), labels)), 
            label_map_file)
    # save the parameter
    write_to_yaml_file((taxonomy_folder, kwargs), op.join(overall_dataset._data_root,
            'generate_parameters.yaml'))
    dest_taxonomy_folder = op.join(overall_dataset._data_root,
            'taxonomy_folder')
    if op.isdir(dest_taxonomy_folder):
        shutil.rmtree(dest_taxonomy_folder)
    shutil.copytree(taxonomy_folder, dest_taxonomy_folder)

    out_dataset = {'with_bb': TSVDataset(dataset_name + '_with_bb'),
            'no_bb': TSVDataset(dataset_name + '_no_bb')}

    for label_type in out_dataset:
        target_file = out_dataset[label_type].get_labelmap_file()
        ensure_directory(op.dirname(target_file))
        shutil.copy(label_map_file, target_file)

    logging.info('cum_images_with_bb: {}'.format(tax.root.cum_images_with_bb))
    logging.info('cum_images_no_bb: {}'.format(tax.root.cum_images_no_bb))

    # write the simplified version of the tree
    dest = op.join(overall_dataset._data_root, 'root.simple.yaml')
    write_to_yaml_file(tax.dump(['images_with_bb']), dest)

    tree_file = overall_dataset.get_tree_file()
    write_to_file('\n'.join(['{} {}{}'.format(c.encode('utf-8'), p, '' if sg < 0 else ' {}'.format(sg))
                             for c, p, sg in child_parent_sgs]),
            tree_file)
    for label_type in out_dataset:
        target_file = out_dataset[label_type].get_tree_file()
        ensure_directory(op.dirname(target_file))
        shutil.copy(tree_file, target_file)

    node_should_have_images(tax.root, 200, 
            op.join(overall_dataset._data_root, 'labels_with_few_images.yaml'))

    # get the information of all train val
    train_vals = []
    ldtsi = []
    logging.info('collecting all candidate images')
    for label_type in out_dataset:
        for dataset in data_sources:
            split_idxes = dataset.select_tsv_rows(label_type)
            for rootlabel, split, idx in split_idxes:
                ldtsi.append((rootlabel, dataset, label_type, split, idx))
    # we need to remove the duplicates. the duplicates could come from such
    # cases: for example, we have Laptop and laptop in the image. Both of the
    # labels are mapped to laptop, which is in the target domain. In this case,
    # the image could be in the list twice
    ldtsi = list(set(ldtsi))

    # split into train val
    num_test = kwargs.get('num_test', 50)
    logging.info('splitting the images into train and test')

    logging.info('duplicating or removing the train images')
    # for each label, let's duplicate the image or remove the image
    max_image = kwargs.get('max_image_per_label', 1000)
    min_image = kwargs.get('min_image_per_label', 200)

    ldtsi, extra_dtsi = remove_or_duplicate(ldtsi, 0, max_image * 3)
    assert len(extra_dtsi) == 0

    train_ldtsi, test_ldtsi = split_train_test(ldtsi, num_test)

    train_ldtsi = remove_test_in_train(train_ldtsi, test_ldtsi)

    train_ldtsi, extra_dtsi = remove_or_duplicate(train_ldtsi, min_image, max_image)

    logging.info('creating the train data')
    create_trainX(train_ldtsi, extra_dtsi, tax, out_dataset)

    populate_output_num_images(test_ldtsi, 'toTest', tax.root)

    # dump the tree to yaml format
    dest = op.join(overall_dataset._data_root, 'root.yaml')
    d = tax.dump()
    write_to_yaml_file(d, dest)
    for label_type in out_dataset:
        target_file = op.join(out_dataset[label_type]._data_root, 'root.yaml')
        ensure_directory(op.dirname(target_file))
        shutil.copy(dest, target_file)

    create_testX(test_ldtsi, tax, out_dataset)

    logging.info('done')

def delift_one_image(rects, tax):
    # currently, for the training, we need to delift the label. That is, if it
    # is man, we should remove the person label
    rects2 = []
    for curr_r in rects:
        curr_label = curr_r['class']
        ious = [calculate_iou(r['rect'], curr_r['rect']) for r in rects2]
        same_place_rects = [r for i, r in izip(ious, rects2) if i > 0.9]
        # if current label is one of parent of the same_place rects, ignore it
        ignore = False
        for same_place_r in same_place_rects:
            ancestors = tax.name_to_ancestors[same_place_r['class']]
            if curr_label in ancestors or same_place_r['class'] == curr_label:
                ignore = True
                break
        if ignore:
            continue
        ancestors = tax.name_to_ancestors[curr_label]
        to_removed = []
        for same_place_r in same_place_rects:
            if same_place_r['class'] in ancestors:
                to_removed.append(same_place_r)
        for t in to_removed:
            rects2.remove(t)
        rects2.append(curr_r)
    return rects2

def lift_one_image(rects, tax):
    rects2 = []
    for curr_r in rects:
        label = curr_r['class']
        all_label = tax.name_to_ancestors[label]
        all_label.add(label)
        for l in all_label:
            same_label_rects = [r for r in rects2 if r['class'] == l]
            ious = [calculate_iou(r['rect'], curr_r['rect']) for r in
                same_label_rects]
            if len(ious) > 0 and max(ious) > 0.9:
                continue
            else:
                r = copy.deepcopy(curr_r)
                r['class'] = l
                rects2.append(r)
    return rects2

def populate_output_num_images(ldtX, suffix, root):
    label_to_node = {n.name: n for n in root.iter_search_nodes() if n != root}
    targetlabel_to_dX = list_to_dict(ldtX, 0)
    for targetlabel in targetlabel_to_dX:
        dtX = targetlabel_to_dX[targetlabel]
        dataset_to_X = list_to_dict(dtX, 0)
        for dataset in dataset_to_X:
            X = dataset_to_X[dataset]
            if len(X) == 0:
                continue
            key = '{}_{}'.format(dataset.name, suffix)
            value = len(X)
            label_to_node[targetlabel].add_feature(key, value)
        labeltype_to_dX = list_to_dict(dtX, 1)
        for labeltype in labeltype_to_dX:
            dX = labeltype_to_dX[labeltype]
            key = '{}_{}'.format(labeltype, suffix)
            value = len(dX)
            label_to_node[targetlabel].add_feature(key, value)

def output_ambigous_noffsets(root, ambigous_noffset_file):
    ambigous = []
    for node in root.iter_search_nodes():
        if hasattr(node, 'noffsets') and node.noffset is None:
            noffsets = node.noffsets.split(',')
            d = create_info_for_ambigous_noffset(node.name, noffsets)
            d['parent_name'] = ','.join([n.name for n in
                        node.get_ancestors()[:-1]])
            ambigous.append(d)
    if len(ambigous) > 0:
        logging.info('output ambigous terms to {}'.format(ambigous_noffset_file))
        write_to_yaml_file(ambigous, ambigous_noffset_file)
    else:
        logging.info('Congratulations on no ambigous terms.')

def output_ambigous_noffsets_main(tax_input_folder, ambigous_file_out):
    all_tax = load_all_tax(tax_input_folder)
    tax = merge_all_tax(all_tax)
    mapper = LabelToSynset()
    mapper.populate_noffset(tax.root)
    imagenet22k = TSVDatasetSource('imagenet22k')
    if op.isfile(imagenet22k.get_labelmap_file()):
        logging.info('remove the noffset if it is not in imagenet22k')
        disambibuity_noffsets(tax.root, imagenet22k.load_noffsets())
    else:
        logging.info('no imagenet22k data used to help remove noffset ambiguities')

    populate_url_for_offset(tax.root)

    output_ambigous_noffsets(tax.root, ambigous_file_out)

def convert_inverted_file(name):
    d = TSVDataset(name)
    splits = ['train', 'trainval', 'test']
    for split in splits:
        logging.info('loading {}-{}'.format(name, split))
        x = d.load_inverted_label(split)
        def gen_rows():
            for label in x:
                idx = x[label]
                yield label, ' '.join(map(str, idx))

        inverted_file = d.get_data(split, 'inverted.label')
        target_file = op.splitext(inverted_file)[0] + '.tsv'
        if not op.isfile(inverted_file):
            if op.isfile(target_file):
                os.remove(target_file)
            continue
        if op.isfile(target_file):
            continue
        tsv_writer(gen_rows(), target_file)

def standarize_crawled(tsv_input, tsv_output):
    rows = tsv_reader(tsv_input)
    def gen_rows():
        for i, row in enumerate(rows):
            if (i % 1000) == 0:
                logging.info(i)
            image_str = row[-1]
            image_label = row[0]
            rects = [{'rect': [0, 0, 0, 0], 'class': image_label}]
            image_name = '{}_{}'.format(op.basename(tsv_input), i)
            yield image_name, json.dumps(rects), image_str
    tsv_writer(gen_rows(), tsv_output)

def get_data_sources():
    return [
        ('coco2017', 10),
        ('voc0712', 10), 
        ('Naturalist', 10),
        ('elder', 10),
        ('imagenet200Diff', 10),
        ('OpenImageV4_448', 10),
        ('open_images_clean_1', 9),
        ('open_images_clean_2', 9),
        ('open_images_clean_3', 9),
        ('imagenet1kLocClean', 9),
        ('imagenet3k_448Clean', 9),
        ('VisualGenomeClean', 9),
        ('brand1048Clean', 8),
        ('mturk700_url_as_keyClean', 8),
        ('crawl_office_v1', 8),
        ('crawl_office_v2', 8),
        ('Materialist', 8),
        ('4000_Full_setClean', 7),
        ('MSLogoClean', 8),
        ('clothingClean', 8),
        ('imagenet22k_448', 7),
        ]


def get_img_url(img_key):
    clean_name = _map_img_key_to_name(img_key)
    url = _get_url_from_name(clean_name)
    return url

def _map_img_key_to_name(key):
    _EXT = ".jpg"
    if key.startswith("brand"):
        return "brand" + str(hash(key)) + _EXT
    else:
        key = key.lower()
        if key.endswith(_EXT):
            return key
        else:
            return key + _EXT

def _get_url_from_name(name):
    _SITE = 'https://cogsimagestorage.blob.core.windows.net/'
    _CONTAINER_NAME = "detectortrain"
    return _SITE + _CONTAINER_NAME + "/" + name

def parse_combine(key, datas):
    def found_starts(key, datas):
        found = None
        for data in datas:
            if key.startswith(data + '_'):
                assert found is None
                found = data
        return found
    data = found_starts(key, datas)
    if data is None:
        return None, None, key
    else:
        key = key[len(data) + 1 : ]
        splits = ['train', 'trainval', 'test']
        split = found_starts(key, splits)
        key = key[len(split) + 1 : ]
        return data, split, key

def convert_to_uhrs_with_url(data):
    dataset = TSVDataset(data)
    datas = [d for d, _ in get_data_sources()]
    for split in ['train', 'trainval', 'test']:
        if not dataset.has(split, 'label'):
            continue
        v = dataset.get_latest_version(split, 'label')
        def gen_rows():
            for row in dataset.iter_data(split, 'label', version=v):
                key = row[0]
                _, _, key = parse_combine(key, datas)
                row.append(get_img_url(key))
                yield row
        dataset.write_data(gen_rows(), split, 
                'url', version=v)

def find_same_rects(target, rects, iou=0.95):
    same_class_rects = [r for r in rects if r['class'] == target['class']]
    return [r for r in same_class_rects if 
        calculate_iou(target['rect'], r['rect']) > iou]

def rect_in_rects(target, rects, iou=0.95):
    same_class_rects = [r for r in rects if r['class'] == target['class']]
    return any(r for r in same_class_rects if 
        calculate_iou(target['rect'], r['rect']) > iou)

def load_key_rects(iter_data):
    result = []
    logging.info('loading key rects')
    for row in tqdm(iter_data):
        assert len(row) == 2
        result.append((row[0], json.loads(row[1])))
    return result

def convert_uhrs_result_back_to_sources(in_tsv, debug=True, tree_file=None):
    rows = tsv_reader(in_tsv)
    key_rects3 = []
    num_yes, num_no, num_un = 0, 0, 0
    for row in rows:
        # key, yes, no, uncertain
        assert len(row) == 4
        rects_yes = json.loads(row[1])
        rects_no = json.loads(row[2])
        rects_un = json.loads(row[3])
        num_yes = num_yes + len(rects_yes)
        num_no = num_no + len(rects_no)
        num_un = num_un + len(rects_un)
        key_rects3.append([row[0], [rects_yes, rects_no, rects_un]])

    logging.info('#yes={}; #no={}; #un={}; #yes/(#yes+#no+#un)={}'.format(
        num_yes, num_no, num_un, 1.*num_yes/(num_yes+num_no+num_un)))
    
    if tree_file:
        tax = Taxonomy(load_from_yaml_file(tree_file))
        mapper = LabelToSynset()
        mapper.populate_noffset(tax.root)

    datas = get_data_sources()
    datas = [data for data, _ in datas]
    datasplitkey_rects3 = [[parse_combine(key, datas), rects3] 
            for key, rects3 in key_rects3]
    data_split_key_rects3 = [(data, split, key, rects3) 
            for (data, split, key), rects3 in datasplitkey_rects3]

    data_to_split_key_rects3 = list_to_dict(data_split_key_rects3, 0)

    for data in data_to_split_key_rects3:
        logging.info(data)
        if tree_file:
            source_dataset = TSVDatasetSource(data, tax.root)
            source_dataset._ensure_initialized()
        else:
            source_dataset = TSVDataset(data)
        split_key_rects3 = data_to_split_key_rects3[data]
        split_to_key_rects3 = list_to_dict(split_key_rects3, 0)
        for split in split_to_key_rects3:
            logging.info(split)
            key_rects3 = split_to_key_rects3[split]
            key_to_rects3 = list_to_dict(key_rects3, 0)
            v = source_dataset.get_latest_version(split, 'label')
            logging.info('{} - {}'.format(data, split))
            source_key_rects = load_key_rects(source_dataset.iter_data(split, 'label', version=v))
            is_equal = True
            num_added, num_removed = 0, 0
            meta = {'in_tsv': in_tsv}
            for i, (key, origin_rects) in tqdm(enumerate(source_key_rects)):
                if debug:
                    old_origin_rects = copy.deepcopy(origin_rects)
                yes_rects, no_rects, un_rects = key_to_rects3.get(key, [[[], [], []]])[0]
                # if yes_rects are not in original, add it
                for r in yes_rects:
                    same_rects = find_same_rects(r, origin_rects)
                    if len(same_rects) > 0:
                        for s in same_rects:
                            s['uhrs_confirm'] = s.get('uhrs_confirm', 0) + 1
                    else:
                        r['uhrs_confirm'] = r.get('uhrs_confirm', 0) + 1
                        origin_rects.append(copy.deepcopy(r))
                        is_equal = False
                        num_added = num_added + 1
                # if no_rects are in original, remove it
                for r in no_rects:
                    delete_rects = find_same_rects(r, origin_rects)
                    if tree_file:
                        if r['class'] in source_dataset._targetlabel_to_sourcelabels:
                            for reverse_class in source_dataset._targetlabel_to_sourcelabels[r['class']]:
                                r2 = copy.deepcopy(r)
                                r2['class'] = reverse_class
                                delete_rects2 = find_same_rects(r2, origin_rects)
                                delete_rects.extend(delete_rects2)
                    if 'class_from' in r:
                        r2 = copy.deepcopy(r)
                        r2['class'] = r2['class_from']
                        delete_rects2 = find_same_rects(r2, origin_rects)
                        delete_rects.extend(delete_rects2)
                    for d in delete_rects:
                        # delete rects may include duplicate terms
                        if d in origin_rects:
                            origin_rects.remove(d)
                            num_removed = num_removed + 1
                            is_equal = False
                if debug:
                    if len(origin_rects) != len(old_origin_rects):
                        for _, _, im_str in source_dataset.iter_data(split, filter_idx=[i]):
                            im = img_from_base64(im_str)
                            old_im = im.copy()
                            draw_bb(old_im, [r['rect'] for r in old_origin_rects],
                                    [r['class'] for r in old_origin_rects])
                            new_im = im.copy()
                            draw_bb(new_im, [r['rect'] for r in origin_rects],
                                    [r['class'] for r in origin_rects])
                            yes_im = im.copy()
                            draw_bb(yes_im, [r['rect'] for r in yes_rects],
                                    [r['class'] for r in yes_rects])
                            no_im = im.copy()
                            draw_bb(no_im, [r['rect'] for r in no_rects],
                                    [r['class'] for r in no_rects])

                            logging.info(pformat(old_origin_rects))
                            logging.info(pformat(origin_rects))
                            logging.info(pformat(yes_rects))
                            logging.info(pformat(no_rects))
                            show_images([old_im, new_im, yes_im, no_im], 2, 2)
    
            meta['num_added_rects'] = num_added
            meta['num_removed_rects'] = num_removed
            meta['total_number_images'] = len(source_key_rects)
    
            meta['avg_added_rects'] = 1. * num_added / meta['total_number_images']
            meta['avg_removed_rects'] = 1. * num_removed / meta['total_number_images']
    
            assert not source_dataset.has(split, 'label', v + 1)
            if not is_equal:
                source_dataset.write_data(((key, json.dumps(rects)) for key, rects in source_key_rects),
                        split, 'label', version=v+1)
                meta_file = source_dataset.get_data(split, 'label.metadata', version=v+1) + '.yaml'
                write_to_yaml_file(meta, meta_file)
                logging.info(pformat(meta))
            else:
                logging.info('equal - {} - {}'.format(data, split))
        populate_dataset_details(data)


def process_tsv_main(**kwargs):
    if kwargs['type'] == 'gen_tsv':
        input_folder = kwargs['input']
        output_folder = kwargs['ouput']
        gen_tsv_from_labeling(input_folder, output_folder)
    elif kwargs['type'] == 'gen_term_list':
        tax_folder = kwargs['input']
        term_list = kwargs['output']
        gen_term_list(tax_folder, term_list)
    elif kwargs['type'] == 'gen_noffset':
        tax_input_folder = kwargs['input']
        tax_output_folder = kwargs['output']
        gen_noffset(tax_input_folder, tax_output_folder)
    elif kwargs['type'] == 'ambigous_noffset':
        tax_input_folder = kwargs['input']
        ambigous_file_out = kwargs['output']
        output_ambigous_noffsets_main(tax_input_folder, ambigous_file_out)
    elif kwargs['type'] == 'standarize_crawled':
        tsv_input = kwargs['input']
        tsv_output = kwargs['output']
        standarize_crawled(tsv_input, tsv_output)
    elif kwargs['type'] == 'taxonomy_to_tsv':
        taxonomy_folder = kwargs['input']
        build_taxonomy_impl(taxonomy_folder, **kwargs)
    elif kwargs['type'] == 'yolo_model_convert':
        old_proto = kwargs['prototxt']
        old_model = kwargs['model']
        new_model = kwargs['output']
        yolo_old_to_new(old_proto, old_model, new_model)
    elif kwargs['type'] == 'build_data_index':
        data = kwargs['input']
        populate_dataset_details(data)
    elif kwargs['type'] == 'build_all_data_index':
        populate_all_dataset_details()
    elif kwargs['type'] == 'extract_for_uhrs':
        taxonomy_folder = kwargs['input']
        data = kwargs['data']
        build_taxonomy_impl(taxonomy_folder,
                data=data,
                datas=get_data_sources(),
                max_image_per_label=kwargs.get('max_image_per_label', 10000000),
                min_image_per_label=kwargs.get('min_image_per_label', 0),
                num_test=0)
        convert_to_uhrs_with_url(data + '_with_bb')
    elif kwargs['type'] == 'merge_labels':
        in_tsv = kwargs['input']
        convert_uhrs_result_back_to_sources(in_tsv)
    else:
        logging.info('unknown task {}'.format(kwargs['type']))

def parse_args():
    parser = argparse.ArgumentParser(description='TSV Management')
    parser.add_argument('-t', '--type', help='what type it is: gen_tsv',
            type=str, required=True)
    parser.add_argument('-i', '--input', help='input',
            type=str, required=False)
    parser.add_argument('-p', '--prototxt', help='proto file',
            type=str, required=False)
    parser.add_argument('-m', '--model', help='model file',
            type=str, required=False)
    parser.add_argument('-o', '--output', help='output',
            type=str, required=False)
    parser.add_argument('-d', '--datas', 
            default=argparse.SUPPRESS,
            nargs='*',
            help='which data are used for taxonomy_to_tsv',
            type=str, 
            required=False)
    parser.add_argument('-da', '--data', 
            default=argparse.SUPPRESS,
            help='the dataset name under data/',
            type=str, 
            required=False)
    parser.add_argument('-maxi', '--max_image_per_label', 
            default=argparse.SUPPRESS,
            type=int, 
            required=False)
    parser.add_argument('-mini', '--min_image_per_label', 
            default=argparse.SUPPRESS,
            type=int, 
            required=False)
    return parser.parse_args()

if __name__ == '__main__':
    init_logging()
    args = parse_args()
    process_tsv_main(**vars(args))

