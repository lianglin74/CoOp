from itertools import izip
from process_image import show_images
import shutil
import glob
import yaml
import magic
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
from qd_common import list_to_dict
from qd_common import dict_to_list
from qd_common import parse_test_data
from tsv_io import load_labels

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


def gen_html_tree_view(data):
    dataset = TSVDataset(data)
    file_name = op.join(dataset._data_root, 'root.yaml')
    with open(file_name, 'r') as fp:
        config_tax = yaml.load(fp)
    tax = Taxonomy(config_tax)
    def gen_html_tree_view_rec(root):
        '''
        include itself
        '''
        if len(root.children) == 0:
            s = u"<li data-jstree='{{\"icon\":\"glyphicon glyphicon-leaf\"}}'>{}</li>".format(root.name)
            return s
        else:
            result = []
            result.append('<li><span>{}</span>'.format(root.name))
            result.append('<ul>')
            for c in root.children:
                r = gen_html_tree_view_rec(c)
                result.append(r)
            result.append('</ul>')
            result.append('</li>')
            return '\n'.join(result)
    s = gen_html_tree_view_rec(tax.root)
    return s


def gt_predict_images(predicts, gts, test_data, target_images, start_id, threshold,
        label_to_idx, image_aps, test_data_split='test'): 
    test_dataset = TSVDataset(test_data)
    test_tsv = TSVFile(test_dataset.get_data(test_data_split))
    for i in xrange(start_id, len(target_images)):
        key = target_images[i]
        logging.info('key = {}, ap = {}'.format(key, image_aps[i][1]))
        idx = label_to_idx[key]
        row = test_tsv.seek(idx)
        im = img_from_base64(row[2])
        origin = np.copy(im)
        im_gt = np.copy(im)
        draw_bb(im_gt, [g['rect'] for g in gts[key]],
                [g['class'] for g in gts[key]])
        im_pred = im
        rects = [p for p in predicts[key] if p['conf'] > threshold]
        draw_bb(im_pred, [r['rect'] for r in rects],
                [r['class'] for r in rects], 
                [r['conf'] for r in rects])
        yield key, origin, im_gt, im_pred, image_aps[i][1]

def get_confusion_matrix_by_predict_file(full_expid, 
        predict_file, threshold, test_data_split='test'):

    test_data = parse_test_data(predict_file)
    predicts, _ = load_labels(op.join('output', full_expid, 'snapshot', predict_file))

    # load the gt
    test_dataset = TSVDataset(test_data)
    test_label_file = test_dataset.get_data(test_data_split, 'label')
    gts, label_to_idx = load_labels(test_label_file)

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


def tsv_details(tsv_file):
    rows = tsv_reader(tsv_file)
    label_count = {}
    sizes = []
    for i, row in enumerate(rows):
        if (i % 1000) == 0:
            logging.info('get tsv details: {}-{}'.format(tsv_file, i))
        rects = json.loads(row[1])
        if type(rects) is list:
            # this is the detection dataset
            # convert it to str. if it is unicode, in yaml, there will be some
            # special tags, which is annoying
            curr_labels = set(str(rect['class']) for rect in rects)
        else:
            # this is classification dataset
            assert type(rects) is int
            curr_labels = [rects]
        for c in curr_labels:
            if c in label_count:
                label_count[c] = label_count[c] + 1
            else:
                label_count[c] = 1
        im = img_from_base64(row[2])
        sizes.append(im.shape[:2])
    min_size_count = sizes[0][0] * sizes[0][1]
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

def populate_dataset_details(data):
    dataset = TSVDataset(data)

    def details_tsv(tsv_file):
        out_file = get_meta_file(tsv_file)
        if not op.isfile(out_file) and op.isfile(tsv_file):
            details = tsv_details(tsv_file)
            write_to_yaml_file(details, out_file)

    splits = ['trainval', 'train', 'test']
    for split in splits:
        details_tsv(dataset.get_data(split))

    # for each data tsv, generate the label tsv and the inverted file
    for split in splits:
        full_tsv = dataset.get_data(split)
        label_tsv = dataset.get_data(split, 'label')
        if not op.isfile(label_tsv) and op.isfile(full_tsv):
            extract_label(full_tsv, label_tsv)
        inverted = dataset.get_data(split, 'inverted.label')
        if not op.isfile(inverted) and op.isfile(label_tsv):
            create_inverted_tsv(label_tsv, inverted,
                    dataset.get_labelmap_file())

    # generate the rows with duplicate keys
    for split in splits: 
        label_tsv = dataset.get_data(split, 'label')
        duplicate_tsv = dataset.get_data(split, 'key_duplicate')
        if op.isfile(label_tsv) and not op.isfile(duplicate_tsv):
            assert detect_duplicate_key(label_tsv, duplicate_tsv) == 0

    if op.isfile(dataset.get_data('trainX')):
        inverted_file = dataset.get_data('train', 'inverted.label')
        if not op.isfile(inverted_file):
            train_files = load_list_file(dataset.get_data('trainX'))
            #train_label_files = ['{}.label{}'.format(*op.splitext(f)) 
                    #for f in train_files]
            train_label_files = load_list_file(dataset.get_data('trainX',
                'label'))
            train_label_tsvs = [TSVFile(f) for f in train_label_files]
            shuffle_file = dataset.get_shuffle_file('train')
            shuffle_tsv_rows = tsv_reader(shuffle_file)
            inverted = {}
            for i, row in enumerate(shuffle_tsv_rows):
                if (i % 1000) == 0:
                    logging.info(i)
                label_row = train_label_tsvs[int(row[0])].seek(int(row[1]))
                all_class = []
                for l in json.loads(label_row[1]):
                    all_class.append(l['class'])
                for c in set(all_class):
                    if c in inverted:
                        inverted[c].append(i)
                    else:
                        inverted[c] = [i]
            def gen_inverted_trainX():
                for c in inverted:
                    yield c, ' '.join(map(str, inverted[c]))
            tsv_writer(gen_inverted_trainX(), inverted_file)
    
    # generate lineidx if it is not generated
    for split in splits:
        lineidx = dataset.get_lineidx(split)
        full_tsv = dataset.get_data(split)
        if not op.isfile(lineidx) and op.isfile(full_tsv):
            logging.info('no lineidx for {}. generating...'.format(split))
            generate_lineidx(full_tsv, lineidx)

    # generate the label map if there is no
    if not op.isfile(dataset.get_labelmap_file()) and \
            not op.islink(dataset.get_labelmap_file()):
        logging.info('no labelmap, generating...')
        labelmap = []
        for split in splits:
            label_tsv = dataset.get_data(split, 'label')
            if not op.isfile(label_tsv):
                continue
            for row in tsv_reader(label_tsv):
                labelmap.extend(set([rect['class'] for rect in
                    json.loads(row[1])]))
        if len(labelmap) == 0: 
            logging.warning('there are no labels!')
        labelmap = list(set(labelmap))
        logging.info('find {} labels'.format(len(labelmap)))
        write_to_file('\n'.join(labelmap), dataset.get_labelmap_file())

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

def visualize_box(data, split, label, start_id, color_map={}):
    dataset = TSVDataset(data)
    logging.info('loading inverted label')
    inverted = dataset.load_inverted_label(split)
    logging.info('inverted label loaded')
    logging.info('keys: {}'.format(inverted.keys()))
    if label != 'any':
        if label not in inverted:
            return
        idx = inverted[label]
    is_composite = False
    if split == 'train' and not op.isfile(dataset.get_data(split)):
        is_composite = True
        tsvs = [TSVFile(f) for f in dataset.get_train_tsvs()]
        tsv_labels = [TSVFile(f) for f in dataset.get_train_tsvs('label')]
        shuffle_tsv_rows = tsv_reader(dataset.get_shuffle_file(split))
        shuffle = []
        for row in shuffle_tsv_rows:
            shuffle.append([int(row[0]), int(row[1])])
        if label == 'any':
            idx = range(len(shuffle))
    else:
        tsv = TSVFile(dataset.get_data(split))
        if label == 'any':
            idx = range(tsv.num_rows())
    logging.info('start to read')
    for i in idx[start_id: ]:
        all_image = []
        if is_composite:
            row_image = tsvs[shuffle[i][0]].seek(shuffle[i][1])
            row_label = tsv_labels[shuffle[i][0]].seek(shuffle[i][1])
        else:
            row_image = tsv.seek(i)
            row_label = row_image
        im = img_from_base64(row_image[-1])
        origin = np.copy(im)
        labels = try_json_parse(row_label[1])
        new_name = row_image[0].replace('/', '_').replace(':', '')
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
            draw_bb(im, all_rect, all_class)
            yield new_name, origin, im
        else:
            yield new_name, origin, im


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
        self._sourcelabel_targetlabel = None
        self._sourcelabel_to_targetlabels = None
        self._targetlabel_to_sourcelabels = None
        self._sourcelabel_to_imagecount = None
        self._split_label_idx = None # list of <split, label, idx>
        self._datasetlabel_to_splitidx = None
        self._initialized = False

    def populate_info(self, root):
        self._ensure_initialized()
        for node in root.iter_search_nodes():
            if root == node:
                continue
            if node.name in self._targetlabel_to_sourcelabels:
                sourcelabels = self._targetlabel_to_sourcelabels[node.name]
                for sourcelabel in sourcelabels:
                    c = self._datasetlabel_to_count.get(sourcelabel, 0)
                    if self._type == 'with_bb':
                        logging.info('with_bb: {}: {}->{}'.format(self.name,
                            node.name, c))
                        node.images_with_bb = node.images_with_bb + c
                    else:
                        assert self._type == 'no_bb'
                        logging.info('no_bb: {}: {}->{}'.format(self.name,
                            node.name, c))
                        node.images_no_bb = node.images_no_bb + c

    def _ensure_initialized(self):
        if self._initialized:
            return
        populate_dataset_details(self.name)
        splits = ['trainval', 'train', 'test']
        # check the type of the dataset
        for split in splits:
            label_tsv = self.get_data(split, 'label')
            if not op.isfile(label_tsv):
                continue
            for row in tsv_reader(label_tsv):
                rects = json.loads(row[1])
                if any(np.sum(r['rect']) > 1 for r in rects):
                    self._type = 'with_bb'
                else:
                    self._type = 'no_bb'
                break
        logging.info('identify {} as {}'.format(self.name, self._type))
        
        # list of <split, label, idx>
        self._split_label_idx = []
        for split in splits:
            logging.info('loading the inverted file: {}-{}'.format(self.name,
                split))
            inverted = self.load_inverted_label(split)
            label_idx = dict_to_list(inverted, 0)
            for label, idx in label_idx:
                self._split_label_idx.append((split, label, idx))

        self._datasetlabel_to_splitidx = list_to_dict(self._split_label_idx, 1)
        self._datasetlabel_to_count = {l: len(self._datasetlabel_to_splitidx[l]) for l in
                self._datasetlabel_to_splitidx}

        self._datasetlabel_to_rootlabel = self.get_label_mapper()
        self._initialized = True

    def get_label_mapper(self):
        root = self._root
        labelmap = self.load_labelmap()
        noffsets = self.load_noffsets()
        tree_noffsets = {}
        for node in root.iter_search_nodes():
            if node == root or not node.noffset:
                continue
            for s in node.noffset.split(','):
                tree_noffsets[s] = node.name
        tree_labels = {}
        for node in root.iter_search_nodes():
            if node == root:
                continue
            assert node.name.lower() not in tree_labels
            # we will keep the lower case always for case-insensitive
            # comparison
            tree_labels[node.name.lower()] = node.name

            #if hasattr(node, 'noffset') and node.noffset:
                #alter_terms = [s.strip() for s in node.noffset.split(',')]
                #for term in alter_terms:
                    #if term.lower() in tree_labels:
                        #assert tree_labels[term.lower()] == node.name
                    #else:
                        #tree_labels[term.lower()] = node.name

            #if hasattr(node, 'alternateTerms'):
                #alter_terms = [s.strip() for s in node.alternateTerms.split(',')]
                #for term in alter_terms:
                    #if term.lower() in tree_labels:
                        #assert tree_labels[term.lower()] == node.name
                    #else:
                        #tree_labels[term.lower()] = node.name

        sourcelabel_targetlabel = [] 

        result = {}
        for l, ns in izip(labelmap, noffsets):
            if l.lower() in tree_labels:
                sourcelabel_targetlabel.append((l, tree_labels[l.lower()]))
                result[l] = tree_labels[l.lower()]
            elif ns != '':
                for n in ns.split(','):
                    n = n.strip()
                    if n in tree_noffsets:
                        sourcelabel_targetlabel.append((l, tree_noffsets[n]))
                        result[l] = tree_noffsets[n]

        self._sourcelabel_targetlabel = sourcelabel_targetlabel
        self._sourcelabel_to_targetlabels = list_to_dict(sourcelabel_targetlabel,
                0)
        self._targetlabel_to_sourcelabels = list_to_dict(sourcelabel_targetlabel,
                1)
    
        return result

    def select_tsv_rows(self, label_type):
        self._ensure_initialized()
        assert self._type is not None
        if label_type != self._type:
            return []
        result = []
        for datasetlabel in self._datasetlabel_to_splitidx:
            if datasetlabel in self._datasetlabel_to_rootlabel:
                split_idxes = self._datasetlabel_to_splitidx[datasetlabel]
                rootlabel = self._datasetlabel_to_rootlabel[datasetlabel]
                result.extend([(rootlabel, split, idx) for split, idx in
                    split_idxes])
        return result

    def gen_tsv_rows(self, root, label_type):
        selected_info = self.select_tsv_rows(root, label_type)
        mapper = self._datasetlabel_to_rootlabel
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
    for rect in rects:
        if rect['class'] in label_mapper:
            rect['class'] = label_mapper[rect['class']]
        else:
            to_remove.append(rect)
    for t in to_remove:
        rects.remove(t)

def convert_label(label_tsv, idx, label_mapper):
    '''
    '''
    tsv = TSVFile(label_tsv)
    result = None
    for i in idx:
        row = tsv.seek(i)
        rects = json.loads(row[1])
        if result is None:
            result = [len(row) * ['d']] * tsv.num_rows()
        to_remove = []
        for rect in rects:
            if rect['class'] in label_mapper:
                rect['class'] = label_mapper[rect['class']]
            else:
                to_remove.append(rect)
        for t in to_remove:
            rects.remove(t)
        assert len(rects) > 0
        row[1] = json.dumps(rects)
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
    
    datas = kwargs.get('datas', ['voc20', 'coco2017', 'imagenet3k_448',
        'crawl_office_v2', 'crawl_office_v1'])
    logging.info('extract the images from: {}'.format(','.join(datas)))

    for d in datas:
        data_sources.append(TSVDatasetSource(d, tax.root))
    
    for s in data_sources:
        s.populate_info(tax.root)

    populate_cum_images(tax.root)

    labels, child_parent_sgs = child_parent_print_tree2(tax.root, 'name')

    label_map_file = overall_dataset.get_labelmap_file() 
    write_to_file('\n'.join(map(lambda l: l.encode('utf-8'), labels)), 
            label_map_file)

    out_dataset = {'with_bb': TSVDataset(dataset_name + '_with_bb'),
            'no_bb': TSVDataset(dataset_name + '_no_bb')}

    for label_type in out_dataset:
        target_file = out_dataset[label_type].get_labelmap_file()
        ensure_directory(op.dirname(target_file))
        shutil.copy(label_map_file, target_file)

    logging.info('cum_images_with_bb: {}'.format(tax.root.cum_images_with_bb))
    logging.info('cum_images_no_bb: {}'.format(tax.root.cum_images_no_bb))

    # dump the tree to yaml format
    dest = op.join(overall_dataset._data_root, 'root.yaml')
    d = tax.dump()
    write_to_yaml_file(d, dest)

    # write the simplified version of the tree
    dest = op.join(overall_dataset._data_root, 'root.simple.yaml')
    write_to_yaml_file(tax.dump(['images_with_bb']), dest)

    tree_file = overall_dataset.get_tree_file()
    write_to_file('\n'.join(['{} {}{}'.format(c.encode('utf-8'), p, '' if sg < 0 else ' {}'.format(sg))
                             for c, p, sg in child_parent_sgs]),
            tree_file)

    node_should_have_images(tax.root, 200, 
            op.join(overall_dataset._data_root, 'labels_with_few_images.yaml'))

    def gen_rows(label_type):
        for s in data_sources:
            for i, row in enumerate(s.gen_tsv_rows(tax.root, label_type)):
                if (i % 1000) == 0:
                    logging.info('gen-rows: {}-{}-{}'.format(s.name, label_type, i))
                yield row
    
    copy_rows = False
    if copy_rows:
        # write trainval.tsv
        for label_type in out_dataset:
            tsv_file = out_dataset[label_type].get_trainval_tsv()
            if op.isfile(tsv_file):
                continue
            tmp_tsv_file = 'tmp.' + op.basename(tsv_file)
            tmp_tsv_file = op.join(op.dirname(tsv_file), tmp_tsv_file)
            if not op.isfile(tsv_file) or True:
                tsv_writer(gen_rows(label_type), tmp_tsv_file)
            logging.info('shuffling {}'.format(tmp_tsv_file))
            rows = tsv_shuffle_reader(tmp_tsv_file)
            tsv_writer(rows, tsv_file)
            logging.info('remove the unshuffled: {}'.format(tmp_tsv_file))
            os.remove(tmp_tsv_file)
            os.remove(op.splitext(tmp_tsv_file)[0] + '.lineidx')

        # split into train and test
        for label_type in out_dataset:
            dataset = out_dataset[label_type]
            trainval_split(dataset, 50)
        
        for label_type in out_dataset:
            populate_dataset_details(out_dataset[label_type].name)
    else:
        # get the information of all train val
        train_vals = []
        ldtsi = []
        logging.info('collecting all candidate images')
        for label_type in out_dataset:
            for dataset in data_sources:
                split_idxes = dataset.select_tsv_rows(label_type)
                for rootlabel, split, idx in split_idxes:
                    ldtsi.append((rootlabel, dataset, label_type, split, idx))
        # split into train val
        num_test = 50
        logging.info('splitting the images into train and test')
        # group by label_type
        t_to_ldsi = list_to_dict(ldtsi, 2)
        train_ldtsi = [] 
        test_ldtsi = []
        for label_type in t_to_ldsi:
            ldsi= t_to_ldsi[label_type]
            l_to_dsi = list_to_dict(ldsi, 0)
            for rootlabel in l_to_dsi:
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

        logging.info('creating the train data')
        t_to_ldsi = list_to_dict(train_ldtsi, 2)
        train_ldtsik = []
        shuffle_idx = []
        for label_type in t_to_ldsi:
            ldsi = t_to_ldsi[label_type]
            d_to_lsi = list_to_dict(ldsi, 1)
            k = 0
            sources = []
            sources_label = []
            for dataset in d_to_lsi:
                lsi = d_to_lsi[dataset]
                s_li = list_to_dict(lsi, 1)
                for split in s_li:
                    li = s_li[split]
                    idx_to_l = list_to_dict(li, 1)
                    idx = idx_to_l.keys()
                    # link the data tsv
                    source = dataset.get_data(split)
                    out_split = 'train{}'.format(k)
                    train_ldtsik.extend([(l, dataset, label_type, split, i,
                        k) for l, i in li])
                    k = k + 1
                    dest = out_dataset[label_type].get_data(
                            out_split)
                    remove_depend_symlink = True
                    if remove_depend_symlink:
                        sources.append(source)
                    else:
                        logging.warn('not supported any longer since philly does not support softlink')
                        sources.append(dest)
                        if op.islink(dest):
                            os.remove(dest)
                        os.symlink(op.relpath(source, op.dirname(dest)), dest)
                        # link the lineidx
                        source = dataset.get_lineidx(split)
                        dest = out_dataset[label_type].get_lineidx(out_split)
                        if op.islink(dest):
                            os.remove(dest)
                        os.symlink(op.relpath(source, op.dirname(dest)), dest)
                        # create the label tsv
                    logging.info('converting labels: {}-{}'.format(
                        dataset.name, split))
                    converted_label = convert_label(dataset.get_data(split, 'label'),
                            idx, dataset._datasetlabel_to_rootlabel)
                    label_file = out_dataset[label_type].get_data(out_split, 'label')
                    tsv_writer(converted_label, label_file)
                    sources_label.append(label_file)
            write_to_file('\n'.join(sources),
                    out_dataset[label_type].get_data('trainX'))
            write_to_file('\n'.join(sources_label), 
                    out_dataset[label_type].get_data('trainX', 'label'))
        logging.info('duplicating or removing the train images')
        # for each label, let's duplicate the image or remove the image
        max_image = kwargs.get('max_image_per_label', 1000)
        min_image = kwargs.get('min_image_per_label', 200)
        label_to_dtsik = list_to_dict(train_ldtsik, 0)
        for label in label_to_dtsik:
            dtsik = label_to_dtsik[label]
            if len(dtsik) > max_image:
                # first remove the images with no bounding box
                num_remove = len(dtsik) - max_image
                type_to_dsik = list_to_dict(dtsik, 1)
                if 'no_bb' in type_to_dsik:
                    dsik = type_to_dsik['no_bb']
                    if num_remove >= len(dsik):
                        # remove all this images
                        del type_to_dsik['no_bb']
                        num_remove = num_remove - len(dsik)
                    else:
                        random.shuffle(dsik)
                        type_to_dsik['no_bb'] = dsik[: len(dsik) - num_remove]
                        num_remove = 0
                if num_remove > 0:
                    assert 'with_bb' in type_to_dsik
                    dsik = type_to_dsik['with_bb']
                    random.shuffle(dsik)
                    assert len(dsik) > num_remove
                    type_to_dsik['with_bb'] = dsik[: len(dsik) - num_remove]
                    num_remove = 0
                dtsik = dict_to_list(type_to_dsik, 1)
            elif len(dtsik) < min_image:
                num_duplicate = int(np.ceil(float(min_image) / len(dtsik)))
                logging.info('duplicate images for label of {}: {}->{}, {}'.format(
                    label, len(dtsik), min_image, num_duplicate))
                dtsik = num_duplicate * dtsik
            label_to_dtsik[label] = dtsik
        logging.info('# train instances before duplication: {}'.format(len(train_ldtsik)))
        train_ldtsik = dict_to_list(label_to_dtsik, 0)
        logging.info('# train instances after duplication: {}'.format(len(train_ldtsik)))

        logging.info('saving the shuffle file')
        type_to_ldsik = list_to_dict(train_ldtsik, 2)
        for label_type in type_to_ldsik:
            ldsik = type_to_ldsik[label_type]
            random.shuffle(ldsik)
            shuffle_str = '\n'.join(['{}\t{}'.format(k, i) for l, d, s, i, k in
                ldsik])
            write_to_file(shuffle_str,
                    out_dataset[label_type].get_shuffle_file('train'))

        logging.info('writing the test data')
        t_to_ldsi = list_to_dict(test_ldtsi, 2)
        for label_type in t_to_ldsi:
            def gen_test_rows():
                ldsi = t_to_ldsi[label_type]
                d_to_lsi = list_to_dict(ldsi, 1)
                for dataset in d_to_lsi:
                    lsi = d_to_lsi[dataset]
                    s_to_li = list_to_dict(lsi, 1)
                    for split in s_to_li:
                        li = s_to_li[split]
                        idx = list_to_dict(li, 1).keys()
                        tsv = TSVFile(dataset.get_data(split))
                        for i in idx:
                            row = tsv.seek(i)
                            rects = json.loads(row[1])
                            convert_one_label(rects, 
                                    dataset._datasetlabel_to_rootlabel)
                            assert len(rects) > 0
                            row[1] = json.dumps(rects)
                            row[0] = '{}_{}_{}'.format(dataset.name,
                                    split, row[0])
                            yield row
            tsv_writer(gen_test_rows(), 
                    out_dataset[label_type].get_test_tsv_file())

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
    return parser.parse_args()

if __name__ == '__main__':
    init_logging()
    args = parse_args()
    process_tsv_main(**vars(args))

