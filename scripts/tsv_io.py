import logging
import re
import glob
import json
import random
from qd_common import default_data_path, ensure_directory
from qd_common import read_to_buffer, load_list_file
from qd_common import write_to_yaml_file, load_from_yaml_file
import os
import os.path as op
from ete2 import Tree
from qd_common import generate_lineidx
from qd_common import parse_test_data
from qd_common import img_from_base64
import numpy as np
import yaml

class TSVFile(object):
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx' 
        self._fp = None
        self._lineidx = None
    
    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx) 

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]
    
    def _ensure_lineidx_loaded(self):
        if not op.isfile(self.lineidx) and not op.islink(self.lineidx):
            generate_lineidx(self.tsv_file, self.lineidx)
        if self._lineidx is None:
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')


class TSVDataset(object):
    def __init__(self, name):
        self.name = name
        proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)));
        result = {}
        data_root = os.path.join(proj_root, 'data', name)
        self._data_root = op.relpath(data_root)
    
    def load_labelmap(self):
        return load_list_file(self.get_labelmap_file())

    def get_tree_file(self):
        return op.join(self._data_root, 'tree.txt')

    def get_labelmap_file(self):
        return op.join(self._data_root, 'labelmap.txt')

    def get_train_shuffle_file(self):
        return self.get_shuffle_file('train') 

    def get_shuffle_file(self, split_name):
        return op.join(self._data_root, '{}.shuffle.txt'.format(split_name))

    def get_labelmap_of_noffset_file(self):
        return op.join(self._data_root, 'noffsets.label.txt')

    def load_key_to_idx(self, split):
        result = {}
        for i, row in enumerate(tsv_reader(self.get_data(split, 'label'))):
            key = row[0]
            assert key not in result
            result[key] = i
        return result

    def load_keys(self, split):
        result = []
        for row in tsv_reader(self.get_data(split, 'label')):
            result.append(row[0])
        return result

    def dynamic_update(self, dataset_ops):
        '''
        sometimes, we update the dataset, and here, we should update the file
        path
        '''
        if len(dataset_ops) >= 1 and dataset_ops[0]['op'] == 'sample':
            self._data_root = op.join('./output/data/',
                    '{}_{}_{}'.format(self.name,
                        dataset_ops[0]['sample_label'],
                        dataset_ops[0]['sample_image']))
        elif len(dataset_ops) >= 1 and dataset_ops[0]['op'] == 'mask_background':
            target_folder = op.join('./output/data',
                    '{}_{}_{}'.format(self.name,
                        '.'.join(map(str, dataset_ops[0]['old_label_idx'])),
                        dataset_ops[0]['new_label_idx']))
            self._data_root = target_folder 

    def get_test_tsv_file(self, t=None):
        return self.get_data('test', t)

    def get_test_tsv_lineidx_file(self):
        return op.join(self._data_root, 'test.lineidx') 

    def get_train_tsvs(self, t=None):
        if op.isfile(self.get_data('train', t)):
            return [self.get_data('train', t)]
        trainx_file = op.join(self._data_root, 'trainX.tsv')
        if not op.isfile(trainx_file):
            return []
        train_x = load_list_file(trainx_file)
        if t is None:
            return train_x
        elif t =='label':
            if op.isfile(self.get_data('trainX', 'label')):
                return load_list_file(self.get_data('trainX', 'label'))
            else:
                files = [op.splitext(f)[0] + '.label.tsv' for f in train_x]
                return files

    def get_train_tsv(self, t=None):
        return self.get_data('train', t)

    def get_lineidx(self, split_name):
        return op.join(self._data_root, '{}.lineidx'.format(split_name))

    def get_data(self, split_name, t=None):
        if t is None:
            return op.join(self._data_root, '{}.tsv'.format(split_name)) 
        else:
            return op.join(self._data_root, '{}.{}.tsv'.format(split_name,
                t))

    def get_num_train_image(self):
        if op.isfile(self.get_data('trainX')):
            if op.isfile(self.get_shuffle_file('train')):
                return len(load_list_file(self.get_shuffle_file('train')))
            else:
                return 0
        else:
            return len(load_list_file(op.join(self._data_root, 'train.lineidx')))

    def get_trainval_tsv(self, t=None):
        return self.get_data('trainval', t)

    def get_noffsets_file(self):
        return op.join(self._data_root, 'noffsets.txt')

    def load_noffsets(self):
        logging.info('deprecated: pls generate it on the fly')
        return load_list_file(self.get_noffsets_file()) 

    def load_inverted_label(self, split):
        fname = self.get_data(split, 'inverted.label')
        if not op.isfile(fname):
            return {}
        else:
            rows = tsv_reader(fname)
            result = {}
            for row in rows:
                assert row[0] not in result
                assert len(row) == 2
                ss = row[1].split(' ')
                if len(ss) == 1 and ss[0] == '':
                    result[row[0]] = []
                else:
                    result[row[0]] = map(int, ss)
            return result 

    def iter_data(self, split, t=None):
        if split == 'train' and op.isfile(self.get_data('trainX')):
            train_files = load_list_file(self.get_data('trainX', t))
            train_tsvs = [TSVFile(f) for f in train_files]
            train_label_files = load_list_file(self.get_data('trainX',
                'label'))
            train_label_tsvs = [TSVFile(f) for f in train_label_files]
            shuffle_file = self.get_shuffle_file('train')
            shuffle_tsv_rows = tsv_reader(shuffle_file)
            for idx_source, idx_row in shuffle_tsv_rows:
                idx_source, idx_row = int(idx_source), int(idx_row)
                data_row = train_tsvs[idx_source].seek(idx_row)
                label_row = train_label_tsvs[idx_source].seek(idx_row)
                assert label_row[0] == data_row[0]
                yield label_row[0], label_row[1], data_row[-1]
        else:
            for row in tsv_reader(self.get_data(split)):
                yield row

def tsv_writer(values, tsv_file_name):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    idx = 0
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    with open(tsv_file_name_tmp, 'w') as fp, open(tsv_lineidx_file_tmp, 'w') as fpidx:
        assert values is not None
        for value in values:
            assert value
            v = '{0}\n'.format('\t'.join(value))
            fp.write(v)
            fpidx.write(str(idx) + '\n')
            idx = idx + len(v)
    os.rename(tsv_file_name_tmp, tsv_file_name)
    os.rename(tsv_lineidx_file_tmp, tsv_lineidx_file)

def tsv_reader(tsv_file_name):
    with open(tsv_file_name, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split('\t')]

def get_meta_file(tsv_file):
    return op.splitext(tsv_file)[0] + '.meta.yaml'

def extract_label(full_tsv, label_tsv):
    if op.isfile(label_tsv):
        logging.info('label file exists and will skip to generate: {}'.format(
            label_tsv))
        return
    if not op.isfile(full_tsv):
        logging.info('the file of {} does not exist'.format(full_tsv))
        return
    rows = tsv_reader(full_tsv)
    def gen_rows():
        for i, row in enumerate(rows):
            if (i % 1000) == 0:
                logging.info('extract_label: {}-{}'.format(full_tsv, i))
            del row[2]
            assert len(row) == 2
            assert type(row[0]) == str
            assert type(row[1]) == str
            yield row
    tsv_writer(gen_rows(), label_tsv)

def create_inverted_tsv(label_tsv, inverted_label_file, label_map_file):
    '''
    save the results based on the label_map in label_map_file. The benefit is
    to seek the row given a label
    '''
    if not op.isfile(label_tsv):
        logging.info('the label file does not exist: {}'.format(label_tsv))
        return 
    label_map = load_list_file(label_map_file)
    rows = tsv_reader(label_tsv)
    inverted = {}
    for i, row in enumerate(rows):
        labels = json.loads(row[1])
        if type(labels) is list:
            # detection dataset
            curr_unique_labels = set([l['class'] for l in labels])
        else:
            assert type(labels) is int
            curr_unique_labels = [label_map[labels]]
        for l in curr_unique_labels:
            assert type(l) == str or type(l) == unicode 
            if l not in inverted:
                inverted[l] = [i]
            else:
                inverted[l].append(i)
    def gen_rows():
        for label in inverted:
            assert label in label_map
        for label in label_map:
            i = inverted[label] if label in inverted else []
            yield label, ' '.join(map(str, i))
    tsv_writer(gen_rows(), inverted_label_file)

def tsv_shuffle_reader(tsv_file):
    logging.warn('deprecated: using TSVFile to randomly seek')
    lineidx_file = op.splitext(tsv_file)[0] + '.lineidx'
    lineidx = load_list_file(lineidx_file)
    random.shuffle(lineidx)
    with open(tsv_file, 'r') as fp:
        for l in lineidx:
            fp.seek(int(float(l)))
            yield [x.strip() for x in fp.readline().split('\t')]
    
def load_labelmap(data):
    dataset = TSVDataset(data)
    return dataset.load_labelmap()

def get_all_data_info2(name=None):
    if name is None:
        return sorted(os.listdir('./data'))
    else:
        dataset = TSVDataset(name)
        if not op.isfile(dataset.get_labelmap_file()):
            return []
        labels = dataset.load_labelmap()
        valid_splits = []
        if len(dataset.get_train_tsvs()) > 0:
            valid_splits.append('train')
        for split in ['trainval', 'test']:
            if not op.isfile(dataset.get_data(split)):
                continue
            valid_splits.append(split)
        name_splits_labels = [(name, valid_splits, labels)]
    return name_splits_labels

def get_all_data_info():
    names = os.listdir('./data')
    name_splits_labels = []
    names.sort(key=lambda n: n.lower())
    for name in names:
        dataset = TSVDataset(name)
        if not op.isfile(dataset.get_labelmap_file()):
            continue
        labels = dataset.load_labelmap()
        valid_splits = []
        if len(dataset.get_train_tsvs()) > 0:
            valid_splits.append('train')
        for split in ['trainval', 'test']:
            if not op.isfile(dataset.get_data(split)):
                continue
            valid_splits.append(split)
        name_splits_labels.append((name, valid_splits, labels))
    return name_splits_labels

def load_labels(file_name):
    rows = tsv_reader(file_name)
    labels = {}
    label_to_idx = {}
    for i, row in enumerate(rows):
        key = row[0]
        rects = json.loads(row[1])
        #assert key not in labels, '{}-{}'.format(file_name, key)
        labels[key] = rects
        label_to_idx[key] = i
    return labels, label_to_idx

