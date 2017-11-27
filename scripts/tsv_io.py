import logging
import json
import os
import os.path as op
from qd_common import read_to_buffer, load_list_file
from qd_common import ensure_directory

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

    def get_test_tsv_file(self):
        return op.join(self._data_root, 'test.tsv')

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
            files = [op.splitext(f)[0] + '.label.tsv' for f in train_x]
            return files

    def get_train_tsv(self):
        return op.join(self._data_root, 'train.tsv') 

    def get_lineidx(self, split_name):
        return op.join(self._data_root, '{}.lineidx'.format(split_name))

    def get_data(self, split_name, t=None):
        label_tsv = op.join(self._data_root, '{}.label.tsv'.format(split_name))
        full_tsv = op.join(self._data_root, '{}.tsv'.format(split_name)) 
        inverted_label_file = op.join(self._data_root,
                '{}.inverted.label.tsv'.format(split_name))
        if t == 'label':
            return label_tsv
        elif t == 'inverted.label':
            return inverted_label_file
        else:
            assert t == None
            return full_tsv

    def get_num_train_image(self):
        return len(load_list_file(op.join(self._data_root, 'train.lineidx')))

    def get_trainval_tsv(self):
        return op.join(self._data_root, 'trainval.tsv')

    def get_noffsets_file(self):
        return op.join(self._data_root, 'noffsets.txt')

    def load_noffsets(self):
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
                result[row[0]] = map(int, row[1].split(' '))
            return result 

def tsv_writer(values, tsv_file_name):
    ensure_directory(os.path.dirname(tsv_file_name))
    tsv_lineidx_file = os.path.splitext(tsv_file_name)[0] + '.lineidx'
    idx = 0
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    tsv_lineidx_file_tmp = tsv_lineidx_file + '.tmp'
    with open(tsv_file_name_tmp, 'w') as fp, open(tsv_lineidx_file_tmp, 'w') as fpidx:
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
            yield row
    tsv_writer(gen_rows(), label_tsv)

def create_inverted_tsv(label_tsv, inverted_label_file):
    if not op.isfile(label_tsv):
        logging.info('the label file does not exist: {}'.format(label_tsv))
        return 
    rows = tsv_reader(label_tsv)
    inverted = {}
    for i, row in enumerate(rows):
        labels = json.loads(row[1])
        for l in set([l['class'] for l in labels]):
            if l not in inverted:
                inverted[l] = [i]
            else:
                inverted[l].append(i)
    def gen_rows():
        for label in inverted:
            yield label, ' '.join(map(str, inverted[label]))
    tsv_writer(gen_rows(), inverted_label_file)

