import os
import os.path as op
from qd_common import read_to_buffer, load_list_file

class TSVDataset(object):
    def __init__(self, name):
        self.name = name
        proj_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)));
        result = {}
        data_root = os.path.join(proj_root, 'data', name)
        self._data_root = data_root
    
    def load_labelmap(self):
        return load_list_file(self.get_labelmap_file())

    def get_tree_file(self):
        return op.join(self._data_root, 'tree.txt')

    def get_labelmap_file(self):
        return op.join(self._data_root, 'labelmap.txt')

    def get_labelmap_of_noffset_file(self):
        return op.join(self._data_root, 'noffsets.label.txt')

    def get_test_tsv_file(self):
        return op.join(self._data_root, 'test.tsv')

    def get_test_tsv_lineidx_file(self):
        return op.join(self._data_root, 'test.lineidx') 

    def get_train_tsv(self):
        return op.join(self._data_root, 'train.tsv') 

    def get_num_train_image(self):
        return len(load_list_file(op.join(self._data_root, 'train.lineidx')))

    def get_train_shuffle_file(self):
        return op.join(self._data_root, 'train_shuffle.txt')

    def get_trainval_tsv(self):
        return op.join(self._data_root, 'trainval.tsv')

    def get_noffsets_file(self):
        return op.join(self._data_root, 'noffsets.txt')

    def load_noffsets(self):
        return load_list_file(self.get_noffsets_file()) 
