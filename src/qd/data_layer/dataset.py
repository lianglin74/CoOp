from torch.utils.data import Dataset
import os.path as op
from qd.tsv_io import TSVFile, CompositeTSVFile


class DatasetPlusTransform(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def get_keys(self):
        return self.dataset.get_keys()

    def __getitem__(self, idx_info):
        data_dict = self.dataset[idx_info]
        data_dict = self.transform(data_dict)
        return data_dict

    def __len__(self):
        return len(self.dataset)

class TSVSplitProperty(Dataset):
    '''
    one instance of this class mean one tsv file or one composite tsv, it could
    be label tsv, or hw tsv, or image tsv
    '''
    def __init__(self, data, split, t=None, version=0, cache_policy=None):
        from qd.tsv_io import TSVDataset
        dataset = TSVDataset(data)
        if op.isfile(dataset.get_data(split, t, version)):
            self.tsv = TSVFile(dataset.get_data(split, t, version),
                    cache_policy)
        else:
            splitX = split + 'X'
            list_file = dataset.get_data(splitX, t)
            seq_file = dataset.get_shuffle_file(split)
            self.tsv = CompositeTSVFile(list_file, seq_file, cache_policy)
            assert version in [0, None]

    def __getitem__(self, index):
        row = self.tsv[index]
        return row

    def __len__(self):
        return len(self.tsv)

    def __iter__(self):
        self.curr_idx = 0
        return self

    def __next__(self):
        if self.curr_idx < len(self):
            result = self[self.curr_idx]
            self.curr_idx += 1
            return result
        else:
            raise StopIteration

class IODataset(object):
    def __init__(self, data, split, version):
        from qd.qd_pytorch import TSVSplitProperty
        self.image_tsv = TSVSplitProperty(data, split, t=None)
        self.label_tsv = TSVSplitProperty(data, split, t='label',
                                          version=version)

    def get_keys(self):
        return [key for key, _ in self.label_tsv]

    def __len__(self):
        return len(self.label_tsv)

    def __getitem__(self, idx_info):
        import copy
        if isinstance(idx_info, dict):
            result = copy.deepcopy(idx_info)
        else:
            result = {'idx': idx_info}
        idx = result['idx']
        image_row = self.image_tsv[idx]
        result['image'] = image_row[-1]
        label_row = self.label_tsv[idx]
        result['label'] = label_row[-1]
        result['key'] = label_row[0]
        result['io_dataset'] = self
        return result

