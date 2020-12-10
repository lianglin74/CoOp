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
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def __repr__(self):
        return 'DatasetPlusTransform(dataset={}, transform={})'.format(
            self.dataset, self.transform
        )

    def __len__(self):
        return len(self.dataset)

class TSVSplitProperty(Dataset):
    '''
    one instance of this class mean one tsv file or one composite tsv, it could
    be label tsv, or hw tsv, or image tsv
    '''
    def __init__(self, data, split, t=None, version=0, cache_policy=None):
        from qd.tsv_io import TSVDataset
        self.data = data
        self.split = split
        self.t = t
        self.version = version
        dataset = TSVDataset(data)
        if op.isfile(dataset.get_data(split, t, version)):
            self.tsv = TSVFile(dataset.get_data(split, t, version),
                    cache_policy)
        else:
            splitX = split + 'X'
            list_file = dataset.get_data(splitX, t, version=version)
            seq_file = dataset.get_shuffle_file(split)
            self.tsv = CompositeTSVFile(list_file, seq_file, cache_policy)

    def __getitem__(self, index):
        row = self.tsv[index]
        return row

    def __len__(self):
        return len(self.tsv)

    def num_rows(self):
        return len(self)

    def __iter__(self):
        self.curr_idx = 0
        return self

    def get_key(self, i):
        return self.tsv.seek_first_column(i)

    def __next__(self):
        if self.curr_idx < len(self):
            result = self[self.curr_idx]
            self.curr_idx += 1
            return result
        else:
            raise StopIteration

    def seek_first_column(self, idx):
        return self.tsv.seek_first_column(idx)

    def get_composite_source_idx(self):
        return self.tsv.get_composite_source_idx()

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

class CaptionIdxTSVDataset(object):
    def __init__(self, data, split, caption_version):
        from qd.data_layer.dataset import TSVSplitProperty
        self.data = data
        self.split = split
        self.caption_version = caption_version
        num_cap_tsv = TSVSplitProperty(data, split, 'num_caption',
                                       version=caption_version,
                                       )
        num_caps = [(key, int(n)) for key, n in num_cap_tsv]
        self.k_img_cap = [(k, idx_img, idx_cap) for idx_img, (k, n) in enumerate(num_caps) for idx_cap in range(n)]

    def __getitem__(self, idx):
        key, idx_img, idx_cap = self.k_img_cap[idx]
        data = {
            'idx': idx,
            'idx_img': idx_img,
            'idx_cap': idx_cap,
            'key': key,
            'dataset': self,
        }
        return data

    def __repr__(self):
        return 'CaptionIdxTSVDataset(data={}, split={}, caption_version={})'.format(
            self.data, self.split, self.caption_version
        )

    def __len__(self):
        return len(self.k_img_cap)

