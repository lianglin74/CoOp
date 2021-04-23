import logging
import time
from qd.qd_common import qd_tqdm as tqdm
from torch.utils.data import Dataset
import os.path as op
from qd.tsv_io import TSVFile, CompositeTSVFile
from qd.tsv_io import TSVDataset
from qd.tsv_io import TSVSplitProperty
from qd.logger import MetricLogger
from qd.logger import SmoothedValue
from torchvision.transforms import transforms


class DatasetPlusTransform(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.log_step = 1000
        self.meters = MetricLogger(delimiter="  ", meter_creator=lambda:
                              SmoothedValue(self.log_step))
        self.next_iter = 1

    def get_keys(self):
        return self.dataset.get_keys()

    def get_composite_source_idx(self):
        return self.dataset.get_composite_source_idx()

    def get_composite_source_files(self):
        return self.dataset.get_composite_source_files()

    def __getitem__(self, idx_info):
        start = time.time()
        data_dict = self.dataset[idx_info]
        end = time.time()
        time_info = {'dataset': end - start}
        i = {'i': 0}
        def unwrap_compose_transform(t, data_dict):
            if not isinstance(t, transforms.Compose):
                start = time.time()
                data_dict = t(data_dict)
                time_info[str(i['i']) + '_' + t.__class__.__name__] = time.time() - start
                i['i'] += 1
                return data_dict
            else:
                for x in t.transforms:
                    data_dict = unwrap_compose_transform(x, data_dict)
                return data_dict
        if self.transform is not None:
            data_dict = unwrap_compose_transform(
                self.transform, data_dict)
        self.meters.update(**time_info)
        #if (self.next_iter % self.log_step) == 0:
            #logging.info(str(self.meters))
        self.next_iter += 1
        return data_dict

    def __repr__(self):
        return 'DatasetPlusTransform(dataset={}, transform={})'.format(
            self.dataset, self.transform
        )

    def __len__(self):
        return len(self.dataset)

class IODataset(object):
    def __init__(self, data, split, version):
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

class ImageIdxTSVDataset(object):
    # this dataset does not load anything, but scan the image one by one.
    # Useful in testing for v-l tasks
    def __init__(self, data, split):
        self.data = data
        self.split = split
        self.total_num = len(TSVSplitProperty(data, split))

    def get_keys(self):
        return self.get_first_columns()

    def get_first_columns(self):
        dataset = TSVDataset(self.data)
        if dataset.has(self.split, 'hw'):
            return [key for key, _ in dataset.iter_data(
                self.split, 'hw')]
        else:
            tsv = TSVSplitProperty(self.data, self.split)
            return [tsv.seek_first_column(i) for i in range(self.total_num)]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            data = {
                'idx': idx,
                'idx_img': idx,
                'dataset': self,
            }
        else:
            assert isinstance(idx, dict)
            data = idx
            assert 'idx_img' not in data
            data['idx_img'] = data['idx']
            assert 'dataset' not in data
            data['dataset'] = self
        return data

    def __len__(self):
        return self.total_num

class CaptionIdxTSVDataset(object):
    def __init__(self, data, split, caption_version):
        self.data = data
        self.split = split
        self.caption_version = caption_version
        #num_cap_tsv = TSVSplitProperty(data, split, 'num_caption',
                                       #version=caption_version,
                                       #)
        #num_caps = [(key, int(n)) for key, n in num_cap_tsv]
        #self.k_img_cap = [(k, idx_img, idx_cap) for idx_img, (k, n) in
                          #tqdm(enumerate(num_caps)) for idx_cap in range(n)]

        self.k_img_cap = TSVSplitProperty(
            data, split, 'key_idximage_idxcaption',
            version=caption_version,
        )

    def get_composite_source_idx(self):
        dataset = TSVDataset(self.data)
        tsv = dataset.get_data(self.split, 'caption', self.caption_version)
        if op.isfile(tsv):
            return [0] * len(self.k_img_cap)
        x_tsv = dataset.get_data(self.split + 'X', 'caption',
                                 self.caption_version)
        assert op.isfile(x_tsv)
        from qd.tsv_io import tsv_reader
        all_idx_source = [int(idx_source) for idx_source, _ in
                tsv_reader(dataset.get_shuffle_file(self.split))]
        return [all_idx_source[int(idx_img)] for _, idx_img, _ in
                tqdm(self.k_img_cap)]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            key, idx_img, idx_cap = self.k_img_cap[idx]
            data = {
                'idx': idx,
                'idx_img': int(idx_img),
                'idx_cap': int(idx_cap),
                'key': key,
                'dataset': self,
            }
        else:
            data = idx
            key, idx_img, idx_cap = self.k_img_cap[data['idx']]
            extra_data = {
                'idx_img': int(idx_img),
                'idx_cap': int(idx_cap),
                'key': key,
                'dataset': self,
            }
            for k in extra_data:
                assert k not in data
            data.update(extra_data)
            if 'future_idx' in data:
                # this is for pre-fetching the data by LoadImage
                key, idx_img, idx_cap = self.k_img_cap[data['future_idx']]
                extra_data = {
                    'future_idx_img': int(idx_img),
                    'future_idx_cap': int(idx_cap),
                }
        return data

    def get_keys(self):
        # this function is used in prediction for re-ordering the prediction
        # files in each rank. In test phase, if we use this dataset, the
        # prediction should return the index as the key
        l = len(self)
        return list(map(str, range(l)))

    def __repr__(self):
        return 'CaptionIdxTSVDataset(data={}, split={}, caption_version={})'.format(
            self.data, self.split, self.caption_version
        )

    def __len__(self):
        return len(self.k_img_cap)

