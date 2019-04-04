import os
import torch
from torch.utils.data import Dataset

from qd.tsv_io import TSVFile, tsv_reader, tsv_writer
from qd.qd_common import img_from_base64, generate_lineidx, load_from_yaml_file, FileProgressingbar, int_rect, hash_sha1

import base64
from collections import OrderedDict, defaultdict
import json
import math
import multiprocessing as mp
import numpy as np
import pathos.multiprocessing
import time
import yaml


class TSVDataset(Dataset):
    """ TSV dataset for ImageNet 1K training
    """
    def __init__(self, tsv_file, transform=None):
        self.tsv = TSVFile(tsv_file)
        self.transform = transform

    def __getitem__(self, index):
        row = self.tsv.seek(index)
        img = img_from_base64(row[-1])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        idx = int(row[1])
        label = torch.from_numpy(np.array(idx, dtype=np.int))
        return img, label

    def __len__(self):
        return self.tsv.num_rows()

    def label_dim(self):
        return 1000

    def is_multi_label(self):
        return False

    def get_labelmap(self):
        return [str(i) for i in range(1000)]


class TSVDatasetPlus(TSVDataset):
    """ TSV dataset plus supporting separate label file and shuffle file
    This dataset class supports the use of:
        1. an optional separate label file - as labels often need to be changed over time.
        2. an optional shuffle file - a list of line numbers to specify a subset of images in the tsv_file.
        3. an optional labelmap file - to map a string label to a class id on the fly
    """
    def __init__(self, tsv_file, label_file=None, shuf_file=None, labelmap=None,
                 col_label=0, multi_label=False,
                 transform=None):
        self.tsv = TSVFile(tsv_file)
        self.tsv_label = None if label_file is None else TSVFile(label_file)
        self.shuf_list = self._load_shuffle_file(shuf_file)
        self.labelmap = self._load_labelmap(labelmap)

        self.transform = transform
        self.col_image = self._guess_col_image()
        self.col_label = col_label
        self.multi_label = multi_label

    def __getitem__(self, index):
        line_no = self._line_no(index)
        cols = self.tsv.seek(line_no)
        img = img_from_base64(cols[self.col_image])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        lbl = cols[self.col_label] if self.tsv_label is None else self.tsv_label.seek(line_no)[self.col_label]
        label = self._transform_label(lbl)

        return img, label

    def __len__(self):
        return self.tsv.num_rows() if self.shuf_list is None else len(self.shuf_list)

    def label_dim(self):
        return 1000 if self.labelmap is None else len(self.labelmap)

    def is_multi_label(self):
        return self.multi_label

    def get_labelmap(self):
        return self.labelmap.keys()

    def _line_no(self, idx):
        return idx if self.shuf_list is None else self.shuf_list[idx]

    def _guess_col_image(self):
        # peek one line to find the longest column as col_image
        with open(self.tsv.tsv_file, 'r') as f:
            cols = [s.strip() for s in f.readline().split('\t')]
        return max(enumerate(cols), key=lambda x: len(x[1]))[0]

    def _load_labelmap(self, labelmap):
        label_dict = None
        if labelmap is not None:
            label_dict = OrderedDict()
            with open(labelmap, 'r') as fp:
                for line in fp:
                    label = line.strip().split('\t')[0]
                    if label in label_dict:
                        raise ValueError("Duplicate label " + label + " in labelmap.")
                    else:
                        label_dict[label] = len(label_dict)
        return label_dict

    def _load_shuffle_file(self, shuf_file):
        shuf_list = None
        if shuf_file is not None:
            with open(shuf_file, 'r') as fp:
                bar = FileProgressingbar(fp, 'Loading shuffle file {0}: '.format(shuf_file))
                shuf_list = []
                for i in fp:
                    shuf_list.append(int(i.strip()))
                    bar.update()
                print
        return shuf_list

    def _transform_label(self, label):
        if self.multi_label:
            assert self.labelmap is not None, 'Expect labelmap for multi labels'
            _label = np.zeros(len(self.labelmap))
            labels = label.replace(',', ';').split(';')
            for l in labels:
                if l in self.labelmap:  # if a label is unknown, it will be skipped
                    _label[self.labelmap[l]] = 1
            return torch.from_numpy(np.array(_label, dtype=np.float32))
        else:
            _label = int(label) if self.labelmap is None else self.labelmap[label]
            return torch.from_numpy(np.array(_label, dtype=np.int))

class TSVDatasetPlusYaml(TSVDatasetPlus):
    """ TSVDatasetPlus taking a Yaml file for easy function call
    """
    def __init__(self, yaml_file, session_name='', transform=None):
        cfg = load_from_yaml_file(yaml_file)
        root = os.path.dirname(yaml_file)

        if session_name:
            cfg = cfg.get(session_name, None)
            assert cfg is not None, 'Invalid session name in Yaml. Please check.'

        tsv_file = os.path.join(root, cfg['tsv'])

        label_file = cfg.get('label', None)
        if label_file is not None:
            label_file = os.path.join(root, label_file)

        shuf_file = cfg.get('shuffle', None)
        if shuf_file is not None:
            shuf_file = os.path.join(root, shuf_file)

        labelmap = cfg.get('labelmap', None)
        if labelmap is not None:
            labelmap = os.path.join(root, labelmap)

        multi_label = cfg.get('multi_label', False)
        col_label = cfg['col_label']

        super(TSVDatasetPlusYaml, self).__init__(
            tsv_file, label_file, shuf_file, labelmap,
            col_label, multi_label,
            transform)


class TSVDatasetWithoutLabel(TSVDatasetPlus):
    """ TSV dataset with no labels. The simplest format for testing.
    """
    def __init__(self, data_file, session_name='', transform=None):
        """ data_file could be just a tsv file, or a yaml file including tsv & shuffle files
        """
        if data_file.endswith('.tsv'):
            tsv_file = data_file
            shuf_file = None
        else:
            cfg = load_from_yaml_file(data_file)
            root = os.path.dirname(data_file)

            if session_name:
                cfg = cfg.get(session_name, None)
                assert cfg is not None, 'Invalid session name in Yaml. Please check.'

            tsv_file = os.path.join(root, cfg['tsv'])

            shuf_file = cfg.get('shuffle', None)
            if shuf_file is not None:
                shuf_file = os.path.join(root, shuf_file)

        self.tsv = TSVFile(tsv_file)
        self.shuf_list = self._load_shuffle_file(shuf_file)

        self.transform = transform
        self.col_image = self._guess_col_image()

    def __getitem__(self, index):
        line_no = self._line_no(index)
        cols = self.tsv.seek(line_no)
        img = img_from_base64(cols[self.col_image])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        cols.pop(self.col_image)
        return img, cols


class CropClassTSVDataset(Dataset):
    def __init__(self, tsvfile, labelmap, labelfile=None, label_filter_fn=None,
                 transform=None, logger=None, for_test=False, reorder_label=False,
                 overwrite_cache=False, enlarge_bbox=1.0):
        """ TSV dataset with cropped images from bboxes labels
        Params:
            tsvfile: image tsv file, columns are key, bboxes, b64_image_string
            labelmap: file of all categories
            labelfile: label tsv file, columns are key, bboxes
            label_filter_fn: callable, filter the bbox list
        """
        self.min_pixels = 3
        self.tsv = TSVFile(tsvfile)
        self.tsvfile = tsvfile
        self.labelfile = labelfile
        self.transform = transform
        self.label_to_idx = {}
        with open(labelmap, 'r') as fp:
            for i, line in enumerate(fp):
                l = line.rstrip('\n')
                assert(l not in self.label_to_idx)
                self.label_to_idx[l] = i
        self.label_filter_fn = label_filter_fn
        self.img_col = 2
        self.label_col = 1
        self.key_col = 0
        self.logger = logger
        self._for_test = for_test
        self._reorder_label = reorder_label
        self._overwrite_cache = overwrite_cache
        self._enlarge_bbox = enlarge_bbox

        _cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        self._bbox_idx_file = os.path.join(_cache_dir, "{}.tsv".format(hash_sha1((tsvfile, labelfile if labelfile else "", str(for_test)))))
        try:
            _class_instance_idx = self._generate_class_instance_index_parallel()
            tsv_writer(_class_instance_idx, self._bbox_idx_file)
            self._bbox_idx_tsv = TSVFile(self._bbox_idx_file)
        except Exception as e:
            if os.path.isfile(self._bbox_idx_file):
                os.remove(self._bbox_idx_file)
            raise e

    def label_dim(self):
        return len(self.label_to_idx)

    def is_multi_label(self):
        return False

    def get_labelmap(self):
        return self.label_to_idx.keys()

    def _read_into_buffer(self, fpath, sep='\t'):
        ret = []
        with open(fpath, 'r') as fp:
            for line in fp:
                ret.append(line.strip().split(sep))
        return ret

    def _generate_class_instance_index_old(self):
        """ DEPRECATED: use _generate_class_instance_index_parallel
        """
        if os.path.isfile(self._cache) and not self._overwrite_cache:
            return self._read_into_buffer(self._cache)

        label_dict = None
        if self.labelfile:
            if self._reorder_label:
                label_dict = {}
                for row in tsv_reader(self.labelfile):
                    label_dict[row[self.key_col]] = json.loads(row[self.label_col])
            else:
                label_tsv = TSVFile(self.labelfile)

        # rewrite cache index
        with open(self._cache, 'w') as fp:
            for img_idx in range(self.tsv.num_rows()):
                row = self.tsv.seek(img_idx)
                if self.labelfile:
                    if label_dict:
                        bboxes = label_dict[row[self.key_col]]
                    else:
                        label_row = label_tsv.seek(img_idx)
                        assert(row[self.key_col] == label_row[self.key_col])
                        bboxes = json.loads(label_row[self.label_col])
                else:
                    bboxes = json.loads(row[self.label_col])
                img = img_from_base64(row[self.img_col])
                height, width, _ = img.shape
                for bbox in bboxes:
                    left, top, right, bot = self._int_rect(bbox["rect"], self._enlarge_bbox)
                    left = np.clip(left, 0, width)
                    right = np.clip(right, 0, width)
                    top = np.clip(top, 0, height)
                    bot = np.clip(bot, 0, height)
                    # ignore invalid bbox
                    if bot <= top or right <= left:
                        if self.logger:
                            self.logger.info("skip invalid bbox in {}".format(row[0]))
                        continue
                    info = [img_idx, left, top, right, bot]
                    if self._for_test:
                        info.append(json.dumps(bbox))
                    else:
                        # label only exists in training data
                        info.append(self.label_to_idx[bbox["class"]])
                    fp.write('\t'.join([str(c) for c in info]))
                    fp.write('\n')

        return self._read_into_buffer(self._cache)

    def _generate_class_instance_index_parallel(self):
        """ For training: (img_idx, rect, label_idx)
            For testing: (img_idx, rect, original_bbox)
            img_idx: line idx of the image tsv file
            rect: left, top, right, bot
        """
        return gen_index(self.tsvfile, self.labelfile, self.label_to_idx, self._for_test,
                self._enlarge_bbox, self.key_col, self.label_col, self.img_col, self.logger,
                self.min_pixels)

    def __getitem__(self, index):
        # info = self._class_instance_idx[index]
        info = self._bbox_idx_tsv.seek(index)
        img_idx, left, top, right, bot = (int(info[i]) for i in range(5))
        row = self.tsv.seek(img_idx)
        img = img_from_base64(row[self.img_col])
        cropped_img = img[top:bot, left:right]
        if self.transform is not None:
            cropped_img = self.transform(cropped_img)

        if self._for_test:
            return cropped_img, (row[self.key_col], info[5])
        else:
            # NOTE: currenly only support single label
            label_idx = info[5]
            label = torch.from_numpy(np.array(label_idx, dtype=np.int))
            return cropped_img, label

    def __len__(self):
        return self._bbox_idx_tsv.num_rows()


class CropClassTSVDatasetYaml(CropClassTSVDataset):
    """ CropClassTSVDataset taking a Yaml file for easy function call
    """
    def __init__(self, yaml_file, session_name='', transform=None, logger=None):
        cfg = load_from_yaml_file(yaml_file)

        if session_name:
            cfg = cfg[session_name]

        tsv_file = cfg['tsv']
        label_file = cfg.get('label', None)
        shuf_file = cfg.get('shuffle', None)
        labelmap = cfg.get('labelmap', None)

        multi_label = cfg.get('multi_label', False)
        for_test = True if session_name=="test" else False

        super(CropClassTSVDatasetYaml, self).__init__(
            tsv_file, labelmap, label_file,
            for_test=for_test, reorder_label=False, transform=transform, logger=logger)

class CropClassTSVDatasetYamlList():
    def __init__(self, yaml_lst_file, session_name='', transform=None, logger=None):
        self.yaml_files = self.load_yaml_list(yaml_lst_file)
        self.datasets = [CropClassTSVDatasetYaml(yaml_file, session_name=session_name,
                transform=transform, logger=logger) for yaml_file in self.yaml_files]
        self.dataset_lengths = [len(d) for d in self.datasets]
        self.length = sum(self.dataset_lengths)

        # check if labelmap match
        self.label_to_idx = self.datasets[0].label_to_idx
        for i in range(1, len(self.datasets)):
            tmp = self.datasets[i].label_to_idx
            assert(len(tmp) == len(self.label_to_idx))
            for k in self.label_to_idx:
                assert(self.label_to_idx[k] == tmp[k])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cum_length = 0
        dataset_idx = 0
        for _, length in enumerate(self.dataset_lengths):
            if cum_length + length > index:
                break
            cum_length += length
            dataset_idx += 1
        assert(dataset_idx < len(self.datasets))
        return self.datasets[dataset_idx][index - cum_length]

    def label_dim(self):
        return len(self.label_to_idx)

    def is_multi_label(self):
        return False

    def get_labelmap(self):
        return self.label_to_idx.keys()

    def load_yaml_list(self, yaml_lst_file):
        yaml_list = []
        for parts in tsv_reader(yaml_lst_file):
            f = parts[0]
            assert(os.path.isfile(f))
            yaml_list.append(f)
        return yaml_list


def gen_index(imgfile, labelfile, label_to_idx, for_test,
                enlarge_bbox_for_test, key_col, label_col, img_col,
                logger, min_pixels):
    enlarge_bbox = enlarge_bbox_for_test if for_test else 1.0
    all_args = []
    num_worker = mp.cpu_count()
    num_tasks = num_worker * 3
    imgtsv = TSVFile(imgfile)
    num_images = imgtsv.num_rows()
    num_image_per_worker = (num_images + num_tasks - 1) // num_tasks
    assert num_image_per_worker > 0
    for i in range(num_tasks):
        curr_idx_start = i * num_image_per_worker
        if curr_idx_start >= num_images:
            break
        curr_idx_end = curr_idx_start + num_image_per_worker
        curr_idx_end = min(curr_idx_end, num_images)
        if curr_idx_end > curr_idx_start:
            all_args.append((curr_idx_start, curr_idx_end))

    def _gen_index_helper(args):
        start, end = args[0], args[1]
        ret = []
        img_tsv = TSVFile(imgfile)
        if labelfile is not None:
            label_tsv = TSVFile(labelfile)
        else:
            label_tsv = None
        for idx in range(start, end):
            img_row = img_tsv.seek(idx)
            if label_tsv:
                label_row = label_tsv.seek(idx)
                if img_row[key_col] != label_row[key_col]:
                    if logger:
                        logger.info("image key do not match in {} and {}".format(imgfile, labelfile))
                    return None
                bboxes = json.loads(label_row[label_col])
            else:
                bboxes = json.loads(img_row[label_col])
            img = img_from_base64(img_row[img_col])
            height, width, channels = img.shape
            assert(channels == 3)
            for bbox in bboxes:
                new_rect = int_rect(bbox["rect"], enlarge_factor=enlarge_bbox,
                            im_h=height, im_w=width)
                left, top, right, bot = new_rect
                # ignore invalid bbox
                if bot - top < min_pixels or right - left < min_pixels:
                    if logger:
                        logger.info("skip invalid bbox in {}: {}".format(img_row[0], str(new_rect)))
                    continue
                info = [idx, left, top, right, bot]
                if for_test:
                    info.append(json.dumps(bbox))
                else:
                    # label only exists in training data
                    c = bbox["class"]
                    info.append(label_to_idx[c])
                ret.append(info)
        return ret

    m = pathos.multiprocessing.ProcessingPool(num_worker)
    all_res = m.map(_gen_index_helper, all_args)
    x = []
    for r in all_res:
        if r is None:
            raise Exception("fail to generate index")
        x.extend(r)
    return x
