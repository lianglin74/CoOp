import os
import torch
from torch.utils.data import Dataset

from qd.tsv_io import TSVDataset, TSVFile, tsv_reader, tsv_writer
from qd.qd_common import img_from_base64, generate_lineidx, load_from_yaml_file, FileProgressingbar, int_rect, hash_sha1, worth_create, load_list_file, json_dump
from qd.qd_pytorch import TSVSplitProperty

import base64
from collections import OrderedDict, defaultdict
import json
import logging
import math
import multiprocessing as mp
import numpy as np
import pathos.multiprocessing
import six
import time
import yaml

class TSVSplitImageBoxCrop(Dataset):
    def __init__(self, data, split, version, transform=None,
            cache_policy=None, labelmap=None, for_test=False, enlarge_bbox=1.0):
        self._image_tsv = TSVSplitProperty(data, split, t=None, cache_policy=cache_policy)
        self._label_tsv = TSVSplitProperty(data, split, t='label', version=version, cache_policy=cache_policy)
        self._crop_index_tsv = TSVSplitProperty(data, split, t='crop.index', version=version, cache_policy=cache_policy)

        # load the label map
        dataset = TSVDataset(data)
        if labelmap is None:
            labelmap = load_list_file(dataset.get_data(split, t='labelmap', version=version))
        elif type(labelmap) is str:
            labelmap = load_list_file(labelmap)
        assert type(labelmap) is list
        self.labelmap = labelmap
        self.label_to_idx = {l: i for i, l in enumerate(labelmap)}

        self.transform = transform
        self.for_test = for_test
        self.enlarge_bbox = enlarge_bbox
        self._label_counts = None

    def __getitem__(self, index):
        img_idx, bbox_idx = self._crop_index_tsv[index]
        img_idx, bbox_idx = int(img_idx), int(bbox_idx)
        key, _, str_img = self._image_tsv[img_idx]
        _, str_rects = self._label_tsv[img_idx]
        bbox = json.loads(str_rects)[bbox_idx]

        # NOTE: convert image array to RGB order
        cropped_img = self._crop_image(str_img, bbox["rect"])[:, :, ::-1]
        if self.transform is not None:
            cropped_img = self.transform(cropped_img)

        if self.for_test:
            return cropped_img, (key, json_dump(bbox))
        else:
            # NOTE: currenly only support single label
            label_idx = self.label_to_idx[bbox["class"]]
            label = torch.from_numpy(np.array(label_idx, dtype=np.int))
            return cropped_img, label

    def __len__(self):
        return len(self._crop_index_tsv)

    def _crop_image(self, img_str, rect):
        # NOTE: the image is BGR order
        img_arr = img_from_base64(img_str)
        height, width, _ = img_arr.shape
        new_rect = int_rect(rect, enlarge_factor=self.enlarge_bbox,
                            im_h=height, im_w=width)
        left, top, right, bot = new_rect
        return img_arr[top:bot, left:right]

    def is_multi_label(self):
        return False

    def get_num_labels(self):
        return len(self.label_to_idx)

    def get_labelmap(self):
        return self.labelmap

    @property
    def label_counts(self):
        assert not self.for_test
        if self._label_counts is None:
            self._label_counts = np.zeros(len(self.label_to_idx))
            for index in range(len(self._crop_index_tsv)):
                img_idx, bbox_idx = self._crop_index_tsv[index]
                img_idx, bbox_idx = int(img_idx), int(bbox_idx)
                _, str_rects = self._label_tsv[img_idx]
                bbox = json.loads(str_rects)[bbox_idx]
                self._label_counts[self.label_to_idx[bbox["class"]]] += 1
        return self._label_counts

    def get_target(self, index):
        img_idx, bbox_idx = self._crop_index_tsv[index]
        img_idx, bbox_idx = int(img_idx), int(bbox_idx)
        _, str_rects = self._label_tsv[img_idx]
        bbox = json.loads(str_rects)[bbox_idx]
        return self.label_to_idx[bbox["class"]]


class CropClassTSVDataset(Dataset):
    def __init__(self, tsvfile, labelmap, labelfile=None,
                 transform=None, logger=None, for_test=False, enlarge_bbox=1.0,
                 use_cache=True):
        """ TSV dataset with cropped images from bboxes labels
        Params:
            tsvfile: image tsv file, columns are key, bboxes, b64_image_string
            labelmap: file or list of all categories. It can be None if the dataset is for prediction
            labelfile: label tsv file, columns are key, bboxes
        """
        self.min_pixels = 3
        self.img_col = 2
        self.label_col = 1
        self.key_col = 0

        self.tsv = TSVFile(tsvfile)
        self.tsvfile = tsvfile
        self.labelfile = labelfile
        self.transform = transform
        self.label_to_idx = {}
        self.labels = []
        if not for_test and not labelmap:
            raise ValueError("must provide labelmap for train/val")
        if labelmap:
            if isinstance(labelmap, six.string_types):
                self.labels = [l[0] for l in tsv_reader(labelmap)]
            elif isinstance(labelmap, list):
                self.labels = labelmap
            else:
                raise ValueError("invalid labelmap type: {}".format(type(labelmap)))
            for i, label in enumerate(self.labels):
                assert(label not in self.label_to_idx)
                self.label_to_idx[label] = i

        self.logger = logger
        self._for_test = for_test
        self._enlarge_bbox = enlarge_bbox
        self._label_counts = None

        # TODO: generate idx file offline
        _cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
        self._bbox_idx_file = os.path.join(_cache_dir, "{}.tsv".format(
                hash_sha1((tsvfile, labelfile if labelfile else "", str(for_test), str(enlarge_bbox)))))
        try:
            if not use_cache or not os.path.isfile(self._bbox_idx_file) or worth_create(tsvfile, self._bbox_idx_file) \
                    or (labelfile and worth_create(labelfile, self._bbox_idx_file)):
                _class_instance_idx = self._generate_class_instance_index_parallel()
                tsv_writer(_class_instance_idx, self._bbox_idx_file)
            self._bbox_idx_tsv = TSVFile(self._bbox_idx_file)
        except Exception as e:
            if os.path.isfile(self._bbox_idx_file):
                os.remove(self._bbox_idx_file)
            raise e

        self.num_samples = 0
        for _ in tsv_reader(self._bbox_idx_file):
            self.num_samples += 1

    def label_dim(self):
        return len(self.label_to_idx)

    def is_multi_label(self):
        return False

    def get_labelmap(self):
        return self.labels

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
        # NOTE: convert image array to RGB order
        img = img_from_base64(row[self.img_col])[:,:,::-1]
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
        return self.num_samples

    def get_target(self, index):
        info = self._bbox_idx_tsv.seek(index)
        return int(info[5])

    @property
    def label_counts(self):
        assert not self._for_test
        if self._label_counts is None:
            self._label_counts = np.zeros(len(self.label_to_idx))
            for parts in tsv_reader(self._bbox_idx_file):
                self._label_counts[int(parts[5])] += 1
        return self._label_counts

class CropClassTSVDatasetYaml():
    def __init__(self, yaml_cfg, session_name, transform=None, logger=None, enlarge_bbox=1.0):
        if isinstance(yaml_cfg, six.string_types):
            cfg = load_from_yaml_file(yaml_cfg)
        else:
            cfg = yaml_cfg

        labelmap = cfg.get('labelmap', None)
        img_files = cfg[session_name]['tsv']
        label_files = cfg[session_name]['label']
        if isinstance(img_files, six.string_types):
            img_files = [img_files]
        if isinstance(label_files, six.string_types):
            label_files = [label_files]
        assert(len(img_files) == len(label_files))

        if not labelmap:
            assert(session_name == "test")
            self.labels = None
        elif isinstance(labelmap, six.string_types):
            self.labels = [l[0] for l in tsv_reader(labelmap)]
        elif isinstance(labelmap, list):
            self.labels = labelmap
        else:
            raise ValueError("invalid labelmap type: {}".format(type(labelmap)))

        self.datasets = [CropClassTSVDataset(img_file, self.labels, label_file,
            for_test=(session_name == "test"), transform=transform, logger=logger, enlarge_bbox=enlarge_bbox)
                for img_file, label_file in zip(img_files, label_files)]
        self.dataset_lengths = [len(d) for d in self.datasets]
        self.length = sum(self.dataset_lengths)
        self._label_counts = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        dataset_idx, sample_idx = self.__get_internal_index(index)
        return self.datasets[dataset_idx][sample_idx]

    def get_target(self, index):
        dataset_idx, sample_idx = self.__get_internal_index(index)
        return self.datasets[dataset_idx].get_target(sample_idx)

    def label_dim(self):
        return len(self.labels)

    def is_multi_label(self):
        return False

    def get_labelmap(self):
        return self.labels

    @property
    def label_counts(self):
        if self._label_counts is None:
            self._label_counts = np.zeros(len(self.labels))
            for d in self.datasets:
                cur_counts = d.label_counts
                assert len(cur_counts) == len(self._label_counts)
                self._label_counts = self._label_counts + cur_counts
        return self._label_counts

    def __get_internal_index(self, index):
        cum_length = 0
        dataset_idx = 0
        for _, length in enumerate(self.dataset_lengths):
            if cum_length + length > index:
                break
            cum_length += length
            dataset_idx += 1
        assert(dataset_idx < len(self.datasets))
        sample_idx = index - cum_length
        return dataset_idx, sample_idx


def gen_index(imgfile, labelfile, label_to_idx, for_test,
                enlarge_bbox, key_col, label_col, img_col,
                logger, min_pixels):
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
                    if c not in label_to_idx:
                        logging.info("label file: {}, class: {} not in labelmap".format(labelfile, c))
                        continue
                    info.append(label_to_idx[c])
                ret.append(info)
        return ret

    logging.info("generating index...")
    m = pathos.multiprocessing.ProcessingPool(num_worker)
    all_res = m.map(_gen_index_helper, all_args)
    x = []
    for r in all_res:
        if r is None:
            raise Exception("fail to generate index")
        x.extend(r)
    logging.info("finished generating index")
    return x
