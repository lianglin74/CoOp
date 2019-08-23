import collections
import json
import logging
import math
import os.path as op
import random

from qd.tsv_io import TSVDataset, tsv_writer, TSVFile
from qd.process_tsv import populate_dataset_details

def ensure_populate_dataset_crop_index(data, split, version, num_min_samples=0):
    from qd.qd_common import int_rect

    dataset = TSVDataset(data)
    outfile = dataset.get_data(split, t='crop.index', version=version)
    if op.isfile(outfile):
        return

    hw_iter = dataset.iter_data(split, t='hw')
    label_iter = dataset.iter_data(split, t='label', version=version)
    label_to_indices = collections.defaultdict(list)

    for img_idx, (parts1, parts2) in enumerate(zip(hw_iter, label_iter)):
        assert parts1[0] == parts2[0]
        im_h, im_w = parts1[1].split(' ')
        for bbox_idx, bbox in enumerate(json.loads(parts2[1])):
            rect = int_rect(bbox["rect"], im_h=int(im_h), im_w=int(im_w))
            left, top, right, bot = rect
            if right - left >= 3 and bot - top >= 3:
                label_to_indices[bbox["class"]].append([img_idx, bbox_idx])
            else:
                logging.info("invalid bbox at data:{} split:{} version:{} image:{} bbox:{}"
                    .format(data, split, version, parts2[0], json.dumps(bbox)))
    all_indices = []
    for label in label_to_indices:
        cur_indices = label_to_indices[label]
        num_copies = 1
        if len(cur_indices) < num_min_samples:
            num_copies = int(math.ceil(num_min_samples / len(cur_indices)))
        for _ in range(num_copies):
            all_indices.extend(cur_indices)

    random.seed(6)
    random.shuffle(all_indices)
    tsv_writer(all_indices, outfile)

def build_balanced_crop_index(all_src_data_info,
            out_data, out_split="train", num_min_samples=0):
    out_dataset = TSVDataset(out_data)

    out_labelmap = []
    out_labelmap_set = set()
    out_imgs = []
    out_labels = []
    out_hws = []
    out_shuffle = []
    for src_idx, src_info in enumerate(all_src_data_info):
        if len(src_info) == 3:
            src_data, src_split, src_version = src_info
            src_weight = 1
        elif len(src_info) == 4:
            src_data, src_split, src_version, src_weight = src_info
        assert src_weight >= 1

        populate_dataset_details(src_data, check_image_details=True)
        src_dataset = TSVDataset(src_data)

        cur_labels = [p[0] for p in src_dataset.iter_data(src_split, t='labelmap', version=src_version)]
        for label in cur_labels:
            if label in out_labelmap_set:
                continue
            out_labelmap.append(label)
            out_labelmap_set.add(label)
        out_imgs.append(src_dataset.get_data(src_split))
        out_hws.append(src_dataset.get_data(src_split, 'hw'))
        out_labels.append(src_dataset.get_data(src_split, 'label', version=src_version))
        for label_idx in range(len(TSVFile(src_dataset.get_data(src_split)))):
            for _ in range(src_weight):
                out_shuffle.append([src_idx, label_idx])

    tsv_writer([[p] for p in out_labelmap], out_dataset.get_data(out_split, t='labelmap'))
    tsv_writer([[p] for p in out_labelmap], out_dataset.get_labelmap_file())
    tsv_writer([[p] for p in out_imgs], out_dataset.get_data(out_split + 'X'))
    tsv_writer([[p] for p in out_hws], out_dataset.get_data(out_split + 'X', 'hw'))
    tsv_writer([[p] for p in out_labels], out_dataset.get_data(out_split + 'X', 'label'))
    tsv_writer(out_shuffle, out_dataset.get_shuffle_file(out_split))

    ensure_populate_dataset_crop_index(out_data, out_split, 0, num_min_samples=num_min_samples)
