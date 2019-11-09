from qd.qd_common import read_to_buffer
import base64
import json
import random
import os.path as op
from qd.qd_common import load_list_file
import logging
from qd.tsv_io import tsv_writer
from qd.process_tsv import populate_dataset_details


def create_wider_face():
    raw_data_root = op.expanduser('~/data/raw_data/WIDER_FACE')
    name = 'WIDER_FACE'
    def wider_face_load_annotation(txt_file, image_folder, shuffle=False):
        all_line = load_list_file(txt_file)
        i = 0
        all_info = []
        while i < len(all_line):
            file_name = all_line[i]
            rects = []
            num_bb = int(float(all_line[i + 1]))
            for j in range(num_bb):
                line = all_line[i + 2 + j]
                info = [float(s.strip()) for s in line.split(' ')]
                assert len(info) == 10
                x1, y1, w, h = info[:4]
                rect = {'rect': [x1, y1, x1 + w, y1 + h], 'class': 'face'}
                rects.append(rect)
            all_info.append((file_name, rects))
            i = i + 2 + num_bb
        if shuffle:
            random.shuffle(all_info)
        for i, (file_name, rects) in enumerate(all_info):
            if (i % 100) == 0:
                logging.info('{}/{}'.format(i, len(all_info)))
            full_file_name = op.join(image_folder, file_name)
            yield file_name, json.dumps(rects), base64.b64encode(read_to_buffer(full_file_name))
    splits_in_tsv = ['train', 'test']
    splits_in_origin = ['train', 'val']
    for split_in_tsv, split_in_origin in zip(splits_in_tsv, splits_in_origin):
        txt_file = op.join(raw_data_root, 'wider_face_split',
            'wider_face_{}_bbx_gt.txt'.format(split_in_origin))
        folder = op.join(raw_data_root, 'WIDER_{}'.format(split_in_origin), 'images')
        tsv_writer(wider_face_load_annotation(txt_file, folder, True),
                op.join('data', name, '{}.tsv'.format(split_in_tsv)))
    populate_dataset_details(name)

