import copy
import base64
import cv2
from future.utils import viewitems
import os.path as op
from tqdm import tqdm
from qd.qd_common import cmd_run
from qd.qd_common import url_to_str
from qd.qd_common import write_to_file
from qd.qd_common import list_to_dict
from qd.qd_common import json_dump
from qd.qd_common import hash_sha1
from qd.qd_common import write_to_yaml_file
from qd.qd_common import read_to_buffer
from qd.qd_common import load_from_yaml_file
from qd.tsv_io import csv_reader
from qd.tsv_io import TSVDataset
from qd.tsv_io import tsv_reader, tsv_writer
from qd.taxonomy import Taxonomy
from qd.process_tsv import populate_dataset_details
from qd.process_image import draw_bb, show_image
from collections import OrderedDict
import simplejson as json
import logging

class OpenImageV6DetCreator(object):
    def __init__(self):
        self.target_dir = 'data/OpenImageV6Det/raw'
        self.out_data = 'OpenImageV6Det'

    def check_rotation(self):
        split = 'validation'
        iter_row = csv_reader(op.join(
            self.target_dir, '{}-images-with-rotation.csv'.format(split)))
        next(iter_row)
        num = 0
        for row in iter_row:
            if row[-1] not in ['', '0.0']:
                num += 1
        logging.info(num)

    def create_tsv_dataset(self):
        from qd.tsv_io import csv_reader
        cid_to_name = self.load_cid_to_name()

        splits = ['train', 'validation']
        splits = ['train']
        dataset = TSVDataset(self.out_data)
        debug = True
        debug = False
        for split in splits:
            anno_file = op.join(self.target_dir,
                    '{}-annotations-bbox.csv'.format(split))
            logging.info('loading {}'.format(anno_file))
            rows = csv_reader(anno_file)
            headers = next(rows)
            key = 'ImageID'
            x1_key, x2_key, y1_key, y2_key = 'XMin', 'XMax', 'YMin', 'YMax'
            label_key = 'LabelName'
            idx_x1, idx_x2, idx_y1, idx_y2 = headers.index(x1_key),\
                    headers.index(x2_key), headers.index(y1_key), \
                    headers.index(y2_key)
            idx_label = headers.index(label_key)
            idx_key = headers.index(key)
            other_idx = list(set(range(len(headers))).difference([idx_key,
                idx_label, idx_x1, idx_x2, idx_y1, idx_y2]))
            other_keys = [headers[i] for i in other_idx]
            #['ImageID', 'LabelName', 'XMin', 'XMax', 'YMin', 'YMax',
                    #'IsGroupOf']
            name_label_normbb_row = [(row[idx_key],
                                      cid_to_name[row[idx_label]],
                                      list(map(float, [row[idx_x1], row[idx_x2],
                                                       row[idx_y1], row[idx_y2]])),
                                      row)
                    for row in tqdm(rows)]
            name_to_label_normbb_row = list_to_dict(name_label_normbb_row, 0)
            def gen_rows():
                iter_rotation_info = csv_reader(op.join(
                    self.target_dir, '{}-images-with-rotation.csv'.format(split)))
                headers = next(iter_rotation_info)
                logging.info(headers)
                for image_info in tqdm(iter_rotation_info):
                    name = image_info[0]
                    if image_info[-1] in ['', '0.0']:
                        rotate = 0
                    else:
                        rotate = int(float(image_info[-1]))
                    assert rotate in [0, 90, 180, 270]
                    label_normbbs_row = name_to_label_normbb_row.get(name, [])
                    fname = op.join(self.target_dir, split, name + '.jpg')
                    im = cv2.imread(fname, cv2.IMREAD_COLOR)
                    assert im is not None
                    height, width = im.shape[:2]
                    for label, normbb, _ in label_normbbs_row:
                        normbb[0] = normbb[0] * width
                        normbb[1] = normbb[1] * width
                        normbb[2] = normbb[2] * height
                        normbb[3] = normbb[3] * height
                    rects = []
                    for label, normbb, row in label_normbbs_row:
                        rect = {'class': label,
                                'rect': [normbb[0], normbb[2], normbb[1],
                                    normbb[3]]}
                        for k, i in zip(other_keys, other_idx):
                            rect[k] = row[i]
                        rects.append(rect)

                    if rotate != 0:
                        from qd.process_image import load_image_by_pil
                        im = load_image_by_pil(fname)
                        im = im.rotate(rotate, expand=True)
                        from qd.qd_common import encoded_from_img
                        str_im = encoded_from_img(im)
                        from qd.qd_common import img_from_base64
                        im = img_from_base64(str_im)
                        for r in rects:
                            x1, y1, x2, y2 = r['rect']
                            if rotate == 90:
                                x1, y1 = y1, width - x1
                                x2, y2 = y2, width - x2
                            elif rotate == 180:
                                x1, y1 = width - x1, height - y1
                                x2, y2 = width - x2, height - y2
                            else:
                                assert rotate == 270
                                x1, y1 = height - y1, x1
                                x2, y2 = height - y2, x2
                            x1, x2 = min(x1, x2), max(x1, x2)
                            y1, y2 = min(y1, y2), max(y1, y2)
                            r['rect'] = [x1, y1, x2, y2]
                        debug = True
                    else:
                        str_im = base64.b64encode(read_to_buffer(fname))
                        debug = False

                    if debug:
                        draw_bb(im, [r['rect'] for r in rects],
                                [r['class'] for r in rects])
                        from qd.process_image import save_image
                        save_image(im,
                                   '/mnt/gpu02_raid/jianfw/work/tmp/{}_{}.jpg'.format(
                                       name, rotate))
                    yield (name, str_im), (name, json.dumps(rects))
            tsv_split = split if split != 'validation' else 'trainval'
            image_tsv = dataset.get_data(tsv_split)
            label_tsv = dataset.get_data(tsv_split, 'label')
            from qd.tsv_io import tsv_writers
            tsv_writers(gen_rows(), [image_tsv, label_tsv])

    def load_cid_to_name(self):
        cid_name = list(csv_reader(op.join(self.target_dir,
            'class-descriptions-boxable.csv')))
        assert len(set([cid for cid, name in cid_name])) == len(cid_name)
        cid_to_name = OrderedDict(cid_name)

        #assert cid_to_name['/m/020lf'] == 'Mouse'
        #cid_to_name['/m/020lf'] = 'Computer mouse'
        #assert cid_to_name['/m/076lb9'] == 'Bench'
        #cid_to_name['/m/076lb9'] = 'Training bench'
        return cid_to_name

    def parse_label_tree(self):
        cid_to_name = self.load_cid_to_name()
        write_to_file('\n'.join([cid_to_name[cid] for cid in cid_to_name]),
                './data/{}/labelmap.txt'.format(self.out_data))
        tsv_writer(((cid, name) for cid, name in viewitems(cid_to_name)),
                './data/{}/cid_to_name.tsv'.format(self.out_data))
        tsv_writer(((name, cid) for cid, name in viewitems(cid_to_name)),
                './data/{}/name_to_cid.tsv'.format(self.out_data))

        def map_cid(data):
            key_sub = 'Subcategory'
            if type(data) is dict:
                if 'LabelName' in data and data['LabelName'] in cid_to_name:
                    data['LabelName'] = cid_to_name[data['LabelName']]
                if key_sub in data:
                    for x in data[key_sub]:
                        map_cid(x)
                if 'Part' in data:
                    for x in data['Part']:
                        map_cid(x)
            else:
                assert False

        data = json.loads(read_to_buffer(op.join(self.target_dir,
            'bbox_labels_600_hierarchy.json')))
        map_cid(data)

        def rename_labelname(data):
            key_sub = 'Subcategory'
            if type(data) is dict:
                if 'LabelName' in data:
                    if key_sub in data:
                        for x in data[key_sub]:
                            rename_labelname(x)
                        data[data['LabelName']] = data[key_sub]
                        del data['LabelName']
                        del data[key_sub]
                    else:
                        data['name'] = data['LabelName']
                        #data[data['LabelName']] = []
                        del data['LabelName']
                else:
                    assert False
                if 'Part' in data:
                    assert isinstance(data['Part'], list)
                    data['Part'] = {'dummy': data['Part']}
            else:
                assert False

        rename_labelname(data)
        assert len(data) == 1
        for _, v in viewitems(data):
            data = v
            break
        tax = Taxonomy(data)
        x = tax.dump(for_train=True)
        write_to_yaml_file(x, './data/{}/taxonomy/a.yaml'.format(self.out_data))

