import base64
from qd.process_image import draw_bb, show_image
import cv2
from qd.tsv_io import TSVDataset
from qd.qd_common import write_to_yaml_file
from qd.taxonomy import Taxonomy
from future.utils import viewitems
import os.path as op
from qd.process_tsv import populate_dataset_details
from qd.qd_common import cmd_run
from tqdm import tqdm
from qd.qd_common import url_to_str
from qd.qd_common import write_to_file
from qd.tsv_io import csv_reader
from collections import OrderedDict
import simplejson as json
from qd.qd_common import read_to_buffer
from qd.tsv_io import tsv_writer
import logging
from qd.qd_common import list_to_dict


class OpenImageV5CCreator(object):
    def __init__(self):
        self.target_dir = op.expanduser('~/data/raw_data/open_image_v5_challenge/')
        self.out_data = 'OpenImageV5C'

    def run(self):
        self.download_image()
        self.download_annotations()
        self.parse_label_tree()
        self.create_tsv_dataset()
        populate_dataset_details(self.out_data)

    def download_image(self):
        splits = ['train', 'validation', 'test']
        splits = ['validation']
        for split in splits:
            cmd = ['aws', 's3', '--no-sign-request', 'sync',
                    's3://open-images-dataset/{}'.format(split),
                    op.join(self.target_dir, split)]
            cmd_run(cmd)

    def download_annotations(self):
        all_url = ['https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-train-detection-bbox.csv',
                'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-validation-detection-bbox.csv',
                'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-label500-hierarchy.json',
                'https://storage.googleapis.com/openimages/challenge_2019/challenge-2019-classes-description-500.csv']
        for u in tqdm(all_url):
            s = url_to_str(u)
            write_to_file(s, op.join(self.target_dir, op.basename(u)))

    def load_cid_to_name(self):
        cid_name = list(csv_reader(op.join(self.target_dir,
            'challenge-2019-classes-description-500.csv')))
        assert len(set([cid for cid, name in cid_name])) == len(cid_name)
        cid_to_name = OrderedDict(cid_name)
        assert cid_to_name['/m/020lf'] == 'Mouse'
        cid_to_name['/m/020lf'] = 'Computer mouse'
        assert cid_to_name['/m/076lb9'] == 'Bench'
        cid_to_name['/m/076lb9'] = 'Training bench'
        return cid_to_name

    def parse_label_tree(self):
        data = json.loads(read_to_buffer(op.join(self.target_dir,
            'challenge-2019-label500-hierarchy.json')))
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
        map_cid(data)

        key_sub = 'Subcategory'
        def rename_labelname(data):
            if type(data) is dict:
                if 'LabelName' in data:
                    if key_sub in data:
                        for x in data[key_sub]:
                            rename_labelname(x)
                        data[data['LabelName']] = data[key_sub]
                        del data['LabelName']
                        del data[key_sub]
                    else:
                        data[data['LabelName']] = []
                        del data['LabelName']
                else:
                    assert False
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

    def create_tsv_dataset(self):
        from qd.tsv_io import csv_reader
        cid_to_name = self.load_cid_to_name()

        splits = ['train', 'validation']
        splits = ['train']
        dataset = TSVDataset(self.out_data)
        debug = False
        for split in splits:
            anno_file = op.join(self.target_dir,
                    'challenge-2019-{}-detection-bbox.csv'.format(split))
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
                for name in tqdm(name_to_label_normbb_row):
                    label_normbbs_row = name_to_label_normbb_row[name]
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
                    if debug:
                        draw_bb(im, [r['rect'] for r in rects],
                                [r['class'] for r in rects])
                        show_image(im)
                    yield name, json.dumps(rects), \
                        base64.b64encode(read_to_buffer(fname))
            dataset.write_data(gen_rows(), 'trainval' if split == 'validation'
                    else split)

def test_open_image_challenge_v5():
    c = OpenImageV5CCreator()
    c.run()

