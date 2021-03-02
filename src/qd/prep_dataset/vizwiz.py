from qd.tsv_io import TSVDataset
import os.path as op
import json
from qd.qd_common import read_to_buffer
import base64
from qd.qd_common import json_dump, list_to_dict
from qd.qd_common import qd_tqdm as tqdm
from qd.tsv_io import tsv_writers
from collections import OrderedDict, defaultdict


class VizWiz(object):
    def __init__(self):
        self.data = 'VizWizCaption'
        self.dataset = TSVDataset(self.data)

    def run(self):
        #self.create_tsv()
        #self.create_test_tsv()
        self.filter_canned_rejected()

    def filter_canned_rejected(self):
        split = 'train'
        for split in ['train', 'val']:
            iter_full = self.dataset.iter_data(split, 'caption', 'full')
            info = defaultdict(int)
            def gen_rows():
                for key, str_cap in iter_full:
                    caps = json.loads(str_cap)
                    info['before'] += len(caps)
                    caps = [c for c in caps if not c['is_rejected'] and not
                            c['is_precanned']]
                    info['after'] += len(caps)
                    yield key, json_dump(caps)
            assert not self.dataset.has(split, 'caption')
            self.dataset.write_data(gen_rows(), split, 'caption')

    def create_test_tsv(self):
        split = 'test'
        json_file = op.join(
            self.dataset._data_root,
            'annotations',
            '{}.json'.format(split),
        )
        data = json.loads(read_to_buffer(json_file))
        def gen_rows():
            for image_info in tqdm(data['images']):
                key = image_info['id']
                image_file = op.join(
                    self.dataset._data_root,
                    split,
                    image_info['file_name'],
                )
                from qd.process_image import load_image
                im = load_image(image_file)
                assert im is not None
                row_image = (key,
                             base64.b64encode(read_to_buffer(image_file)).decode())
                row_key_to_fname = (key, image_info['file_name'])
                row_label = (key, json_dump([]))

                yield row_image, row_key_to_fname, row_label

        tsv_image = self.dataset.get_data(split)
        tsv_key_to_fname = self.dataset.get_data(split, 'key2fname')
        tsv_label = self.dataset.get_data(split, 'label')
        tsv_writers(gen_rows(), (tsv_image, tsv_key_to_fname,
                                 tsv_label))
        from qd.process_tsv import populate_dataset_details
        from qd.process_tsv import populate_dataset_hw
        populate_dataset_details(self.data)
        populate_dataset_hw(self.data)

    def create_tsv(self):
        split = 'train'
        for split in ['train']:
            self.create_tsv_split(split)

    def create_tsv_split(self, split):
        json_file = op.join(
            self.dataset._data_root,
            'annotations',
            '{}.json'.format(split),
        )
        data = json.loads(read_to_buffer(json_file))
        iid_annos = [(x['image_id'], x) for x in data['annotations']]
        iid_to_annos = list_to_dict(iid_annos, 0)
        def gen_rows():
            for image_info in data['images']:
                key = image_info['id']
                image_file = op.join(
                    self.dataset._data_root,
                    split,
                    image_info['file_name'],
                )
                from qd.process_image import load_image
                im = load_image(image_file)
                assert im is not None
                row_image = (key,
                             base64.b64encode(read_to_buffer(image_file)).decode())
                row_key_to_fname = (key, image_info['file_name'])
                annos = iid_to_annos.get(key, [])
                row_caption = (key, json_dump(annos))
                row_label = (key, json_dump([]))

                yield row_image, row_key_to_fname, row_caption, row_label

        tsv_image = self.dataset.get_data(split)
        tsv_key_to_fname = self.dataset.get_data(split, 'key2fname')
        tsv_caption = self.dataset.get_data(split, 'caption', 'full')
        tsv_label = self.dataset.get_data(split, 'label')
        tsv_writers(gen_rows(), (tsv_image, tsv_key_to_fname, tsv_caption,
                                 tsv_label))
        from qd.process_tsv import populate_dataset_details
        from qd.process_tsv import populate_dataset_hw
        populate_dataset_details(self.data)
        populate_dataset_hw(self.data)
        from qd.process_tsv import generate_num_caption
        generate_num_caption(self.data, split, version=None)

