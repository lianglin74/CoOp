import unittest
import os.path as op
from mmdet.datasets import get_dataset
from tqdm import tqdm
import logging


class TestMMTSVDataset(unittest.TestCase):
    def setUp(self):
        pass

    def test_parity_check(self):
        img_norm_cfg = {'mean': [123.675, 116.28, 103.53],
                        'std': [58.395, 57.12, 57.375],
                        'to_rgb': True}
        annfile = op.expanduser('~/data/raw_data/raw_coco/annotations/instances_train2017.json')
        img_prefix = op.expanduser('~/data/raw_data/raw_coco/images/train2017')
        train_config = {
                'type': 'CocoDataset',
                'ann_file': annfile,
                'img_prefix': img_prefix,
                'img_scale':(1333, 800),
                'img_norm_cfg':img_norm_cfg,
                'size_divisor':32,
                'flip_ratio':0,
                'with_mask':False,
                'with_crowd':True,
                'with_label':True}

        train_dataset = get_dataset(train_config)
        del train_config['ann_file']
        del train_config['img_prefix']

        train_config['type'] = 'MMTSVDataset'
        train_config['data'] = 'coco2017Full'
        train_config['split'] = 'train'
        train_config['version'] = 0
        tsv_dataset = get_dataset(train_config)

        key_to_idx = {tsv_dataset.read_hw(i)[0]: i for i in range(len(tsv_dataset))}

        ignore_checked = 0
        assert len(train_dataset) == len(tsv_dataset)
        for i, row in tqdm(enumerate(train_dataset)):
            key = str(row['img_meta'].data['key'])
            idx = key_to_idx[key]
            row2 = tsv_dataset[idx]
            assert key == row2['img_meta'].data['key']
            d = (row['img'].data - row2['img'].data).abs().sum()
            assert float(d.cpu()) < 1
            d = (row['gt_bboxes'].data - row2['gt_bboxes'].data).abs().sum()
            assert float(d.cpu()) < 1
            if len(row['gt_bboxes_ignore'].data) > 0:
                d = (row['gt_bboxes_ignore'].data - row2['gt_bboxes_ignore'].data).abs(
                        ).sum()
                assert float(d.cpu()) < 1
                ignore_checked = ignore_checked + 1
            if (i % 1000) == 0:
                logging.info(ignore_checked)
            for x1, x2 in zip(row['gt_labels'].data, row2['gt_labels'].data):
                l1 = train_dataset.CLASSES[int(x1) - 1]
                l2 = tsv_dataset.CLASSES[int(x2) - 1]
                l2 = l2.replace(' ', '_')
                assert l1 == l2

if __name__ == '__main__':
    unittest.main()
