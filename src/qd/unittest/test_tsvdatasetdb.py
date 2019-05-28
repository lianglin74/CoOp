import unittest
from qd.process_tsv import TSVDatasetDB
from qd.process_tsv import TSVDataset
from qd.process_tsv import rect_in_rects
import json


class TestTSVDatasetDB(unittest.TestCase):

    def setUp(self):
        pass

    def test_iter_gt(self):
        data, split, version = 'voc0712', 'test', -1
        c = 0
        for version in [0, 2, -1]:
            dataset = TSVDatasetDB(data)
            key_to_rects = {key: rects for idx, key, rects in dataset.iter_gt(split,
                version=version)}

            ref_dataset = TSVDataset(data)
            for key, str_rects in ref_dataset.iter_data(split, 'label',
                    version=version):
                ref_rects = json.loads(str_rects)
                rects = key_to_rects[key]
                for r in rects:
                    self.assertTrue(rect_in_rects(r, ref_rects, 0.99))
                    c = c + 1
        self.assertGreater(c, 0)

    def test_iter_gt_with_idx(self):
        data, split, version = 'voc0712', 'test', -1
        c = 0
        idx = [9, 8, 1000]
        for version in [0, 2, -1]:
            dataset = TSVDatasetDB(data)
            key_to_rects = {key: rects for idx, key, rects in dataset.iter_gt(split,
                version=version, idx=idx)}

            ref_dataset = TSVDataset(data)
            for key, str_rects in ref_dataset.iter_data(split, 'label',
                    version=version, filter_idx=idx):
                ref_rects = json.loads(str_rects)
                rects = key_to_rects[key]
                for r in rects:
                    self.assertTrue(rect_in_rects(r, ref_rects, 0.99))
                    c = c + 1
        self.assertGreater(c, 0)

if __name__ == '__main__':
    unittest.main()
