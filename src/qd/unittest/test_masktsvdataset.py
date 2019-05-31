#!nosetests --nocapture
import logging
import unittest
from qd.qd_common import init_logging

from maskrcnn_benchmark.data.datasets.masktsvdataset import MaskTSVDataset

class TestMaskTSVDataset(unittest.TestCase):
    def setUp(self):
        init_logging()

    def test_get_img_info_by_not_remove(self):
        dataset = MaskTSVDataset(data='coco2017Full', split='train',
                remove_images_without_annotations=False)
        result = dataset.get_img_info(999)
        self.assertEqual(result['height'], 483)
        self.assertEqual(result['width'], 640)

    def test_get_img_info_by_remove(self):
        dataset = MaskTSVDataset(data='coco2017Full', split='train',
                remove_images_without_annotations=True)
        result = dataset.get_img_info(999)
        self.assertEqual(result['height'], 640)
        self.assertEqual(result['width'], 363)

    def test_get_keys(self):
        dataset = MaskTSVDataset(data='coco2017Full', split='train',
                remove_images_without_annotations=True)
        key = dataset.get_keys()[999]
        self.assertEqual(key, '366009')

    def test_id_to_img_map(self):
        dataset = MaskTSVDataset(data='coco2017Full', split='train',
                remove_images_without_annotations=False)
        key = dataset.id_to_img_map[999]
        self.assertEqual(key, '366009')

if __name__ == '__main__':
    unittest.main()
