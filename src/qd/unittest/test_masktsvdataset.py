#!nosetests --nocapture
import logging
import unittest
from qd.qd_common import init_logging

from maskrcnn_benchmark.data.datasets.masktsvdataset import MaskTSVDataset
from maskrcnn_benchmark.data.transforms.build import build_transforms

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

    def test_run_yolo_transform(self):
        from maskrcnn_benchmark.config import cfg
        cfg.INPUT.USE_FIXED_SIZE_AUGMENTATION = True

        transforms = build_transforms(cfg, is_train=True)
        dataset = MaskTSVDataset(data='TaxOI5CV1_1_5k_with_bb',
                split='train',
                transforms=transforms,
                remove_images_without_annotations=False,)
        dataset[2445892]

    def test_id_to_img_map(self):
        dataset = MaskTSVDataset(data='coco2017Full', split='train',
                remove_images_without_annotations=False)
        key = dataset.id_to_img_map[999]
        self.assertEqual(key, '366009')

    def test_input_same_for_classification(self):
        from maskrcnn_benchmark.config import cfg
        cfg.INPUT.TO_BGR255 = False
        cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
        cfg.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
        transform = build_transforms(cfg, is_train=False)

        dataset = MaskTSVDataset(data='voc20',
                split='test',
                transforms=transform,
                bgr2rgb=False,
                remove_images_without_annotations=False,)
        row0 = dataset[0]

        import torchvision.transforms as transforms
        from qd.qd_pytorch import get_data_normalize
        normalize = get_data_normalize()
        x = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
            ])
        import torchvision.datasets as datasets
        datasets.ImageFolder
        from qd.qd_pytorch import TSVSplitImage
        dataset2 = TSVSplitImage('voc20', 'test',
                    version=0,
                    transform=x)
        im0 = row0[0]
        im1 = dataset2[0][0]
        d = (im0.double() - im1.double()).abs().sum()
        logging.info(d)
        # note: remove the resizer in maskrcnn's build_transforms to assert d
        # == 0
        logging.info(im0[:, 0, 0])
        logging.info(im1[:, 0, 0])

if __name__ == '__main__':
    unittest.main()
