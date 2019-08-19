import logging
import unittest
import torch
import torch.nn.functional as F

from qd.qd_common import init_logging

class TestLayers(unittest.TestCase):
    def setUp(self):
        init_logging()

    def test_merge_batch_norm(self):
        import time
        def compare_tensors(t1, t2):
            s1 = t1.shape
            s2 = t2.shape
            for i, j in zip(s1, s2):
                self.assertEqual(i, j)
            self.assertAlmostEqual((t1 - t2).abs().mean().item(), 0, delta=1e-5)

        def compare_models(input, model1, model2):
            model1.eval()
            tic = time.time()
            with torch.no_grad():
                res1 = model1(input)
            time1 = time.time() - tic
            tic = time.time()
            model2.eval()
            with torch.no_grad():
                res2 = model2(input)
            time2 = time.time() - tic
            compare_tensors(res1, res2)
            logging.info("Time cost for models: {} v.s. {}".format(time1, time2))

        from qd.layers import MergeBatchNorm
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, (3,3), bias=True),
            torch.nn.BatchNorm2d(4))
        input = torch.randn(1, 2, 5, 5)
        compare_models(input, model, MergeBatchNorm(model))

        import torchvision
        model = torchvision.models.resnet50(pretrained=True)
        input = torch.randn(1, 3, 224, 224)
        compare_models(input, model, MergeBatchNorm(model))

        from mtorch.caffetorch import Scale
        model = torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, (3,3), bias=False),
            torch.nn.BatchNorm2d(4, affine=False),
            Scale(4),
            )
        input = torch.randn(1, 2, 5, 5)
        compare_models(input, model, MergeBatchNorm(model))


if __name__ == '__main__':
    unittest.main()
