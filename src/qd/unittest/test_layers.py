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
            base = t1.abs().mean().item()
            if base != 0:
                self.assertAlmostEqual((t1 - t2).abs().mean().item() / base, 0, delta=1e-5)
            else:
                self.assertAlmostEqual((t1 - t2).abs().mean().item(), 0, delta=1e-5)
            #print(t1.abs().mean())
            #print(t2.abs().mean())

        def compare_models(input, model):
            model.eval()
            tic = time.time()
            with torch.no_grad():
                out1 = model(input)
            cost1 = time.time() - tic
            new_model = MergeBatchNorm(model)
            new_model.eval()
            tic = time.time()
            with torch.no_grad():
                out2 = new_model(input)
            cost2 = time.time() - tic
            compare_tensors(out1, out2)
            return cost1, cost2

        num_replica = 50
        from qd.layers import MergeBatchNorm
        total_cost1, total_cost2 = 0, 0
        for i in range(num_replica):
            model = torch.nn.Sequential(
                torch.nn.Conv2d(2, 4, (3,3), bias=True),
                torch.nn.BatchNorm2d(4))
            input = torch.randn(1, 2, 5, 5)
            cost1, cost2 = compare_models(input, model)
            total_cost1 += cost1
            total_cost2 += cost2
        total_cost1 /= float(num_replica)
        total_cost2 /= float(num_replica)
        logging.info("Time cost: {} v.s. {}".format(total_cost1,
                        total_cost2))

        import torchvision
        total_cost1, total_cost2 = 0, 0
        for i in range(num_replica):
            model = torchvision.models.resnet50(pretrained=True)
            input = torch.randn(1, 3, 224, 224)
            cost1, cost2 = compare_models(input, model)
            total_cost1 += cost1
            total_cost2 += cost2
        total_cost1 /= float(num_replica)
        total_cost2 /= float(num_replica)
        logging.info("Time cost: {} v.s. {}".format(total_cost1,
                        total_cost2))

        from mtorch.caffetorch import Scale
        total_cost1, total_cost2 = 0, 0
        for i in range(num_replica):
            model = torch.nn.Sequential(
                torch.nn.Conv2d(2, 4, (3,3), bias=False),
                torch.nn.BatchNorm2d(4, affine=False),
                Scale(4),
                )
            input = torch.randn(1, 2, 5, 5)
            cost1, cost2 = compare_models(input, model)
            total_cost1 += cost1
            total_cost2 += cost2
        total_cost1 /= float(num_replica)
        total_cost2 /= float(num_replica)
        logging.info("Time cost: {} v.s. {}".format(total_cost1,
                        total_cost2))


if __name__ == '__main__':
    unittest.main()
