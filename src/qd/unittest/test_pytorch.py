import torch
import unittest
import torch.nn.functional as F

class TestQDPyTorch(unittest.TestCase):
    def test_multi_hot_softmax_loss_with_bkg(self):
        num_label = 80
        num_feature = 10
        target = (torch.rand(num_feature) * num_label).long() + 1
        feature = torch.rand((num_feature, 1 + num_label))
        mceb_target = torch.zeros((num_feature, num_label),
                dtype=torch.long)
        for i in range(num_feature):
            l = target[i] - 1
            mceb_target[i][l] = 1

        ce_loss = F.cross_entropy(feature, target)
        from qd.qd_pytorch import MCEBLoss
        mceb_loss = MCEBLoss()(feature, mceb_target)
        self.assertEqual(ce_loss, mceb_loss)

if __name__ == '__main__':
    unittest.main()
