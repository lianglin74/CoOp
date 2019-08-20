import torch
import unittest
import torch.nn.functional as F
import logging

class TestQDPyTorch(unittest.TestCase):
    def setUp(self):
        from qd.qd_common import init_logging
        init_logging()
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

    def test_resnet34_in_maskrcnn(self):
        config_file = './src/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_34_FPN_1x.yaml'
        net = 'resnet34'

        self.parity_check_resnet(config_file, net)

    def test_resnet50_in_maskrcnn(self):
        config_file = './src/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_50_FPN_1x_tbase.yaml'
        net = 'resnet50'

        self.parity_check_resnet(config_file, net)

    def test_x101_in_maskrcnn(self):
        config_file = './src/maskrcnn-benchmark/configs/e2e_faster_rcnn_X_101_32x8d_FPN_1x_tbase.yaml'
        net = 'resnext101_32x8d'

        self.parity_check_resnet(config_file, net)

    def test_r152_in_maskrcnn(self):
        config_file = './src/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_152_FPN_1x_tb.yaml'
        net = 'resnet152'

        self.parity_check_resnet(config_file, net)

    def parity_check_resnet(self, config_file, net):
        from qd.qd_pytorch import get_data_normalize
        import torchvision.transforms as transforms
        from qd.process_image import load_image
        import torchvision.models as models
        im = load_image('./aux_data/sample_images/TaylorSwift.jpg')
        #im = load_image('./aux_data/sample_images/square.png')

        min_size = 224

        normalize = get_data_normalize()
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(min_size),
            transforms.ToTensor(),
            normalize,
            ])
        net_input = transform(im)
        net_input = net_input[None, :]
        from torchvision.models.resnet import model_urls
        model = models.__dict__[net](pretrained=True)
        model.eval()
        model = model.double()
        net_input = net_input.double()
        net_output = model(net_input)
        net_output = F.softmax(net_output, dim=1)
        logging.info(net_output.max())
        logging.info(net_output.argmax())

        from maskrcnn_benchmark.config import cfg
        from qd.qd_common import load_from_yaml_file
        from qd.qd_maskrcnn import merge_dict_to_cfg
        param = load_from_yaml_file(config_file)
        param['INPUT']['MIN_SIZE_TEST'] = min_size
        merge_dict_to_cfg(param, cfg)
        from maskrcnn_benchmark.data.transforms import build_transforms
        mask_transforms = build_transforms(cfg, is_train=False)
        pil_image = transforms.ToPILImage()(im)
        from maskrcnn_benchmark.structures.bounding_box import BoxList
        import torch
        box = torch.empty((0, 4))
        mask_net_input, _ = mask_transforms(pil_image, BoxList(box, image_size=(416,
            416)))
        from maskrcnn_benchmark.modeling.detector import build_detection_model
        mask_model = build_detection_model(cfg)
        mask_model.eval()
        mask_net_input = mask_net_input[None, :].double()
        import torch.nn as nn
        avgpool = nn.AdaptiveAvgPool2d((1, 1))

        from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
        checkpointer = DetectronCheckpointer(cfg, mask_model)
        self.assertEqual(cfg.MODEL.WEIGHT, model_urls[net])
        checkpointer.load(cfg.MODEL.WEIGHT,
                model_only=True)

        outputs = mask_model.backbone.body.double()(mask_net_input)
        diff = (mask_net_input - net_input).abs().sum()
        s = mask_net_input.abs().sum()
        logging.info('input diff = {}'.format(diff / s))
        self.assertLess(diff / s, 0.01)
        x = outputs[-1]
        x = avgpool(x)
        x = x.view(x.size(0), -1)
        x = model.fc(x)
        x = F.softmax(x, dim=1)
        self.assertEqual(x.argmax(), net_output.argmax())
        diff = (x.max() - net_output.max()).abs()
        s = x.max()
        self.assertLess(diff / s, 0.1)

if __name__ == '__main__':
    unittest.main()
