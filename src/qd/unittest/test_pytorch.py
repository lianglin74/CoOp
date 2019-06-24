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
        model = models.__dict__['resnet34'](pretrained=True)
        model.eval()
        net_output = model(net_input)
        net_output = F.softmax(net_output, dim=1)
        logging.info(net_output.max())
        logging.info(net_output.argmax())

        from maskrcnn_benchmark.config import cfg
        from qd.qd_common import load_from_yaml_file
        from qd.qd_maskrcnn import merge_dict_to_cfg
        param = load_from_yaml_file('./src/maskrcnn-benchmark/configs/e2e_faster_rcnn_R_34_FPN_1x.yaml')
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
        mask_net_input = mask_net_input[None, :]
        import torch.nn as nn
        avgpool = nn.AdaptiveAvgPool2d((1, 1))

        from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
        checkpointer = DetectronCheckpointer(cfg, mask_model)
        checkpointer.load(cfg.MODEL.WEIGHT,
                model_only=True)

        outputs = mask_model.backbone.body(mask_net_input)

        logging.info((mask_net_input - net_input).abs().sum())
        x = outputs[-1]
        x = avgpool(x)
        x = x.view(x.size(0), -1)
        x = model.fc(x)
        x = F.softmax(x, dim=1)
        logging.info(x.max())
        logging.info(x.argmax())



if __name__ == '__main__':
    unittest.main()
