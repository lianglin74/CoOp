from collections import OrderedDict
from maskrcnn_benchmark.modeling.detector.generalized_rcnn import GeneralizedRCNN
from maskrcnn_benchmark.structures.image_list import to_image_list
from torch.nn.functional import interpolate
from maskrcnn_benchmark.structures.image_list import ImageList
import torch
from qd.qd_pytorch import visualize_maskrcnn_input


def create_ssfnp(**kwargs):
    return lambda cfg: SSFPN(cfg=cfg, **kwargs)

def make_divisible_by_padding(tensors, divisible):
    out_height = (tensors.shape[-2] + divisible - 1) // divisible * divisible
    out_width = (tensors.shape[-1] + divisible - 1) // divisible * divisible
    if out_height == tensors.shape[-2] and out_width == tensors.shape[-1]:
        return tensors

    out_shape = tensors.shape[:-2] + (out_height, out_width)
    out_tensors = tensors.new(*out_shape).zero_()

    out_tensors[:, :, : tensors.shape[-2], : tensors.shape[-1]].copy_(tensors)

    return out_tensors

class SSFPN(GeneralizedRCNN):
    def __init__(self, cfg,
            scale_factor, weight_original, weight_scaled, weight_ss,
            detach_larger=False, extra_conv_cfg=None):
        super().__init__(cfg)
        self.scale_factor = scale_factor
        self.size_divisible = cfg.DATALOADER.SIZE_DIVISIBILITY
        self.weight_original = weight_original
        self.weight_scaled = weight_scaled
        self.weight_ss = weight_ss
        self.debug = False
        self.detach_larger = detach_larger

        if extra_conv_cfg is not None:
            num = extra_conv_cfg['num']
            in_channels = extra_conv_cfg['in_channels']
            out_channels = extra_conv_cfg['out_channels']
            ms = []
            for i in range(num):
                conv = torch.nn.Conv2d(in_channels=in_channels,
                        out_channels=out_channels, kernel_size=1)
                ms.append(('extra_gap_conv_{}'.format(i), conv))
            self.extra_conv = torch.nn.Sequential(OrderedDict(ms))
        else:
            self.extra_conv = None

    def forward_one(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        assert self.training
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, features

    def forward(self, images, targets=None):
        if not self.training:
            return super().forward(images, targets)
        if self.debug:
            visualize_maskrcnn_input(images, targets)
        loss1, feature1 = self.forward_one(images, targets)
        images, targets = self.downsample(images, targets)
        if self.debug:
            visualize_maskrcnn_input(images, targets)
        loss2, feature2 = self.forward_one(images, targets)

        loss3 = self.ssloss(feature1, feature2)

        self.aggregate_loss_(loss1, loss2, loss3)
        return loss1

    def aggregate_loss_(self, loss1, loss2, loss3):
        for k in loss1:
            loss1[k] *= self.weight_original
        for k in loss2:
            loss1[k + '_downsample'] = loss2[k] * self.weight_scaled
        loss1['feature_diff'] = loss3 * self.weight_ss

    def ssloss(self, feature1, feature2):
        loss = 0
        for idx in range(1, len(feature1)):
            f1 = feature1[idx]
            if self.detach_larger:
                f1 = f1.detach()
            f2 = feature2[idx - 1]

            if self.extra_conv is not None:
                f2 = self.extra_conv[idx - 1](f2)
            if self.scale_factor <= 0.5:
                f1to2 = interpolate(f1, size=f2.shape[-2:],
                        mode='bilinear', align_corners=False)
                loss += (f2 - f1to2).abs().mean()
            else:
                raise ValueError()
                f2to1 = interpolate(f2, size=f1.shape[-2:],
                        mode='bilinear', align_corners=False)
                loss += (f1 - f2to1).abs().mean()
        return loss

    def downsample(self, images, targets):
        out_tensors = interpolate(images.tensors, scale_factor=self.scale_factor,
                mode='bilinear', align_corners=False)
        # make it the same as what BatchCollator does
        out_tensors = make_divisible_by_padding(out_tensors, self.size_divisible)
        out_image_sizes = tuple([torch.Size([int(x * self.scale_factor) for x
            in image_size]) for image_size in images.image_sizes])
        out_targets = [t.resize((image_size[1], image_size[0]))
                for t, image_size in zip(targets, out_image_sizes)]
        return ImageList(out_tensors, out_image_sizes), out_targets

