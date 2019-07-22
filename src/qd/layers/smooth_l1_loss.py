from maskrcnn_benchmark.layers import smooth_l1_loss
import torch
import logging


class SmoothL1LossWithIgnore(torch.nn.Module):
    def __init__(self, beta=1./9, size_average=True,
            valid_iou_lower=0.1, valid_iou_upper=1.):
        super(SmoothL1LossWithIgnore, self).__init__()
        self.beta = beta
        self.size_average = size_average
        self.valid_iou_lower = valid_iou_lower
        self.valid_iou_upper = valid_iou_upper

        self.log_step = 100
        self.curr_iter = 0

    def forward(self, input, target, iou):
        indicator = ((iou >= self.valid_iou_lower) & (iou <= self.valid_iou_upper))
        if (self.curr_iter % self.log_step) == 0:
            logging.info('valid = {}/{}; lower={}; upper={}'.format(
                indicator.sum(),
                len(indicator),
                self.valid_iou_lower,
                self.valid_iou_upper))
        self.curr_iter += 1
        return smooth_l1_loss(input[indicator, :], target[indicator, :],
                beta=self.beta,
                size_average=self.size_average)

