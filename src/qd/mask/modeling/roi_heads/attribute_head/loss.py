# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F


class AttributeRCNNLossComputation(torch.nn.Module):
    def __init__(self, cfg):
        """
        """
        super().__init__()
        # self.proposal_matcher = proposal_matcher
        # self.discretization_size = discretization_size
        self.loss_weight = cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT

    def forward(self, proposals, attribute_logits, targets=None):
        """
        Arguments:
            proposals (list[BoxList]): already contain gt_attributes
            attribute_logits (Tensor)

        Return:
            attribute_loss (Tensor): scalar tensor containing the loss
        """
        attributes = torch.cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

        # prepare attribute targets
        sim_attributes = attribute_logits.new(attribute_logits.size()).zero_()
        for i in range(len(attributes)):
            if len(torch.nonzero(attributes[i], as_tuple=False)) > 0:
                sim_attributes[i][attributes[i][torch.nonzero(attributes[i],
                                                              as_tuple=False)].long()] = 1.0 / len(
                    torch.nonzero(attributes[i], as_tuple=False))
        # TODO: do we need to ignore the all zero vector?
        attribute_loss = self.cross_entropy(attribute_logits, sim_attributes, loss_type="softmax")

        return self.loss_weight * attribute_loss

    def cross_entropy(self, pred, soft_targets, loss_type="softmax"):
        if loss_type == "sigmoid":
            return torch.mean(torch.sum(- soft_targets * F.logsigmoid(pred), 1))

        return torch.mean(torch.sum(- soft_targets * F.log_softmax(pred, dim=1), 1))


def make_roi_attribute_loss_evaluator(cfg):
    # no need to match any more because it is already done in box_head
    loss_evaluator = AttributeRCNNLossComputation(cfg)

    return loss_evaluator
