# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from qd.mask.layers import smooth_l1_loss
from qd.mask.modeling.box_coder import BoxCoder
from qd.mask.modeling.matcher import Matcher
from qd.mask.structures.boxlist_ops import boxlist_iou
from qd.mask.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from qd.mask.modeling.utils import cat


class FastRCNNLossComputation(torch.nn.Module):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg=False,
        classification_loss_type='CE',
        num_classes=81,
        attribute_on=False,
        boundingbox_loss_type='SL1',
        cfg=None,
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        super().__init__()
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.attribute_on = attribute_on
        self.classification_loss_type = classification_loss_type
        if self.classification_loss_type == 'CE':
            #self._classifier_loss = F.cross_entropy
            self._classifier_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        elif self.classification_loss_type == 'BCE':
            from qd.qd_pytorch import BCEWithLogitsNegLoss
            self._classifier_loss = BCEWithLogitsNegLoss()
        elif self.classification_loss_type == 'BCEByPos':
            from qd.layers.loss import BCELogitsNormByPositive
            self._classifier_loss = BCELogitsNormByPositive('each')
        elif self.classification_loss_type.startswith('IBCE'):
            param = map(float, self.classification_loss_type[4:].split('_'))
            from qd.qd_pytorch import IBCEWithLogitsNegLoss
            self._classifier_loss = IBCEWithLogitsNegLoss(*param)
        elif self.classification_loss_type == 'MCEB':
            from qd.qd_pytorch import MCEBLoss
            self._classifier_loss = MCEBLoss()
        elif self.classification_loss_type == 'tree':
            tree_file = cfg.MODEL.ROI_BOX_HEAD.TREE_0_BKG
            from mtorch.softmaxtree_loss import SoftmaxTreeWithLoss
            self._classifier_loss = SoftmaxTreeWithLoss(
                tree_file,
                ignore_label=-1, # this is dummy value since this will not happend
                loss_weight=1,
                valid_normalization=True,
            )
        elif self.classification_loss_type == 'multi_domain':
            from qd.layers.loss import MultiDomainCrossEntropyWithLogits
            self._classifier_loss = MultiDomainCrossEntropyWithLogits()

        self.copied_fields = ["labels"]
        if self.attribute_on:
            self.copied_fields.append("attributes")

        self.num_classes = num_classes
        assert boundingbox_loss_type == 'SL1'

    def create_all_bkg_labels(self, num, device):
        if self.classification_loss_type in ['CE', 'tree']:
            return torch.zeros(num,
                dtype=torch.float32,
                device=device)
        elif self.classification_loss_type in ['BCE', 'BCEByPos'] or \
                self.classification_loss_type.startswith('IBCE'):
            return torch.zeros((num, self.num_classes),
                dtype=torch.float32,
                device=device)
        elif self.classification_loss_type in ['MCEB']:
            return torch.zeros((num, self.num_classes - 1),
                dtype=torch.float32,
                device=device)
        else:
            raise NotImplementedError(self.classification_loss_type)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(self.copied_fields)
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        if len(target) == 0:
            dummy_bbox = torch.zeros((len(matched_idxs), 4),
                    dtype=torch.float32, device=matched_idxs.device)
            from qd.mask.structures.bounding_box import BoxList
            matched_targets = BoxList(dummy_bbox, target.size, target.mode)
            matched_targets.add_field('labels', self.create_all_bkg_labels(
                len(matched_idxs), matched_idxs.device))
            matched_targets.add_field('tightness', torch.zeros(len(matched_idxs),
                        device=matched_idxs.device))
            matched_targets.add_field(
                'attributes',
                torch.zeros((len(matched_idxs), 16),
                            device=matched_idxs.device))
        else:
            matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        attributes = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

            if self.attribute_on:
                attributes_per_image = matched_targets.get_field("attributes")
                attributes_per_image = attributes_per_image.to(dtype=torch.int64)
                if len(targets_per_image) > 0:
                    # Label background (below the low threshold)
                    # attribute 0 is ignored in the loss
                    attributes_per_image[bg_inds,:] = 0
                    # Label ignore proposals (between low and high thresholds)
                    attributes_per_image[ignore_inds,:] = 0
                    # return attributes
                attributes.append(attributes_per_image)
            else:
                attributes.append([])

        #return labels, regression_targets
        result = {
            'labels': labels,
            'regression_targets': regression_targets,
        }
        if self.attribute_on:
            result['attributes'] = attributes
        return result

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        prepare_result = self.prepare_targets(proposals, targets)
        labels = prepare_result['labels']
        regression_targets = prepare_result['regression_targets']

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for i, (labels_per_image, regression_targets_per_image,
                proposals_per_image) in enumerate(zip(
            labels, regression_targets, proposals
        )):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            if self.attribute_on:
                # add attributes labels
                attributes_per_image = prepare_result['attributes'][i]
                proposals_per_image.add_field(
                    "attributes", attributes_per_image
                )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img,
                                             as_tuple=False).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def forward(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = box_regression.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )
        classification_loss = self._classifier_loss(class_logits, labels)

        if labels.dim() == 1:
            # get indices that correspond to the regression targets for
            # the corresponding ground truth labels, to be used with
            # advanced indexing
            sampled_pos_inds_subset = torch.nonzero(labels > 0, as_tuple=False).squeeze(1)
            if sampled_pos_inds_subset.numel() == 0:
                box_loss = torch.tensor(0., device=device)
            else:
                labels_pos = labels[sampled_pos_inds_subset]
                if self.cls_agnostic_bbox_reg:
                    map_inds = torch.tensor([4, 5, 6, 7], device=device)
                else:
                    map_inds = 4 * labels_pos[:, None] + torch.tensor(
                        [0, 1, 2, 3], device=device)
                sampled_box_regression = box_regression[sampled_pos_inds_subset[:, None], map_inds]
                sampled_box_target = regression_targets[sampled_pos_inds_subset]
                box_loss = smooth_l1_loss(
                    sampled_box_regression,
                    sampled_box_target,
                    size_average=False,
                    beta=1,
                )

                box_loss = box_loss / labels.numel()
        else:
            assert labels.dim() == 2
            x = torch.nonzero(labels > 0)
            if x.numel() == 0:
                box_loss = torch.tensor(0., device=device)
            else:
                sampled_pos_inds_subset = x[:, 0]
                if self.num_classes == labels.shape[1]:
                    labels_pos = x[:, 1]
                else:
                    assert self.num_classes == labels.shape[1] + 1
                    labels_pos = x[:, 1] + 1
                if self.cls_agnostic_bbox_reg:
                    map_inds = torch.tensor([0, 1, 2, 3], device=device)
                else:
                    map_inds = 4 * labels_pos[:, None] + torch.tensor(
                        [0, 1, 2, 3], device=device)
                sampled_box_regression = box_regression[sampled_pos_inds_subset[:, None], map_inds]
                sampled_box_target = regression_targets[sampled_pos_inds_subset]
                box_loss = smooth_l1_loss(
                    sampled_box_regression,
                    sampled_box_target,
                    size_average=False,
                    beta=1,
                )
                box_loss = box_loss / labels.shape[0]


        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)
    attribute_on = cfg.MODEL.ATTRIBUTE_ON

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    classification_loss_type = cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_LOSS
    num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
    cfg = cfg
    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg,
        classification_loss_type,
        num_classes,
        attribute_on=attribute_on,
        boundingbox_loss_type=cfg.MODEL.ROI_BOX_HEAD.BOUNDINGBOX_LOSS_TYPE,
        cfg=cfg,
    )

    return loss_evaluator
