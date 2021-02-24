# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from qd.mask.modeling.box_coder import BoxCoder
from qd.mask.structures.bounding_box import BoxList
from qd.mask.structures.boxlist_ops import cat_boxlist
from qd.mask.structures.boxlist_ops import boxlist_nms
from qd.mask.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from .utils import permute_and_flatten

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
        x2y2correction=-1,
        fpn_post_nms_conf_th=-1,
        fpn_post_nms_top_n_each_image_train=0,
        nms_policy=None,
        master_mask_style_select_all=False,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.x2y2correction = x2y2correction

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_conf_th = fpn_post_nms_conf_th
        self.fpn_post_nms_top_n_each_image_train = fpn_post_nms_top_n_each_image_train
        if nms_thresh != nms_policy.THRESH:
            if nms_policy.TYPE == 'nms':
                import logging
                logging.info('num_policy.THRESH changed from {} to {}'.format(
                    nms_policy.THRESH, nms_thresh))
                nms_policy.THRESH = nms_thresh

        from qd.layers.boxlist_nms import create_nms_func
        self.nms_func = create_nms_func(nms_policy,
                max_proposals=self.post_nms_top_n,
                score_field='objectness')
        self.master_mask_style_select_all = master_mask_style_select_all

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors, objectness, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness.device
        N, A, H, W = objectness.shape

        # put in the same format as anchors
        objectness = permute_and_flatten(objectness, N, A, 1, H, W).view(N, -1)
        objectness = objectness.sigmoid()

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness, topk_idx = objectness.topk(pre_nms_top_n, dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression = box_regression[batch_idx, topk_idx]

        image_shapes = [box.size for box in anchors]
        concat_anchors = torch.cat([a.bbox for a in anchors], dim=0)
        concat_anchors = concat_anchors.reshape(N, -1, 4)[batch_idx, topk_idx]

        proposals = self.box_coder.decode(
            box_regression.view(-1, 4), concat_anchors.view(-1, 4),
            x2y2correction=self.x2y2correction
        )

        proposals = proposals.view(N, -1, 4)

        result = []
        for proposal, score, im_shape in zip(proposals, objectness, image_shapes):
            boxlist = BoxList(proposal, im_shape, mode="xyxy")
            boxlist.add_field("objectness", score)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            boxlist = self.nms_func(boxlist)
            boxlist = boxlist.convert('xyxy')
            result.append(boxlist)
        return result

    def forward(self, anchors, objectness, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(objectness)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, objectness, box_regression):
            sampled_boxes.append(
                self.forward_for_single_feature_map(a, o, b)
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        if num_levels > 1:
            boxlists = self.select_over_all_levels(boxlists)

        # append ground-truth bboxes to proposals
        if self.training and targets is not None:
            boxlists = self.add_gt_proposals(boxlists, targets)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # TODO resolve this difference and make it consistent. It should be per image,
        # and not per batch

        if self.master_mask_style_select_all:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")

                objectness_len = objectness.shape[0]
                if not isinstance(objectness_len, torch.Tensor):
                    objectness_len = torch.tensor(objectness_len)

                post_nms_top_n = torch.min(
                    torch.stack(
                        [torch.tensor(self.fpn_post_nms_top_n, dtype=torch.long),
                         objectness_len]).float()
                ).long()

                # post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
            return boxlists

        if self.training:
            fpn_post_nms_top_n = self.fpn_post_nms_top_n
            if self.fpn_post_nms_top_n_each_image_train:
                fpn_post_nms_top_n = num_images * self.fpn_post_nms_top_n_each_image_train
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.bool)
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                if self.fpn_post_nms_conf_th < 0:
                    # this is the default setting
                    post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                    top_objs, inds_sorted = torch.topk(
                        objectness, post_nms_top_n, dim=0, sorted=True
                    )
                else:
                    inds_sorted = (objectness >= self.fpn_post_nms_conf_th).nonzero(as_tuple=False).squeeze(dim=1)
                    if inds_sorted.numel() == 0:
                        inds_sorted = objectness.argmax().unsqueeze(dim=0)
                    elif inds_sorted.numel() > self.fpn_post_nms_top_n:
                        _, ii = torch.topk(objectness[inds_sorted],
                                           self.fpn_post_nms_top_n, dim=0)
                        inds_sorted = inds_sorted[ii]

                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    x2y2correction = config.MODEL.RPN.X2Y2CORRECTION
    fpn_post_nms_top_n_each_image_train = config.MODEL.RPN.FPN_POST_NMS_TOP_N_EACH_IMAGE_TRAIN
    master_mask_style_select_all = config.MODEL.RPN.MASTER_MASK_STYLE_SELECT_ALL

    fpn_post_nms_conf_th = -1
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST
        fpn_post_nms_conf_th = config.MODEL.RPN.FPN_POST_NMS_CONF_TH_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    nms_policy = config.MODEL.RPN.NMS_POLICY
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        x2y2correction=x2y2correction,
        fpn_post_nms_conf_th=fpn_post_nms_conf_th,
        fpn_post_nms_top_n_each_image_train=fpn_post_nms_top_n_each_image_train,
        nms_policy=nms_policy,
        master_mask_style_select_all=master_mask_style_select_all,
    )
    return box_selector
