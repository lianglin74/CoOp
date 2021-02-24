# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from qd.mask.structures.bounding_box import BoxList
from qd.mask.structures.boxlist_ops import boxlist_nms
from qd.mask.structures.boxlist_ops import cat_boxlist
from qd.mask.modeling.box_coder import BoxCoder
from qd.mask.layers import nms as box_nms


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False,
        classification_activate='softmax',
        cfg=None,
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        self.min_detections_per_img = cfg.MODEL.ROI_HEADS.MIN_DETECTIONS_PER_IMG
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.min_size = cfg.TEST.BOX_MIN_SIZE
        self.output_feature = cfg.TEST.OUTPUT_FEATURE
        self.bbox_aug_enabled = bbox_aug_enabled
        self.nms_on_max_conf_agnostic = cfg.MODEL.ROI_HEADS.NMS_ON_MAX_CONF_AGNOSTIC
        if not self.cls_agnostic_bbox_reg:
            assert not self.nms_on_max_conf_agnostic
        self.classification_activate = classification_activate
        if self.output_feature:
            # needed to extract features when they have not been pooled yet
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        if classification_activate == 'softmax':
            self.logits_to_prob = lambda x: F.softmax(x, -1)
            self.cls_start_idx = 1
        elif classification_activate == 'sigmoid':
            self.logits_to_prob = torch.nn.Sigmoid()
            self.cls_start_idx = 0
        elif classification_activate == 'tree':
            from qd.layers import SoftMaxTreePrediction
            self.logits_to_prob = SoftMaxTreePrediction(
                    tree=cfg.MODEL.ROI_BOX_HEAD.TREE_0_BKG,
                    pred_thresh=self.score_thresh)
            self.cls_start_idx = 1
        else:
            raise NotImplementedError()
        self.cfg = cfg

        self.force_boxes = cfg.MODEL.ROI_BOX_HEAD.FORCE_BOXES

        #self.filter_method = self.filter_results
        #if self.cfg.MODEL.ROI_HEADS.NM_FILTER == 1:
            #self.filter_method = self.filter_results_nm
        if self.cfg.MODEL.ROI_HEADS.NM_FILTER == 2:
            self.filter_method = self.filter_results_peter
        elif self.cfg.MODEL.ROI_HEADS.NM_FILTER == 3:
            # from master maskrcnn code
            self.filter_method = self.filter_results_fast
        elif self.cfg.MODEL.ROI_HEADS.NM_FILTER == 4:
            self.filter_method = self.filter_results_ml_nms
        else:
            assert self.cfg.MODEL.ROI_HEADS.NM_FILTER == 0
            if self.nms_on_max_conf_agnostic:
                self.filter_method = self.filter_results_nms_on_max
            elif not self.bbox_aug_enabled:
                self.filter_method = self.filter_results

        self.x2y2correction = cfg.MODEL.RPN.X2Y2CORRECTION

    def forward(self, x, boxes, features):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        if self.output_feature:
            # TODO: ideally, we should have some more general way to always
            #       extract pooled features
            if len(features.shape) > 2:
                features = self.avgpool(features)
                features = features.view(features.size(0), -1)
        class_logits, box_regression = x
        class_prob = self.logits_to_prob(class_logits)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        boxes_per_image = [len(box) for box in boxes]
        if self.force_boxes:
            proposals = concat_boxes
        else:
            if self.cls_agnostic_bbox_reg:
                box_regression = box_regression[:, -4:]
            proposals = self.box_coder.decode(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes,
                        x2y2correction=self.x2y2correction
            )
            if self.cls_agnostic_bbox_reg and \
                not self.nms_on_max_conf_agnostic and \
                    self.cfg.MODEL.ROI_HEADS.NM_FILTER != 3:
                proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        features = features.split(boxes_per_image, dim=0)

        results = []
        start_idx_in_batch = 0
        for prob, boxes_per_img, image_shape, feature in zip(
            class_prob, proposals, image_shapes, features,
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            if self.force_boxes:
                if len(boxlist) > 0:
                    # predict the most likely object in the box
                    # Skip j = 0, because it's the background class
                    scores, labels = torch.max(prob[:, 1:], dim=1)
                    boxlist.extra_fields['scores'] = scores
                    boxlist.add_field('labels', labels + 1)
                    if self.output_feature:
                        boxlist.add_field('box_features', feature)
                else:
                    boxlist = self.prepare_empty_boxlist(boxlist)
            else:
                boxlist = boxlist.clip_to_image(remove_empty=False)
                boxlist.add_field(
                    'start_idx_in_batch',
                    torch.ones(len(boxlist), device=prob.device) * start_idx_in_batch)
                start_idx_in_batch += len(feature)
                boxlist = self.filter_method(boxlist, num_classes, feature)
                #if self.nms_filter_fast:
                    #boxlist = self.filter_results_fast(boxlist, num_classes, feature)
                #elif self.nms_on_max_conf_agnostic:
                    #boxlist = self.filter_results_nms_on_max(
                        #boxlist, num_classes,
                        #feature)
                #elif not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                    #boxlist = self.filter_results(
                        #boxlist, num_classes, feature)

            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def prepare_empty_boxlist(self, boxlist):
        device = boxlist.bbox.device
        boxlist_empty = BoxList(torch.zeros((0,4)).to(device), boxlist.size,
                mode='xyxy')
        boxlist_empty.add_field("scores", torch.Tensor([]).to(device))
        boxlist_empty.add_field("labels", torch.full((0,), -1,
                dtype=torch.int64, device=device))
        return boxlist_empty

    def filter_results_peter(self, boxlist, num_classes, feature=None):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        if len(boxlist) == 0:
            return boxlist
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        #if self.cfg.TEST.ACCUMULATE_SCORES:
            #scores = self.accumulate_scores(scores)

        nms_mask = scores.clone()
        nms_mask.zero_()

        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        for j in range(1, num_classes):
            scores_j = scores[:, j]
            boxes_j = boxes[:, j * 4: (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class.add_field("idxs",
                                        torch.arange(0, scores.shape[0]).long())
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, 0.3
            )
            nms_mask[:, j][boxlist_for_class.get_field("idxs")] = 1

        dists_all = nms_mask * scores

        # filter duplicate boxes
        scores_pre, labels_pre = dists_all.max(1)
        inds_pre = scores_pre.nonzero()
        assert inds_pre.dim() != 0
        inds_pre = inds_pre.squeeze(1)

        labels_pre = labels_pre[inds_pre]
        scores_pre = scores_pre[inds_pre]

        box_inds_pre = inds_pre * scores.shape[1] + labels_pre
        result = BoxList(boxlist.bbox.view(-1, 4)[box_inds_pre], boxlist.size,
                         mode="xyxy")
        result.add_field("labels", labels_pre)
        result.add_field("scores", scores_pre)
        if self.output_feature:
            features_pre = feature[inds_pre]
            result.add_field("box_features", features_pre)

        vs, idx = torch.sort(scores_pre, dim=0, descending=True)
        keep_boxes = torch.nonzero(scores_pre >= self.score_thresh, as_tuple=True)[0]
        num_dets = len(keep_boxes)
        if num_dets < self.min_detections_per_img:
            keep_boxes = idx[:self.min_detections_per_img]
        elif num_dets > self.detections_per_img:
            keep_boxes = idx[:self.detections_per_img]
        else:
            keep_boxes = idx[:num_dets]

        result = result[keep_boxes]
        return result

    def filter_results_nms_on_max(self, boxlist, num_classes, feature=None):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist

        # cpu version is faster than gpu. revert it to gpu only by verifying
        boxlist = boxlist.to('cpu')

        boxes = boxlist.bbox.reshape(-1, 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        result = []
        max_scores, _ = scores[:, self.cls_start_idx:].max(dim=1, keepdim=False)
        keep = (max_scores > self.score_thresh).nonzero(as_tuple=False).squeeze(-1)
        if len(keep) == 0:
            return self.prepare_empty_boxlist(boxlist)
        boxes, scores, max_scores = boxes[keep], scores[keep], max_scores[keep]

        boxlist = BoxList(boxes, boxlist.size, mode=boxlist.mode)
        boxlist.add_field("scores", max_scores)
        boxlist.add_field('original_scores', scores)
        if self.output_feature:
            boxlist.add_field("box_features", feature)
        boxlist = boxlist_nms(boxlist, self.nms)

        scores = boxlist.get_field('original_scores')
        all_idxrow_idxcls = (scores[:, self.cls_start_idx:] > self.score_thresh).nonzero()
        all_idxrow_idxcls[:, 1] += self.cls_start_idx

        boxes = boxlist.bbox
        boxes = boxes[all_idxrow_idxcls[:, 0]]
        if boxes.dim() == 1:
            boxes = boxes[None, :]
        labels = all_idxrow_idxcls[:, 1]
        scores =  scores[all_idxrow_idxcls[:, 0], all_idxrow_idxcls[:, 1]]
        result = BoxList(boxes, boxlist.size, mode=boxlist.mode)
        result.add_field("labels", labels)
        result.add_field("scores", scores)
        if self.output_feature:
            boxlist.add_field(
                "box_features",
                boxlist.get_field('box_features')[all_idxrow_idxcls[:, 0]],
            )

        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result

    def filter_results_ml_nms(self, boxlist, num_classes, feature=None):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero(as_tuple=False).squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            if self.output_feature:
                feature_j = feature.index_select(0, inds)
                boxlist_for_class.add_field("box_features", feature_j)
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        from qd.mask.structures.boxlist_ops import boxlist_ml_nms
        boxlist_for_class = boxlist_ml_nms(
            boxlist_for_class, self.nms
        )

        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            result = result[keep]
        return result

    def filter_results_fast(self, boxlist, num_classes, feature):
        """ perform only one NMS for all classes. 
        """
        assert boxlist.bbox.shape[1] == 4
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        # for each box, select max conf exclude background
        scores, labels = scores[:, 1:].max(1)
        labels += 1
        bbox = boxlist.bbox
        boxlist.add_field("scores", scores)
        boxlist.add_field("labels", labels)
        boxlist.add_field("box_features", feature)

        # threshold by size and confidence 
        # use a relatively low thresh to output enough boxes
        x1, y1, x2, y2 = bbox.split(1, dim=1)
        ws = (x2 - x1).squeeze(1)
        hs = (y2 - y1).squeeze(1)
        keep = (
            (ws >= self.min_size) & (hs >= self.min_size) & (scores > self.score_thresh * 0.01)
        ).nonzero(as_tuple=False).squeeze(1)
        del ws, hs

        # apply nms to the previous low-thresholded results
        nms_boxes = box_nms(bbox[keep], scores[keep], self.nms)
        nms_idx = keep[nms_boxes]  # indices that pass NMS and low-threshold
        nms_scores = scores[nms_idx]
        # sort above low-thresholded scores high to low
        _, idx = torch.sort(nms_scores, dim=0, descending=True)
        idx = nms_idx[idx]
        num_dets = (scores[keep] >= self.score_thresh).sum()

        if not isinstance(num_dets, torch.Tensor):
            num_dets = torch.as_tensor(num_dets, device=scores.device)
        min_det = torch.stack([num_dets, torch.as_tensor(self.min_detections_per_img, device=scores.device)]).max()
        max_det = torch.stack([min_det, torch.as_tensor(self.detections_per_img, device=scores.device)]).min()

        # first add good boxes then potentially append from high score
        keep_boxes = idx[:max_det]
        boxlist = boxlist[keep_boxes]
        # we keep this information, so that other layer's (e.g. feature
        # extraction) can find a way to figure out the correspondence. The
        # usage here is also the featur extraction.
        boxlist.add_field('nms_keep', keep_boxes)
        return boxlist

    def filter_results(self, boxlist, num_classes, feature=None):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(self.cls_start_idx, num_classes):
            inds = inds_all[:, j].nonzero(as_tuple=False).squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            if self.output_feature:
                feature_j = feature.index_select(0, inds)
                boxlist_for_class.add_field("box_features", feature_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED
    classification_activate = cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_ACTIVATE

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled,
        classification_activate=classification_activate,
        cfg=cfg,
    )
    return postprocessor
