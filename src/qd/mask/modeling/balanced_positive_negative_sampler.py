# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from collections import defaultdict


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.iter = 0

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        verbose = (self.iter % 100) == 0
        self.iter += 1
        info = defaultdict(int)
        for matched_idxs_per_image in matched_idxs:
            if matched_idxs_per_image.dim() == 2:
                # as long as it matched to one, it is good
                positive = torch.nonzero(
                    (matched_idxs_per_image >= 1).sum(dim=1), as_tuple=False).squeeze(1)
                negative = torch.nonzero(
                    (matched_idxs_per_image == 0).sum(dim=1) == matched_idxs_per_image.shape[1],
                    as_tuple=False).squeeze(1)
                pos_idx_per_image_mask = torch.zeros_like(
                    matched_idxs_per_image[:, 0], dtype=torch.uint8
                )
                neg_idx_per_image_mask = torch.zeros_like(
                    matched_idxs_per_image[:, 0], dtype=torch.uint8
                )
            else:
                positive = torch.nonzero(matched_idxs_per_image >= 1,
                                         as_tuple=False).squeeze(1)
                negative = torch.nonzero(matched_idxs_per_image == 0,
                                         as_tuple=False).squeeze(1)
                # create binary mask from indices
                pos_idx_per_image_mask = torch.zeros_like(
                    matched_idxs_per_image, dtype=torch.uint8
                )
                neg_idx_per_image_mask = torch.zeros_like(
                    matched_idxs_per_image, dtype=torch.uint8
                )
            if verbose:
                info['num_pos'] += positive.numel()
                info['num_neg'] += negative.numel()

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)
            if verbose:
                info['num_sel_pos'] += num_pos
                info['num_sel_neg'] += num_neg

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)
        if verbose:
            info = {k: v / len(matched_idxs) for k, v in info.items()}
            import logging
            from pprint import pformat
            logging.info('info = \n{}'.format(pformat(info)))

        return pos_idx, neg_idx
