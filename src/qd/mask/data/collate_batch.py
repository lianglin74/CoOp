# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from qd.mask.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        if isinstance(batch[0], dict):
            images = [b['image'] for b in batch]
            images = to_image_list(images, self.size_divisible)
            targets = [b['target'] for b in batch]
            idx = [b['idx'] for b in batch]
            return {
                'images': images,
                'targets': targets,
                'idx': idx,
            }
        else:
            transposed_batch = list(zip(*batch))
            images = to_image_list(transposed_batch[0], self.size_divisible)
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
            return images, targets, img_ids
