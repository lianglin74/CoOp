# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
        adaptive = cfg.INPUT.COLORJITTER_ADAPTIVE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0
        adaptive = None

    to_bgr255 = cfg.INPUT.TO_BGR255
    from qd.qd_pytorch import (DictTransformMaskNormalize,
            DictTransformMaskColorJitter,
            DictTransformMaskToTensor,
            DictTransformMaskRandomHorizontalFlip,
            )
    normalize_transform = DictTransformMaskNormalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
            )
    color_jitter = DictTransformMaskColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            adaptive=adaptive,
            )
    to_tensor = DictTransformMaskToTensor()
    flipper = DictTransformMaskRandomHorizontalFlip(flip_prob)

    from qd.qd_pytorch import DictTransformMaskResize
    if is_train:
        if cfg.INPUT.TRAIN_RESIZER != '':
            from qd.qd_common import load_from_yaml_str
            from qd.qd_common import execute_func
            resizer = execute_func(load_from_yaml_str(cfg.INPUT.TRAIN_RESIZER))
        elif not cfg.INPUT.USE_FIXED_SIZE_AUGMENTATION:
            resizer = DictTransformMaskResize(
                min_size,
                max_size,
                cfg.INPUT.MIN_SIZE_ON_ITER,
                cfg.INPUT.TREAT_MIN_AS_MAX,
            )
        else:
            from qd.qd_yolov2pt import DictResizeAndPlaceForMaskRCNN
            resizer = DictResizeAndPlaceForMaskRCNN(cfg)
    else:
        resizer = DictTransformMaskResize(
            min_size,
            max_size,
            cfg.INPUT.MIN_SIZE_ON_ITER,
            cfg.INPUT.SMART_RESIZE_ON_MIN_IN_TEST,
            resize_method=cfg.INPUT.TEST_RESIZE_METHOD
        )

    from qd.qd_pytorch import DictTransformCompose
    transform = DictTransformCompose(
            [
                color_jitter,
                resizer,
                flipper,
                to_tensor,
                normalize_transform,
            ]
        )
    return transform

def build_transforms_mmask(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        random_crop = cfg.INPUT.RANDOM_CROP_TRAIN
        crop_min_scale = cfg.INPUT.CROP_MIN_SCALE
        crop_max_scale = cfg.INPUT.CROP_MAX_SCALE
        crop_iou_thresh = cfg.INPUT.CROP_IOU_THRESH
        flip_horizontal_prob = cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        random_crop = False
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255,
        cv2_input=cfg.INPUT.CV2_OUTPUT
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    if cfg.INPUT.CV2_OUTPUT:
        # TODO: random_crop not supported right now
        transform = T.Compose(
            [
                T.RandomHorizontalFlip(flip_horizontal_prob, cv2_input=True),
                T.RandomVerticalFlip(flip_vertical_prob, cv2_input=True),
                normalize_transform,
                T.Resize(min_size, max_size, cv2_input=True),
                T.ToTensor(), # numpy array to tensor; no need to change
            ]
        )
        return transform

    if random_crop:
        transform = T.Compose(
            [
                color_jitter,
                T.RandomCrop(crop_min_scale, crop_max_scale, crop_iou_thresh),
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomVerticalFlip(flip_vertical_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                color_jitter,
                T.Resize(min_size, max_size),
                T.RandomHorizontalFlip(flip_horizontal_prob),
                T.RandomVerticalFlip(flip_vertical_prob),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform
