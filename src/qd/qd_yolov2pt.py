from torch import Tensor
import torch
import numpy as np
from mtorch.Transforms import IMAGE, LABEL
from torchvision import transforms
from mtorch.Transforms import DarknetRandomResizeAndPlaceOnCanvas
from qd.qd_maskrcnn import boxlist_to_rects, list_dict_to_boxlist
import torchvision.transforms.functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList

def tensor2pil(t):
    trans = transforms.ToPILImage()
    p = trans(t)
    return p

def pil2tensor(p):
    t = F.to_tensor(p)
    return t

def box_list_to_pyyolov2_aug_general(target):
    # to the input arg for data augmentation impplemented in PyYoloV2
    labels5 = np.zeros((len(target), 5), dtype=np.float32)
    width, height = target.size
    for i, target_box in enumerate(target.bbox):
        labels5[i, :4] = (target_box[0] / width, target_box[1] / height,
            target_box[2] / width, target_box[3] / height)
        labels5[i, 4] = i

    meta = {'width': width, 'height': height}
    return labels5, meta

def box_list_to_pyyolov2_aug(target):
    # to the input arg for data augmentation impplemented in PyYoloV2
    labels5 = np.zeros((len(target), 5), dtype=np.float32)
    target_labels = target.get_field('labels')
    label5_id_to_target_label = {}
    width, height = target.size
    for i, (target_box, target_label) in enumerate(zip(target.bbox, target_labels)):
        labels5[i, :4] = (target_box[0] / width, target_box[1] / height,
            target_box[2] / width, target_box[3] / height)
        labels5[i, 4] = i
        label5_id_to_target_label[i] = target_label
    meta = {'width': width, 'height': height, 'id_to_label':
            label5_id_to_target_label}
    return labels5, meta

def pyyolov2_aug_to_box_list_general(labels5, meta, target):
    width, height = meta['width'], meta['height']
    boxes = [[l[0] * width, l[1] * height, l[2] * width, l[3] * height]
        for l in labels5]
    if len(boxes) > 0:
        boxes = Tensor(boxes)
    else:
        boxes = torch.empty((0, 4))
    result = BoxList(boxes, image_size=(width, height), mode='xyxy')
    idxes = [int(l[4]) for l in labels5]
    for field in target.fields():
        # masktsvdataset outputs float32 for the labels. We do not test if int64 or
        # not converting to float works, but float32 works.
        result.add_field(field, target.get_field(field)[idxes])
    return result

def pyyolov2_aug_to_box_list(labels5, meta):
    width, height = meta['width'], meta['height']
    boxes = [[l[0] * width, l[1] * height, l[2] * width, l[3] * height]
        for l in labels5]
    if len(boxes) > 0:
        boxes = Tensor(boxes)
    else:
        boxes = torch.empty((0, 4))
    result = BoxList(boxes, image_size=(width, height), mode='xyxy')
    # masktsvdataset outputs float32 for the labels. We do not test if int64 or
    # not converting to float works, but float32 works.
    if len(labels5) > 0:
        labels = torch.stack([meta['id_to_label'][l[4]] for l in labels5]).float()
    else:
        labels = torch.Tensor([])
    result.add_field('labels', labels)
    return result

class ResizeAndPlaceForMaskRCNN(object):
    def __init__(self, cfg):
        input_size = cfg.INPUT.FIXED_SIZE_AUG.INPUT_SIZE
        random_scale_min = cfg.INPUT.FIXED_SIZE_AUG.RANDOM_SCALE_MIN
        random_scale_max = cfg.INPUT.FIXED_SIZE_AUG.RANDOM_SCALE_MAX
        tries = cfg.INPUT.FIXED_SIZE_AUG.TRIES
        jitter = cfg.INPUT.FIXED_SIZE_AUG.JITTER
        self.placer = DarknetRandomResizeAndPlaceOnCanvas(
                canvas_size=(input_size, input_size),
                fixed_offset=False,
                default_pixel_value=0.5,
                scale=(random_scale_min, random_scale_max),
                tries=tries,
                jitter=jitter)

    def visualize(self, image, target):
        # used for debugging. testing __call__
        from qd.process_image import draw_rects, show_image
        im = np.array(image)

        rects = []
        for i, box in enumerate(target.bbox):
            rect = {'rect': list(map(float, box))}
            rect['class'] = str(target.get_field('labels')[i])
            rects.append(rect)

        draw_rects(rects, im)
        show_image(im)

    def __call__(self, image, target):
        #self.visualize(image, target)

        t_image = pil2tensor(image)
        labels5, meta = box_list_to_pyyolov2_aug_general(target)
        sample = {IMAGE: t_image, LABEL: labels5}

        sample = self.placer(sample)

        image = tensor2pil(sample[IMAGE])
        meta['width'], meta['height'] = image.size
        target = pyyolov2_aug_to_box_list_general(
                sample[LABEL],
                meta,
                target)

        #self.visualize(image, target)
        return image, target

