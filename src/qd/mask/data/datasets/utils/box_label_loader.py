import torch
import numpy as np
import math
import pycocotools.mask as mask_utils

from qd.mask.structures.bounding_box import BoxList


class BoxLabelLoader(object):
    def __init__(self, labelmap, extra_fields=(), ignore_attrs=(),
            mask_mode="poly"):
        self.labelmap = labelmap
        self.extra_fields = extra_fields
        self.ignore_attrs = ignore_attrs
        assert mask_mode == "poly" or mask_mode == "mask"
        self.mask_mode = mask_mode
        self.all_fields = ["class", "mask", "confidence", 
            "attributes_encode", "IsGroupOf", "IsProposal"]

    def __call__(self, annotations, img_size, remove_empty=True):
        boxes = [obj["rect"] for obj in annotations]
        boxes = torch.as_tensor(boxes).reshape(-1, 4) 
        target = BoxList(boxes, img_size, mode="xyxy")

        for field in self.extra_fields:
            assert field in self.all_fields, "Unsupported field {}".format(field)
            if field == "class":
                classes = self.add_classes_with_ignore(annotations)
                target.add_field("labels", classes)
            elif field == "mask":
                masks, is_box_mask = self.add_masks(annotations, img_size)
                target.add_field("masks", masks)
                target.add_field("is_box_mask", is_box_mask)
            elif field == "confidence":
                confidences = self.add_confidences(annotations)
                target.add_field("confidences", confidences)
            elif field == "attributes_encode":
                attributes = self.add_attributes(annotations)
                target.add_field("attributes", attributes)
            elif field == "IsGroupOf":
                is_group = [1 if 'IsGroupOf' in obj and obj['IsGroupOf']==1 else 0 
                            for obj in annotations]
                target.add_field("IsGroupOf", torch.tensor(is_group))
            elif field == "IsProposal":
                is_proposal = [1 if "IsProposal" in obj and obj['IsProposal']==1 else 0
                            for obj in annotations]
                target.add_field("IsProposal", torch.tensor(is_proposal))
                
        target = target.clip_to_image(remove_empty=remove_empty)
        return target

    def add_classes_with_ignore(self, annotations):
        class_names = [obj["class"] for obj in annotations]
        classes = [None] * len(class_names)       
        if self.ignore_attrs:
            for i, obj in enumerate(annotations):
                if any([obj[attr] for attr in self.ignore_attrs if attr in obj]):
                    classes[i] = -1
        for i, cls in enumerate(classes):
            if cls!= -1:
                classes[i] = self.labelmap[class_names[i]] + 1 # 0 is saved for background
        return torch.tensor(classes)

    def add_masks(self, annotations, img_size):
        masks = []
        is_box_mask = []
        for obj in annotations:
            if "mask" in obj:
                masks.append(obj["mask"])
                is_box_mask.append(0)
            else:
                masks.append(self.get_box_mask(obj["rect"], img_size))
                is_box_mask.append(1)
        from mmask.structures.segmentation_mask import SegmentationMask
        masks = SegmentationMask(masks, img_size, mode=self.mask_mode)
        is_box_mask = torch.tensor(is_box_mask)
        return masks, is_box_mask

    def get_box_mask(self, rect, img_size):
        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        if self.mask_mode == "poly":
            return [[x1, y1, x1, y2, x2, y2, x2, y1]]
        elif self.mask_mode == "mask":
            # note the order of height/width order in mask is opposite to image
            mask = np.zeros([img_size[1], img_size[0]], dtype=np.uint8)
            mask[math.floor(y1):math.ceil(y2), math.floor(x1):math.ceil(x2)] = 255
            encoded_mask = mask_utils.encode(np.asfortranarray(mask))
            encoded_mask["counts"] = encoded_mask["counts"].decode("utf-8")
            return encoded_mask

    def add_confidences(self, annotations):
        confidences = []
        for obj in annotations:
            if "confidence" in obj:
                confidences.append(obj["confidence"])
            elif "conf" in obj:
                confidences.append(obj["conf"])
            else:
                confidences.append(1.0)
        return torch.tensor(confidences)

    def add_attributes(self, annotations):
        # we know that the maximal number of attributes per object is 16
        attributes = [ [0] * 16 for _ in range(len(annotations))]
        for i, obj in enumerate(annotations):
            attributes[i][:len(obj["attributes_encode"])] = obj["attributes_encode"]
        return torch.tensor(attributes)

