from PIL import Image
import random
import torch
import math
import numpy as np
import cv2
import random
from qd.qd_common import img_from_base64
<<<<<<< HEAD
import json


class FeatureDecoder(object):
    def __call__(self, row):
        key, str_rects = row
        rects = json.loads(str_rects)
        result = {'key': key,
         'feature': torch.tensor([r['feature'] for r in rects])}
        return result

class RemoveUselessKeys(object):
    def __init__(self, keys=None):
        if keys is None:
            self.keys = ['io_dataset']
        else:
            self.keys = keys
    def __call__(self, sample):
        for k in self.keys:
            if k in sample:
                del sample[k]
        return sample

class TwoCropsTransform(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, first_transform, second_transform=None):
        self.first_transform = first_transform
        self.second_transform = first_transform
        assert first_transform is not None
        if self.second_transform is None:
            self.second_transform = first_transform

    def __call__(self, x):
        q = self.first_transform(x)
        k = self.second_transform(x)
        return [q, k]

class MultiCropsTransform(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, transform_infos):
        self.transform_infos = transform_infos

    def __call__(self, x):
        result = []
        for trans_info in self.transform_infos:
            transform = trans_info['transform']
            repeat = trans_info['repeat']
            result.append([transform(x) for _ in range(repeat)])
        return result

class ImageCutout(object):
    def __init__(self, ratio):
        assert ratio < 1
        self.ratio = ratio
    def __repr__(self):
        return 'ImageCutout(ratio={})'.format(self.ratio)

    def __call__(self, im):
        depth, height, width = im.shape
        assert height == width
        size = int(height * self.ratio)
        h_center = int(random.random() * height)
        w_center = int(random.random() * width)
        side1 = size // 2
        side2 = size - side1
=======


class ImageCutout(object):
    def __init__(self, size):
        self.size = size

    def __repr__(self):
        return 'ImageCutout(size={})'.format(self.size)

    def __call__(self, im):
        depth, height, width = im.shape
        h_center = int(random.random() * height)
        w_center = int(random.random() * width)
        side1 = self.size // 2
        side2 = self.size - side1
>>>>>>> dffeca8... add utilities for data layer
        h1 = h_center - side1
        h2 = h_center + side2
        w1 = w_center - side1
        w2 = w_center + side2
        h1 = min(max(0, h1), height - 1)
        h2 = min(max(0, h2), height)
        w1 = min(max(0, w1), width - 1)
        w2 = min(max(0, w2), width)
        im[:, h1:h2, w1:w2] = 0
        return im

class LabelNormalizeByOriginWH(object):
    def __call__(self, data_dict):
        h = data_dict['original_height']
        w = data_dict['original_width']
        for l in data_dict['label']:
            x1, y1, x2, y2 = l['rect']
            l['rect'] = [x1 / w, y1 / h, x2 / w, y2 / h]
        return data_dict

class LabelNormalizeByNpImageWH(object):
    def __call__(self, data_dict):
        h, w = data_dict['image'].shape[:2]
        for l in data_dict['label']:
            x1, y1, x2, y2 = l['rect']
            l['rect'] = [x1 / w, y1 / h, x2 / w, y2 / h]
        return data_dict

class Label2IndexDict(object):
    def __init__(self, labelmap):
        self.label2idx = {l: i for i, l in enumerate(labelmap)}

    def __call__(self, dict_data):
        for l in dict_data['label']:
            l['class'] = self.label2idx[l['class']]
        return dict_data

class LabelXYXY2CXYWH(object):
    def __call__(self, dict_data):
        for l in dict_data['label']:
            x1, y1, x2, y2 = l['rect']
            l['rect'] = [(x1 + x2) / 2,
                         (y1 + y2) / 2,
                         (x2 - x1),
                         (y2 - y1)]
        return dict_data

class List2NumpyXYXYCLabelDict(object):
    def __call__(self, dict_data):
        label = dict_data['label']
        label = np.array([(r['rect'][0], r['rect'][1], r['rect'][2], r['rect'][3], r['class'])
                          for r in label
                          ])
        if len(label) == 0:
            label = np.zeros((0, 5))
        dict_data['label'] = label
        return dict_data

class List2NumpyLabelRectDict(object):
    def __call__(self, dict_data):
        label = dict_data['label']
        label = np.array([(r['class'], r['rect'][0], r['rect'][1], r['rect'][2], r['rect'][3])
                          for r in label
                          ])
        if len(label) == 0:
            label = np.zeros((0, 5))
        dict_data['label'] = label
        return dict_data

class CapBoxCount(object):
    # if the number of boxes is too large, teh trainign might crash, because
    # some detectors, e.g. faster-rcnn depends on the box count. When it
    # calculates teh assignment, it calculates the matrix of teh anchors vs gt
    def __call__(self, dict_data):
        label = dict_data['label']
        if len(label) > 300:
            random.shuffle(label)
            label = label[:300]
            dict_data['label'] = label
        return dict_data

class RemoveSmallLabelDict(object):
    def __call__(self, dict_data):
        label = dict_data['label']
        label = [l for l in label if l['rect'][2] - l['rect'][0] >= 1 and
                 l['rect'][3] - l['rect'][1] >= 1]
        dict_data['label'] = label
        return dict_data

class RemoveCrowdLabelDict(object):
    def __call__(self, dict_data):
        label = dict_data['label']
        label = [l for l in label if l.get('iscrowd', 0) == 0]
        dict_data['label'] = label
        return dict_data

class DecodeJsonLabelDict(object):
    def __call__(self, dict_data):
        label = dict_data['label']
        import json
        label = json.loads(label)
        dict_data['label'] = label
        return dict_data

class NpImageNorm01Dict(object):
    def __call__(self, dict_data):
        img = dict_data['image']
        img = img.astype(np.float32) / 255.
        dict_data['image'] = img
        return dict_data

class ImageNumpyArray2Tensor(object):
    def __call__(self, data_dict):
        image = data_dict['image']
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image)
        data_dict['image'] = image
        return data_dict

class CvImageBGR2RGBDict(object):
    def __call__(self, dict_data):
        img = cv2.cvtColor(dict_data['image'], cv2.COLOR_BGR2RGB)
        dict_data['image'] = img
        return dict_data

class CvImageDecodeDict(object):
    def __call__(self, dict_data):
        im = img_from_base64(dict_data['image'])
        assert im is not None
        dict_data['image'] = im
        dict_data['original_height'] = im.shape[0]
        dict_data['original_width'] = im.shape[1]
        return dict_data

class Normalize(object):
    pass

class RandomResizedCropDict(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        assert (scale[0] < scale[1]) and (ratio[0] < ratio[1])

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = img.size
        area = height * width

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, dict_data):
        out = dict(dict_data.items())
        img = dict_data['image']
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        import torch
        out['crop'] = torch.tensor([j, i, j + w, i + h])
        import torchvision.transforms.functional as F
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        out['image'] = img
        return out

    def __repr__(self):
        from torchvision.transforms.transforms import _pil_interpolation_to_str
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string

class SimCLRGaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample

class ImageTransform2Dict(object):
    def __init__(self, image_transform):
        self.image_transform = image_transform

    def __call__(self, dict_data):
        out = dict(dict_data.items())
        out['image'] = self.image_transform(dict_data['image'])
        return out

    def __repr__(self):
        return 'ImageTransform2Dict(image_transform={})'.format(
            self.image_transform,
        )

class RandomResizedCropMultiSize(object):
    def __init__(self, sizes, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.sizes = sizes
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
    def __call__(self, data_dict):
        size = self.sizes[data_dict['iteration'] % len(self.sizes)]
        from torchvision.transforms import transforms
        trans_func = transforms.RandomResizedCrop(size, scale=self.scale, ratio=self.ratio,
                                     interpolation=self.interpolation)
        data_dict['image'] = trans_func(data_dict['image'])
        return data_dict

class ImageToImageDictTransform(ImageTransform2Dict):
    pass
