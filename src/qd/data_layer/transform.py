from PIL import Image
import random
import torch
import math
import numpy as np
import cv2
import random
from qd.qd_common import img_from_base64
import json
import logging
import PIL
from .dataset import TSVSplitProperty
from torchvision.transforms import transforms
from qd.torch_common import IgnoreLastDimSparseTensor


class FeatureDecoder(object):
    def __call__(self, row):
        key, str_rects = row
        rects = json.loads(str_rects)
        result = {'key': key,
         'feature': torch.tensor([r['feature'] for r in rects])}
        return result

class RenameKey(object):
    def __init__(self, ft=None):
        # from to
        self.ft = ft
    def __call__(self, data):
        if self.ft is None:
            return data
        for k, k1 in self.ft.items():
            if k in data:
                v = data[k]
                data[k1] = v
                del data[k]
        return data

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
    def __repr__(self):
        return 'MultiCropsTransform({})'.format(self.transform_infos)

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
            # during test, sometimes the labels are not in the training
            # domains
            l['class'] = self.label2idx.get(l['class'], -1)
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
                          ], dtype=np.float)
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

class DecodeFeatureDict(object):
    def __call__(self, dict_data):
        infos = json.loads(dict_data['label'])
        fs = [i['feature'] for i in infos]
        fs = torch.tensor(fs)
        assert fs.shape[0] == 1, 'each image has one feature'
        fs = fs.squeeze(0)
        dict_data['label'] = fs
        return dict_data

class PILImageDecodeDict(object):
    def __call__(self, dict_data):
        from qd.qd_common import pilimg_from_base64
        im = pilimg_from_base64(dict_data['image'])
        if im is None:
            cv_im = img_from_base64(dict_data['image'])
            from qd.process_image import cvimg_to_pil
            im = cvimg_to_pil(cv_im)
            assert im is not None
        dict_data['image'] = im
        try:
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        w, h = im.size
        dict_data['original_height'] = h
        dict_data['original_width'] = w
        return dict_data

class PILShortestResize(object):
    # used by grid feature
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, dict_data):
        pil_image = dict_data['image']
        interp_method = 2
        w, h = pil_image.size

        scale = self.min_size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.min_size, scale * w
        else:
            newh, neww = scale * h, self.min_size

        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        pil_image = pil_image.resize((neww, newh), interp_method)
        dict_data['image'] = pil_image
        return dict_data

class PILToBGRTensor(object):
    # used by grid feature
    def __call__(self, dict_data):
        image = dict_data['image']
        image = image.convert('RGB')
        image = np.asarray(image)
        image = image[:, :, ::-1]
        dict_data['image'] = image
        dict_data["image"] = torch.as_tensor(np.ascontiguousarray(
            image.transpose(2, 0, 1)))
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


class RandomResizedCropMultiChoices(object):
    def __init__(self, choices):
        default_size = 224
        default_scale = (0.08, 1.)
        default_ratio = (3. / 4, 4. / 3)
        default_interpolation = Image.BILINEAR

        import copy
        self.choices = copy.deepcopy(choices)
        for c in self.choices:
            if 'size' not in c:
                c['size'] = default_size
            if 'scale' not in c:
                c['scale'] = default_scale
            if 'ratio' not in c:
                c['ratio'] = default_ratio
            if 'interpolation' not in c:
                c['interpolation'] = default_interpolation

    def __call__(self, data_dict):
        c = self.choices[data_dict['iteration'] % len(self.choices)]
        trans_func = transforms.RandomResizedCrop(
            c['size'],
            scale=c['scale'],
            ratio=c['ratio'],
            interpolation=c['interpolation'])
        data_dict['image'] = trans_func(data_dict['image'])
        return data_dict

class RandomResizedCropMultiSize(object):
    def __init__(self, sizes, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.sizes = sizes
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
    def __call__(self, data_dict):
        size = self.sizes[data_dict['iteration'] % len(self.sizes)]
        trans_func = transforms.RandomResizedCrop(size, scale=self.scale, ratio=self.ratio,
                                     interpolation=self.interpolation)
        data_dict['image'] = trans_func(data_dict['image'])
        return data_dict

class FourRotateImage(object):
    def __init__(self, prob_rotate):
        self.prob_rotate = prob_rotate

    def __call__(self, data_dict):
        image = data_dict['image']
        if random.random() > self.prob_rotate:
            degree = random.choice([1, 2, 3]) * 90
        else:
            degree = 0
        if degree != 0:
            image = image.rotate(degree)
        data_dict['image'] = image
        data_dict['four_rotate_idx'] = degree // 90
        return data_dict

class RandomGrayscaleDict(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, data_dict):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        img = data_dict['image']
        data_dict['is_color'] = 1 if (img.mode != 'L') else 0
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            import torchvision.transforms.functional as F
            img = F.to_grayscale(img, num_output_channels=num_output_channels)
            data_dict['image'] = img
            data_dict['is_color'] = 0
        return data_dict

    def __repr__(self):
        return self.__class__.__name__ + '(p={0})'.format(self.p)

class ImageToImageDictTransform(ImageTransform2Dict):
    pass

class IoURandomResizedCrop(object):
    def __init__(self, iou, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.iou = iou
        self.not_found = 0

    def __repr__(self):
        return ('IoURandomResizedCrop(iou={}, size={}, scale={}, ratio={}, '
                'interpolation={})'.format(self.iou,
                                           self.size,
                                           self.scale,
                                           self.ratio,
                                           self.interpolation))

    def get_params(self, img, anchor=None):
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
        scale = self.scale
        ratio = self.ratio

        for attempt in range(500):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                if anchor is None:
                    return i, j, h, w
                else:
                    from qd.qd_common import calculate_iou
                    i1, j1, h1, w1 = anchor
                    iou = calculate_iou([j, i, j + w, i + h],
                                        [j1, i1, j1 + w1, i1 + h1])
                    if iou > self.iou:
                        return i, j, h, w
        if anchor is not None:
            if (self.not_found % 100) == 0:
                logging.info('not found after 500 trials')
            self.not_found += 1
            return anchor

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

    def __call__(self, img1, img2):
        i1, j1, h1, w1 = self.get_params(img1)
        i2, j2, h2, w2 = self.get_params(img2, anchor=[i1, j1, h1, w1])
        import torchvision.transforms.functional as F
        img1 = F.resized_crop(img1, i1, j1, h1, w1, self.size, self.interpolation)
        img2 = F.resized_crop(img2, i2, j2, h2, w2, self.size, self.interpolation)
        return img1, img2

class MultiScaleRandomResizedCrop(object):
    def __init__(self, iou, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.iou = iou
        self.not_found = 0

    def __repr__(self):
        return ('IoURandomResizedCrop(iou={}, size={}, scale={}, ratio={}, '
                'interpolation={})'.format(self.iou,
                                           self.size,
                                           self.scale,
                                           self.ratio,
                                           self.interpolation))

    def get_params(self, img, anchor=None):
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
        scale = self.scale
        ratio = self.ratio

        for attempt in range(500):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                if anchor is None:
                    return i, j, h, w
                else:
                    from qd.qd_common import calculate_iou
                    i1, j1, h1, w1 = anchor
                    iou = calculate_iou([j, i, j + w, i + h],
                                        [j1, i1, j1 + w1, i1 + h1])
                    if iou > self.iou:
                        return i, j, h, w
        if anchor is not None:
            if (self.not_found % 100) == 0:
                logging.info('not found after 500 trials')
            self.not_found += 1
            return anchor

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

    def __call__(self, x):
        y1 = dict(x.items())
        y2 = dict(x.items())
        i1, j1, h1, w1 = self.get_params(x['image'])
        import torchvision.transforms.functional as F
        img1 = F.resized_crop(x['image'], i1, j1, h1, w1, self.size, self.interpolation)
        img2 = F.resized_crop(x['image'], i1, j1, h1, w1, [s // 2 for s in self.size], self.interpolation)
        y1['image'] = img1
        y2['image'] = img2
        return y1, y2

class TwoCropsTransformX():
    def __init__(self, aug1, aug_join, aug2):
        self.aug1 = aug1
        self.aug_join = aug_join
        self.aug2 = aug2

    def __repr__(self):
        s = 'TwoCropsTransformX(aug1={}, aug_join={}, aug2={})'.format(
            self.aug1, self.aug_join, self.aug2,
        )
        return s

    def __call__(self, x):
        y1 = self.aug1(x)
        y2 = self.aug1(x)
        y1, y2 = self.aug_join(y1, y2)
        y1 = self.aug2(y1)
        y2 = self.aug2(y2)
        return [y1, y2]

class TwoCropsTransform112():
    def __init__(self, aug1, aug_join, aug2):
        self.aug1 = aug1
        self.aug_join = aug_join
        self.aug2 = aug2

    def __repr__(self):
        s = 'TwoCropsTransformX(aug1={}, aug_join={}, aug2={})'.format(
            self.aug1, self.aug_join, self.aug2,
        )
        return s

    def __call__(self, x):
        x = self.aug1(x)
        y1, y2 = self.aug_join(x)
        y1 = self.aug2(y1)
        y2 = self.aug2(y2)
        return [y1, y2]

# together with CaptionIdxTSVDataset
class LoadLabel(object):
    def __init__(self, data, split, version):
        from .dataset import TSVSplitProperty
        self.label_tsv = TSVSplitProperty(
            data, split, 'label', version=version)

    def __repr__(self):
        return 'LoadLabel(data={}, split={}, version={})'.format(
            self.label_tsv.data, self.label_tsv.split, self.label_tsv.version,
        )

    def __call__(self, data):
        idx_img = data['idx_img']
        key, str_label = self.label_tsv[idx_img]
        rects = json.loads(str_label)
        assert key == data['key']
        data['label'] = rects
        return data

class LoadHW(object):
    def __init__(self, data, split):
        self.tsv = TSVSplitProperty( data, split, 'hw')

    def __call__(self, data):
        idx_img = data['idx_img']
        key, str_hw = self.tsv[idx_img]
        assert key == data['key']
        try:
            hw_info = json.loads(str_hw)
            if isinstance(hw_info, list):
                assert len(hw_info) == 1
                hw_info = hw_info[0]
            data.update(hw_info)
        except:
            h, w = map(int, str_hw.split(' '))
            data['height'] = h
            data['width'] = w
        return data

class LoadFeature(object):
    def __init__(self, data, split, version,
                 img_feature_dim,
                 max_len=50,
                 sort_by_conf=True,
                 ):
        self.tsv = TSVSplitProperty(
            data, split, 'feature', version=version)
        self.sort_by_conf = sort_by_conf
        self.max_len = max_len
        self.img_feature_dim = img_feature_dim

    def get_spatial_features(self, feat_info, data):
        img_height, img_width = data['height'], data['width']
        spatial_feats = []
        for f in feat_info:
            # spatial features follow OSCAR pre-processing
            box = f['rect'] # xyxy
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            scaled_width = box_width / img_width
            scaled_height = box_height / img_height
            scaled_x = box[0] / img_width
            scaled_y = box[1] / img_height
            spatial_feat = np.array([scaled_x, scaled_y, scaled_x + scaled_width,
                scaled_y + scaled_height, scaled_width, scaled_height], dtype=np.float32)
            spatial_feats.append(spatial_feat)
        return spatial_feats

    def __call__(self, data):
        idx_img = data['idx_img']
        key, str_feat = self.tsv[idx_img]
        assert key == data['key']
        feat_info = json.loads(str_feat)

        if self.sort_by_conf and any('conf' in f for f in feat_info):
            feat_info = sorted(feat_info, key = lambda x : -x['conf'])

        if len(feat_info) > self.max_len:
            if any('conf' in f for f in feat_info):
                feat_info = feat_info[:self.max_len]
            else:
                step = len(feat_info) // self.max_len
                feat_info = feat_info[0:(step * self.max_len):step]
                assert len(feat_info) == self.max_len, (self.max_len, step, len(feat_info))

        if len(feat_info) == 0:
            data['img_feats'] = torch.zeros((0, self.img_feature_dim))
            data['feats_conf'] = torch.zeros((0,))
            data['feats_class'] = []
            return data

        if all('feature' in f for f in feat_info):
            import base64
            feats = [np.frombuffer(base64.b64decode(f['feature']), np.float32) for f in feat_info]
        else:
            from qd.qd_common import decode_np
            feats = [decode_np(f['zlib_feature']).astype(np.float32) for f in feat_info]
        if any('rect' in f for f in feat_info):
            spatial_feats = self.get_spatial_features(feat_info, data)
            data['img_feats'] = torch.Tensor(np.concatenate((feats, spatial_feats), 1))
        else:
            # grid feature, where there is no box location
            data['img_feats'] = torch.Tensor(feats)
        assert data['img_feats'].shape[1] == self.img_feature_dim
        data['feats_conf'] = torch.tensor([
            f.get('conf', 1.) for f in feat_info])
        data['feats_class'] = [f['class'] for f in feat_info]
        return data

class LoadCaption(object):
    def __init__(self, data, split, version):
        super().__init__()
        self.tsv = TSVSplitProperty(data, split, 'caption', version=version)
    def __call__(self, data):
        idx_img = data['idx_img']
        key, str_cap = self.tsv[idx_img]
        assert key == data['key']
        caps = json.loads(str_cap)
        idx_cap = data['idx_cap']
        cap = caps[idx_cap]
        data['caption'] = cap
        return data

class LazyValue(object):
    def __init__(self, func, *args, **kwargs):
        self._value = None
        self.func = func
        self.evaluated = False
        self.args = args
        self.kwargs = kwargs

    @property
    def value(self):
        if not self.evaluated:
            args = [x.value if isinstance(x, LazyValue) else x for x in self.args]
            kwargs = dict((k, v.value if isinstance(v, LazyValue) else v) for k, v in
                          self.kwargs.items())
            self._value = self.func(*args, **kwargs)
            self.evaluated = True
        return self._value

class LazyTransform(object):
    def __init__(self):
        pass

    def get_input_fields(self):
        raise ValueError()

    def get_output_fields(self):
        raise ValueError()

    def transform(self, *args):
        raise ValueError()

    def __call__(self, data):
        args = [data[k] for k in self.get_input_fields()]
        x = LazyValue(self.transform, *args)
        out_fields = self.get_output_fields()
        for i, f in enumerate(out_fields):
            data[f] = LazyValue(self.pick, x, i)
        return data

    def pick(self, x, idx):
        return x[idx]

class IdentifyTextAB(object):
    # if it is captioning dataset, captioning description is text a; optionally
    # label str is text b; if it is qa, we have several options. by default,
    # question is text a and answer is text b
    def __init__(self, add_od_labels, od_label_conf, label_sort_by_conf,
                 unique_labels_on, qa2caption=None, sep_token=None):
        super().__init__()
        self.add_od_labels = add_od_labels
        self.od_label_conf = od_label_conf
        self.sort_by_conf = label_sort_by_conf
        self.unique_labels_on = unique_labels_on
        self.qa2caption = qa2caption
        self.sep_token = sep_token

    def __call__(self, data):
        # currently, this function is used to load information for current
        # instance and the negative instance. If we'd like to have different
        # behaviors for the negative instance, we can add options to this
        # function
        caption_dict = data['caption']
        if self.add_od_labels:
            label_info = data['label']
            for lab in label_info:
                if 'conf' not in lab:
                    lab['conf'] = 1.0
            if len(label_info) > 0 and self.od_label_conf > 0 and 'conf' in label_info[0]:
                # select labels based on confidence
                label_info = [l for l in label_info if l['conf'] >= self.od_label_conf]
            if self.sort_by_conf:
                label_info = sorted(label_info, key = lambda x : -x['conf'])
            if self.unique_labels_on:
                # keep the order so it is deterministic
                label_list = []
                for lab in label_info:
                    if lab['class'].lower() not in label_list:
                        label_list.append(lab['class'].lower())
                od_labels = ' '.join(label_list)
            else:
                od_labels = ' '.join([l['class'].lower() for l in label_info])
        else:
            od_labels = ''
        if 'caption' in caption_dict:
            caption = caption_dict['caption']
            data['text_ab_type'] = 'cap_label'
        else:
            question = caption_dict['question']
            if 'shortAnswer' in caption_dict:
                answer = caption_dict['shortAnswer']
            elif 'answer' in caption_dict:
                answer = caption_dict['answer']
            else:
                answer = ' '.join(caption_dict['answers'])
            if self.qa2caption is None:
                caption = question
                od_labels = answer
            elif self.qa2caption == 'QT_A':
                caption = question + od_labels
                od_labels = answer
            elif self.qa2caption == 'Q_TA':
                caption = question
                # no need to add whitespace around sep-token
                od_labels = od_labels + self.sep_token + answer
            elif self.qa2caption == 'QST_A':
                # no need to add whitespace around sep-token
                caption = question + self.sep_token + od_labels
                od_labels = answer
            elif self.qa2caption == 'QA_T':
                caption = question + answer
            else:
                raise NotImplementedError
            data['text_ab_type'] = 'qa'
        data['text_a'] = caption
        data['text_b'] = od_labels
        return data

class RandomPairNegative(object):
    def __init__(self, pert_caption_prob, pert_labels_prob,
                 load_negative_transform):
        self.pert_caption_prob = pert_caption_prob
        self.pert_labels_prob = pert_labels_prob
        self.pert_prob = pert_caption_prob + pert_labels_prob
        self.load_negative_transform = load_negative_transform

    def __call__(self, data):
        rand_num = random.random()
        if rand_num <= self.pert_prob:
            # randomly select caption or labels from a different image to form a negative pair
            dataset_len = len(data['dataset'])
            while True:
                rand_idx = random.randint(0, dataset_len - 1)
                neg_data = data['dataset'][rand_idx]
                if neg_data['idx_img'] != data['idx_img']:
                    break
            neg_data = self.load_negative_transform(neg_data)
            if rand_num <= self.pert_caption_prob:
                data['text_a'] = neg_data['text_a']
                data['text_changed'] = 'a'
            else:
                data['text_b'] = neg_data['text_b']
                data['text_changed'] = 'b'
            data['text_a_or_b_changed'] = True
        else:
            data['text_a_or_b_changed'] = False
        return data

class NoChange(object):
    def __call__(self, data):
        return data

class TokenizeTransform(object):
    def __init__(self, tokenizer, fields):
        self.tokenizer = tokenizer
        self.fields = fields

    def __call__(self, data):
        for f in self.fields:
            token_field = f + '_tokens'
            assert token_field not in data
            if isinstance(data[f], (list, tuple)):
                tokens = [self.tokenizer.tokenize(x) for x in data[f]]
                data[token_field] = tokens
                token_id_field = f + '_token_ids'
                assert token_id_field not in data
                data[token_id_field] = [self.tokenizer.convert_tokens_to_ids(t)
                                        for t in tokens]
        return data

class CaptionTensorizer(object):
    def __init__(self, tensorizer):
        self.tensorizer = tensorizer
    def __call__(self, data):
        x = self.tensorizer.tensorize_example(
                data['text_a'], data['img_feats'], text_b=data['text_b'], return_dict=True,
        )
        for k in x:
            assert k not in data or k == 'img_feats'
        data.update(x)
        data['max_seq_a_len'] = self.tensorizer.max_seq_a_len
        data['vocab_size'] = self.tensorizer.tokenizer.vocab_size
        return data

class PrepareLabel(object):
    def __init__(self,
                 label_type=None,
                 img_feat_label_type=None,
                 region_loss_for_unmatched=True
                 ):
        self.mask_loss_for_unmatched = False
        self.region_loss_for_unmatched = region_loss_for_unmatched
        self.label_type = label_type
        # 0). None, no need to predict the region
        # 1). randK, randomly to choose at most K regions for prediction
        # 2). topK, sort by the confidence score and choose the top-k regions
        self.img_feat_label_type = img_feat_label_type
        if img_feat_label_type is not None:
            if img_feat_label_type.startswith('rand'):
                self.img_feat_k = int(img_feat_label_type[4:])
            elif img_feat_label_type.startswith('top'):
                self.img_feat_k = int(img_feat_label_type[3:])

    def __repr__(self):
        return ('PrepareLabel(label_type={}, img_feat_label_type={}, '
                'region_loss_for_unmatched={})').format(
                    self.label_type,
                    self.img_feat_label_type,
                    self.region_loss_for_unmatched,
                )

    def __call__(self, data):
        img_feats = data['img_feats']
        input_ids = data['input_ids']
        is_matched = not data['text_a_or_b_changed']
        masked_pos = data['masked_pos']

        lm_labels_id_text = torch.ones_like(input_ids) * (-1)
        masked_ids = data['masked_ids']
        if not self.mask_loss_for_unmatched and not is_matched:
            max_seq_a_len = data['max_seq_a_len']
            # no masked loss for unmatched part
            num_masked_cap = sum(masked_pos[:max_seq_a_len]).item()
            if data['text_changed'] == 'a':
                # no masked loss for caption
                masked_pos[:max_seq_a_len] = 0
                masked_ids[:num_masked_cap] = 0
            else:
                # no masked loss for labels
                masked_pos[max_seq_a_len:] = 0
                masked_ids[num_masked_cap:] = 0
        lm_labels_id_text[masked_pos==1] = masked_ids[masked_ids!=0]

        if self.label_type is None:
            assert self.img_feat_label_type is None
            lm_labels_id_img = torch.ones(img_feats.shape[0], dtype=torch.long) * (-1)
            lm_labels_id = torch.cat((lm_labels_id_text, lm_labels_id_img))
            data['masked_lm_labels'] = lm_labels_id
            data['next_sentence_label'] = torch.tensor(is_matched).long()
        elif self.label_type == 'hot':
            raise ValueError('slow, use hots which leverages the sparse'
                             'representation')
            # this strategy is slow and it is better not to use this one
            from qd.torch_common import convert_single_label_to_one_hot_label
            if self.img_feat_label_type is None:
                lm_labels_id_img = torch.ones(img_feats.shape[0], dtype=torch.long) * (-1)
                lm_labels_id = torch.cat((lm_labels_id_text, lm_labels_id_img))
                data['masked_lm_labels'] = convert_single_label_to_one_hot_label(
                    lm_labels_id, data['vocab_size'])
            elif any(self.img_feat_label_type.startswith(k) for k in ['rand', 'top']):
                if self.img_feat_label_type.startswith('rand'):
                    feat_idx = random.choice(range(img_feats.shape[0]), self.img_feat_k)
                elif self.img_feat_label_type.startswith('top'):
                    feat_idx = sorted(list(range(data['feats_conf'])), key=lambda i:
                           -data['feats_conf'][i])
                    feat_idx = feat_idx[:self.img_feat_k]
                feat_gt = torch.ones((img_feats.shape[0], data['vocab_size'])) * -1
                for i in feat_idx:
                    token_ids = data['feats_class_token_ids'][i]
                    conf = data['feats_conf'][i]
                    feat_gt[i, token_ids] = conf
                lm_labels_id_text = convert_single_label_to_one_hot_label(
                    lm_labels_id_text, data['vocab_size'])
                data['masked_lm_labels'] = torch.cat((lm_labels_id_text, feat_gt))
            data['next_sentence_label'] = convert_single_label_to_one_hot_label(
                torch.tensor(is_matched).long(), 2)
        elif self.label_type == 'hots': # use sparse matrix
            # this strategy is slow and it is better not to use this one
            from qd.torch_common import convert_single_label_to_one_hot_label
            if self.img_feat_label_type is None or (not is_matched and not self.region_loss_for_unmatched):
                size = (len(lm_labels_id_text) + img_feats.shape[0], data['vocab_size'])
                xs = torch.nonzero(lm_labels_id_text != -1).reshape(-1)
                ys = lm_labels_id_text[xs]
                idx = torch.stack((xs, ys))
                value = torch.ones(len(xs))
                lm_labels_id_img = torch.ones(img_feats.shape[0], dtype=torch.long) * (-1)
                lm_labels_id = torch.cat((lm_labels_id_text, lm_labels_id_img))
                ignore_index = torch.nonzero(lm_labels_id == -1).reshape(-1)
                data['masked_lm_labels'] = IgnoreLastDimSparseTensor(
                    idx, value, size, ignore_index)
                #parity_check = convert_single_label_to_one_hot_label( lm_labels_id, data['vocab_size'])
                #x = data['masked_lm_labels'].to_dense()
                #x[x[:,0] == -1] = -1
                #logging.info('debugging')
                #assert (x - parity_check).abs().sum() == 0
            elif any(self.img_feat_label_type.startswith(k) for k in ['rand', 'top']):
                size = (len(lm_labels_id_text) + img_feats.shape[0], data['vocab_size'])
                ignore_index = torch.nonzero(lm_labels_id_text == -1).reshape(-1)

                xs = torch.nonzero(lm_labels_id_text != -1).reshape(-1)
                ys = lm_labels_id_text[xs].long()
                idx = torch.stack((xs, ys))
                value = torch.ones(len(xs))

                # use data['feats_conf'] rather than img_feats since img_feats
                # has been appended
                if self.img_feat_label_type.startswith('rand'):
                    if len(data['feats_conf']) <= self.img_feat_k:
                        feat_idx = list(range(len(data['feats_conf'])))
                    else:
                        feat_idx = random.sample(list(range(len(data['feats_conf']))), self.img_feat_k)
                elif self.img_feat_label_type.startswith('top'):
                    feat_idx = sorted(list(range(len(data['feats_conf']))), key=lambda i:
                           -data['feats_conf'][i])
                    feat_idx = feat_idx[:self.img_feat_k]

                img_idx = []
                img_value = []
                img_ignore = torch.ones(img_feats.shape[0])
                img_ignore[feat_idx] = 0
                img_ignore = torch.nonzero(img_ignore).reshape(-1) + len(lm_labels_id_text)
                ignore_index = torch.cat((ignore_index, img_ignore))
                for i in feat_idx:
                    token_ids = data['feats_class_token_ids'][i]
                    conf = data['feats_conf'][i]
                    for token_id in token_ids:
                        img_idx.append((len(lm_labels_id_text) + i, token_id))
                        img_value.append(conf)
                img_idx = torch.tensor(img_idx).t().long()
                img_value = torch.tensor(img_value)
                idx = torch.cat((idx, img_idx), dim=1)
                value = torch.cat((value, img_value))
                data['masked_lm_labels'] = IgnoreLastDimSparseTensor(
                    idx, value, size, ignore_index)

            data['next_sentence_label'] = convert_single_label_to_one_hot_label(
                torch.tensor(is_matched).long(), 2)
        else:
            raise NotImplementedError(self.label_type)

        return data

