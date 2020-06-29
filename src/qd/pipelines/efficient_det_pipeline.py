import torch
import numpy as np
import torchvision.transforms as transforms
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
import logging
from qd.tsv_io import TSVDataset
from qd.qd_pytorch import TSVSplitProperty
from qd.qd_common import img_from_base64
import cv2
import json
from qd.qd_pytorch import get_default_mean, get_default_std
from qd.data_layer.transform import RemoveUselessKeys


class EfficientDetDataset(object):
    def __init__(self, data, split, version, labelmap=None, transform=None):
        self.data = data

        self.image_tsv = TSVSplitProperty(data, split, t=None)
        self.label_tsv = TSVSplitProperty(data, split, t='label',
                                          version=version)
        self.transform = transform
        self._labelmap = labelmap

    @property
    def label2idx(self):
        return {l: i for i, l in enumerate(self.labelmap)}

    @property
    def labelmap(self):
        if self._labelmap is None:
            self._labelmap = TSVDataset(self.data).load_labelmap()
        return self._labelmap

    def get_keys(self):
        keys = [self.label_tsv[i][0] for i in
                range(len(self.label_tsv))]
        return keys

    def __getitem__(self, idx_info):
        idx = idx_info['idx']
        #logging.info('debugging')
        #idx = (idx % 100)
        image_row = self.image_tsv[idx]
        image_key, str_im = image_row[0], image_row[-1]
        img = img_from_base64(str_im)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        label_row = self.label_tsv[idx]
        label_key = label_row[0]
        assert label_key == image_key
        rects = json.loads(label_row[-1])

        annot = np.array([(r['rect'][0], r['rect'][1], r['rect'][2], r['rect'][3], self.label2idx.get(r['class'], -1))
                          for r in rects if not r.get('iscrowd') and
                          r['rect'][2] >= r['rect'][0] + 1 and
                          r['rect'][3] >= r['rect'][1] + 1
                          ])
        if len(annot) == 0:
            annot = np.zeros((0, 5))

        sample = {'image': img, 'label': annot}
        if self.transform:
            sample = self.transform(sample)
        sample['key'] = image_key
        return sample

    def __len__(self):
        return len(self.label_tsv)

class Augmenter(object):
    def __init__(self, flip_x=0.5):
        self.flip_x = 0.5

    def __call__(self, sample):
        if np.random.rand() < self.flip_x:
            image, annots = sample['image'], sample['label']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample['image'] = image
            sample['label'] = annots

        return sample

def test_collater(data):
    images = [b.pop('image') for b in data]
    from torch.utils.data._utils.collate import default_collate
    result = default_collate(data)
    all_height_width = [i.shape[1:] for i in images]
    if all(all_height_width[i] == all_height_width[0]
           for i in range(1, len(all_height_width))):
        result['image'] = torch.stack(images, 0)
    else:
        max_height = max([h for h, w in all_height_width])
        max_width = max([w for h, w in all_height_width])
        images2 = []
        for im in images:
            im2 = torch.zeros((im.shape[0], max_height, max_width),
                                   dtype=im.dtype)
            im2[:, :im.shape[1], :im.shape[2]] = im
            images2.append(im2)
        result['image'] = torch.stack(images2, 0)
    return result

def train_collater(data):
    imgs = [s['image'] for s in data]
    annots = [s['label'] for s in data]
    #scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    #return {'image': imgs, 'label': annot_padded, 'scale': scales}
    return imgs, annot_padded, None

class RandomResizeCrop(object):
    def __init__(self, all_crop_size, random_scale_factor=1.2):
        self.all_crop_size = all_crop_size
        self.random_scale_factor = random_scale_factor

    def __call__(self, sample):
        #visualize(sample)
        if len(self.all_crop_size) == 1:
            crop_size = self.all_crop_size[0]
        else:
            crop_size = self.all_crop_size[sample['iteration'] % len(self.all_crop_size)]
        image, annots = sample['image'], sample['label']
        height, width, _ = image.shape
        s1, s2 = crop_size / height, crop_size / width
        min_scale = min(s1, s2) / self.random_scale_factor
        max_scale = max(s1, s2) * self.random_scale_factor
        import random
        scale = random.random() * (max_scale - min_scale) + min_scale

        resized_height = int(scale * height)
        resized_width = int(scale * width)

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        top = int(random.random() * (resized_height - height))
        left = int(random.random() * (resized_width - width))
        new_image = np.zeros((crop_size, crop_size, 3))
        new_image_top = 0 if top > 0 else -top
        new_image_left = 0 if left > 0 else -left
        real_crop = image[top:(top + crop_size), left: (left + crop_size)]
        if real_crop.size > 0:
            new_image[new_image_top:(new_image_top + real_crop.shape[0]),
                      new_image_left: (new_image_left + real_crop.shape[1])] = real_crop

        annots[:, :4] *= scale
        annots[:, 0] -= left
        annots[:, 1] -= top
        annots[:, 2] -= left
        annots[:, 3] -= top
        annots = annots.clip(0, crop_size)
        valid = (annots[:, 2] > annots[:, 0]) & (annots[:, 3] > annots[:, 1])
        annots = annots[valid, :]

        #visualize({'image': new_image, 'label': annots})

        sample['image'] = torch.from_numpy(new_image).to(torch.float32)
        sample['label'] = torch.from_numpy(annots)
        sample['scale'] = scale
        sample['top'] = top
        sample['left'] = left

        return sample

def visualize(sample):
    mean = get_default_mean()
    std = get_default_std()
    np_mean = np.array([mean])
    np_std = np.array([std])
    image = sample['image']
    image = image * np_std + np_mean
    image *= 255
    image = image.astype(np.uint8)
    image = image[:, :, [2, 1, 0]]
    from qd.process_image import show_image, draw_rects
    rects = [{'rect': r[:4], 'class': r[4]} for r in sample['label']]
    draw_rects(rects, image)
    show_image(image)

class SmartResizer(object):
    def __init__(self, all_crop_size):
        self.all_crop_size = all_crop_size

    def __call__(self, sample):
        if len(self.all_crop_size) == 1:
            crop_size = self.all_crop_size[0]
        else:
            crop_size = self.all_crop_size[sample['iteration'] % len(self.all_crop_size)]
        image, annots = sample['image'], sample['label']
        height, width, _ = image.shape

        from qd.process_tsv import smart_resize_lefttop
        new_image, scale = smart_resize_lefttop(image, crop_size, lower=True)

        annots[:, :4] *= scale

        sample['image'] = torch.from_numpy(new_image).to(torch.float32)
        sample['label'] = torch.from_numpy(annots)
        sample['scale'] = scale

        #visualize({'image': new_image, 'label': annots})

        return sample

class Resizer(object):
    def __init__(self, all_crop_size):
        self.all_crop_size = all_crop_size

    def __call__(self, sample):
        if len(self.all_crop_size) == 1:
            crop_size = self.all_crop_size[0]
        else:
            crop_size = self.all_crop_size[sample['iteration'] % len(self.all_crop_size)]
        image, annots = sample['image'], sample['label']
        height, width, _ = image.shape
        if height > width:
            scale = crop_size / height
            resized_height = crop_size
            resized_width = int(width * scale)
        else:
            scale = crop_size / width
            resized_height = int(height * scale)
            resized_width = crop_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((crop_size, crop_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        sample['image'] = torch.from_numpy(new_image).to(torch.float32)
        sample['label'] = torch.from_numpy(annots)
        sample['scale'] = scale

        #visualize(sample)

        return sample

class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image = sample['image']
        sample['image'] = ((image.astype(np.float32) - self.mean) / self.std)
        return sample

class HWC2CHWTest(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        x = sample['image']
        x = x.permute(2, 0, 1)
        sample['image'] = x
        # this class is used in inference only and annot is useless here. We
        # explicitly remove it because each image has different number of boxes
        # and we need a custom collater function to handle it.
        del sample['label']
        return sample

class EfficientDetPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        curr_default = {
            'net': 0,
            'anchors_scales': [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
            'anchors_ratios': [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
            'convert_bn': 'SBN',
            'focal_alpha': 0.25,
            'prior_prob': 0.01,
            'cls_loss_type': 'FL',
            'smooth_bce_pos': 0.99,
            'smooth_bce_neg': 0.01,
            'focal_gamma': 2.,
            'reg_loss_type': 'L1',
            'at_least_1_assgin': False,
            'neg_iou_th': 0.4,
            'pos_iou_th': 0.5,
            'cls_weight': 1.,
            'reg_weight': 1.,
            'adaptive_up': False,
            'test_resize_type': 'smart_lower',
            'test_crop_size': None,
            'cls_target_on_iou': False,
        }
        self._default.update(curr_default)

        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

        self.train_collate_fn = train_collater
        self.test_collate_fn = test_collater

    def get_train_transform(self):
        mean = get_default_mean()
        std = get_default_std()
        if self.dict_trainer:
            from qd.data_layer.transform import (
                CvImageDecodeDict,
                CvImageBGR2RGBDict,
                DecodeJsonLabelDict,
                RemoveCrowdLabelDict,
                RemoveSmallLabelDict,
                List2NumpyXYXYCLabelDict,
                NpImageNorm01Dict,
                Label2IndexDict,
            )
            labelmap = TSVDataset(self.data).load_labelmap()
            if self.train_crop_sizes:
                all_crop_size = self.train_crop_sizes
            elif self.min_size_range32:
                all_crop_size = tuple(range(self.min_size_range32[0],
                                            self.min_size_range32[1] + 32, 32))
            else:
                all_crop_size = [self.input_sizes[self.net]]
            logging.info(all_crop_size)
            if self.affine_resize == 'RC':
                resizer = RandomResizeCrop(all_crop_size)
            else:
                resizer = Resizer(all_crop_size)
            transform = transforms.Compose([
                CvImageDecodeDict(),
                CvImageBGR2RGBDict(),
                NpImageNorm01Dict(),

                DecodeJsonLabelDict(),
                RemoveCrowdLabelDict(),
                RemoveSmallLabelDict(),
                Label2IndexDict(labelmap),
                List2NumpyXYXYCLabelDict(),

                Normalizer(mean=mean, std=std),
                Augmenter(),
                resizer,
                RemoveUselessKeys(),
            ])
        else:
            logging.info('deprecating')
            transform = transforms.Compose([
                Normalizer(mean=mean, std=std),
                Augmenter(),
                Resizer([self.input_sizes[self.net]])])
            # no need of HWC2CHWTest() since it has been done in train_collater
        return transform

    #def get_train_batch_sampler(self, sampler, stage, start_iter):
        #batch_sampler = super().get_train_batch_sampler(sampler, stage, start_iter)
        #from qd.qd_pytorch import AttachIterationNumberBatchSampler
        #batch_sampler = AttachIterationNumberBatchSampler(batch_sampler,
                #start_iter, self.max_iter)
        #return batch_sampler

    def append_predict_param(self, cc):
        super().append_predict_param(cc)
        if self.test_resize_type is not None:
            cc.append(self.test_resize_type)

    def get_test_transform(self):
        from qd.qd_pytorch import get_default_mean, get_default_std
        mean = get_default_mean()
        std = get_default_std()
        if self.dict_trainer:
            from qd.data_layer.transform import (
                CvImageDecodeDict,
                CvImageBGR2RGBDict,
                DecodeJsonLabelDict,
                RemoveCrowdLabelDict,
                RemoveSmallLabelDict,
                List2NumpyXYXYCLabelDict,
                NpImageNorm01Dict,
                Label2IndexDict,
            )
            labelmap = TSVDataset(self.data).load_labelmap()
            if self.test_crop_size:
                input_size = self.test_crop_size
            else:
                input_size = self.input_sizes[self.net]
            if self.test_resize_type is None:
                resizer = Resizer([input_size])
            else:
                assert self.test_resize_type == 'smart_lower'
                resizer = SmartResizer([input_size])
            transform = transforms.Compose([
                CvImageDecodeDict(),
                CvImageBGR2RGBDict(),
                NpImageNorm01Dict(),

                DecodeJsonLabelDict(),
                RemoveCrowdLabelDict(),
                RemoveSmallLabelDict(),
                Label2IndexDict(labelmap),
                List2NumpyXYXYCLabelDict(),

                Normalizer(mean=mean, std=std),
                resizer,
                HWC2CHWTest(),
                RemoveUselessKeys(),
            ])
        else:
            raise ValueError('deprecated')
            logging.info('deprecating')
            transform = transforms.Compose([
                Normalizer(mean=mean, std=std),
                Resizer([self.input_sizes[self.net]]),
                HWC2CHWTest(),
            ])
        return transform

    def create_dataset_with_transform(self, data, split, stage, labelmap, dataset_type,
                       transform):
        if self.dict_trainer:
            from qd.data_layer.dataset import IODataset, DatasetPlusTransform
            io_set = IODataset(data, split, version=0)
            dataset = DatasetPlusTransform(io_set, transform)
        else:
            logging.info('deprecating')
            dataset = EfficientDetDataset(data,
                                          split,
                                          version=0,
                                          transform=transform)
        return dataset

    def _get_model(self, pretrained, num_class):
        from qd.layers.efficient_det import EfficientDetBackbone
        if self.efficient_net_simple_padding:
            from qd.layers import efficient_det
            efficient_det.g_simple_padding = True
        model = EfficientDetBackbone(num_classes=len(self.labelmap),
                                     compound_coef=self.net,
                                     ratios=self.anchors_ratios,
                                     scales=self.anchors_scales,
                                     prior_prob=self.prior_prob,
                                     adaptive_up=self.adaptive_up,
                                     anchor_scale=self.anchor_scale,
                                     drop_connect_rate=self.drop_connect_rate,
                                     )
        return model

    def _get_criterion(self):
        from qd.layers.efficient_det import FocalLoss
        return FocalLoss(alpha=self.focal_alpha,
                         gamma=self.focal_gamma,
                         cls_loss_type=self.cls_loss_type,
                         reg_loss_type=self.reg_loss_type,
                         smooth_bce_pos=self.smooth_bce_pos,
                         smooth_bce_neg=self.smooth_bce_neg,
                         at_least_1_assgin=self.at_least_1_assgin,
                         neg_iou_th=self.neg_iou_th,
                         pos_iou_th=self.pos_iou_th,
                         cls_weight=self.cls_weight,
                         reg_weight=self.reg_weight,
                         cls_target_on_iou=self.cls_target_on_iou,
                         )

    def combine_model_criterion(self, model, criterion):
        from qd.layers.efficient_det import ModelWithLoss
        model = ModelWithLoss(model, criterion)
        return model

    def get_test_model(self):
        model = super().get_test_model()
        from qd.layers.efficient_det import InferenceModel
        model = InferenceModel(model)
        #if self.dict_trainer:
            #from qd.layers.forward_image_model import ForwardImageModel
            #model = ForwardImageModel(model)
        return model

    def predict_output_to_tsv_row(self, outputs, inputs, **kwargs):
        keys = inputs['key']
        for key, out in zip(keys, outputs):
            if len(out['rois']) == 0:
                rects = []
            else:
                rects = [
                    {
                        'rect': roi.tolist(),
                        'class': self.labelmap[cls],
                        'conf': float(s),
                    }
                    for roi, cls, s in zip(out['rois'].cpu(),
                                           out['class_ids'].cpu(),
                                           out['scores'].cpu())]

            from qd.qd_common import json_dump
            #logging.info('debugging')
            #dataset = TSVDataset('coco2017Full')
            #img = img_from_base64(dataset.seek_by_key(key, split='test')[-1])
            #from qd.process_image import draw_rects, show_image
            #draw_rects(rects, img)
            #show_image(img)
            yield key, json_dump(rects)

    def _get_test_normalize_module(self):
        return

