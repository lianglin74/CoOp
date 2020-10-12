import sys
from tqdm import tqdm
from datetime import datetime
import simplejson as json
from qd.qd_common import ensure_directory
from qd.qd_common import init_logging
from qd.qd_common import write_to_yaml_file
from qd.qd_common import img_from_base64, load_from_yaml_file
from qd.qd_common import worth_create
from qd.qd_common import read_to_buffer
from qd.qd_common import write_to_file
from qd.tsv_io import load_list_file
from qd.qd_common import get_mpi_rank, get_mpi_size
from qd.qd_common import get_mpi_local_rank, get_mpi_local_size
from qd.qd_common import parse_general_args
from qd.qd_common import plot_to_file
from qd.qd_common import ensure_remove_dir
from qd.process_image import is_pil_image
from collections import OrderedDict
from qd.process_tsv import load_key_rects
from qd.process_tsv import hash_sha1
from qd.tsv_io import tsv_writer, tsv_reader
from qd.tsv_io import TSVFile, CompositeTSVFile
from qd.tsv_io import TSVDataset
from shutil import copyfile
import os
import os.path as op
import copy
from pprint import pformat
import logging
import torch
from torch.utils.data import Dataset
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import transforms
import numpy as np
import torchvision
try:
    from itertools import izip as zip
except:
    # python 3
    pass
import time
import re
import glob
from torchvision.transforms import functional as F
import cv2
import math
from qd.qd_common import is_hvd_initialized
import PIL
from qd.torch_common import all_gather_grad_curr
from qd.layers.loss import BCEWithLogitsNegLoss
from qd.data_layer.dataset import TSVSplitProperty
from qd.torch_common import synchronize
from qd.torch_common import ensure_init_process_group
from qd.torch_common import get_master_node_ip
from qd.torch_common import get_aml_mpi_host_names
from qd.torch_common import get_philly_mpi_hosts
from qd.torch_common import torch_save, torch_load
from qd.torch_common import freeze_parameters
from qd.layers.loss import MultiHotCrossEntropyLoss
from qd.layers.loss import multi_hot_cross_entropy
from qd.torch_common import concat_all_gather
from qd.data_layer.transform import ImageTransform2Dict
from qd.data_layer.samplers import AttachIterationNumberBatchSampler
from qd.data_layer.transform import ImageCutout
from qd.data_layer.transform import TwoCropsTransform, IoURandomResizedCrop
from qd.data_layer.transform import TwoCropsTransformX
from qd.torch_common import replace_module



class InputAsDict(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
    def forward(self, data_dict):
        if isinstance(data_dict, torch.Tensor):
            im = data_dict
        else:
            im = data_dict['image']
        return self.module(im)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        from PIL import ImageFilter
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

@torch.no_grad()
def broadcast(x):
    if get_mpi_size() > 1:
        if not is_hvd_initialized():
            torch.distributed.broadcast(x, src=0)
        else:
            from horovod import hvd
            hvd.torch.broadcast(x, src=0)

@torch.no_grad()
def batch_shuffle_ddp(x):
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    if get_mpi_size() > 1:
        # broadcast to all gpus
        broadcast(idx_shuffle)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    idx_this = idx_shuffle.view(num_gpus, -1)[get_mpi_rank()]

    return x_gather[idx_this], idx_unshuffle

@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    idx_this = idx_unshuffle.view(num_gpus, -1)[get_mpi_rank()]

    return x_gather[idx_this]

def detection_evalation(data, split, pred, ovthresh=[0.3,0.4,0.5]):
    from qd.deteval import deteval_iter
    dataset = TSVDataset(data)
    evaluate_file = pred + '.report'
    deteval_iter(
        dataset.iter_data(split, 'label',
                          version=0),
        pred,
        report_file=evaluate_file,
        ovthresh=ovthresh,
    )
    ensure_create_evaluate_meta_file(evaluate_file)

def print_module_param_grad(model):
    info = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            info.append('name = {}, param = {}; grad = {}'.format(
                n,
                p.abs().mean(),
                p.grad.abs().mean() if p.grad is not None else 0,
                ))
    logging.info('\n'.join(info))

def visualize_maskrcnn_input(images, targets, show_box=True):
    import torch
    if hasattr(images, 'tensors'):
        images = images.tensors
    images = images.cpu()
    scale = [0.229, 0.224, 0.225]
    scale = torch.tensor(scale)
    images = images * scale[None, :, None, None]
    mean = torch.tensor([0.485, 0.456, 0.406])
    images = images + mean[None, :, None, None]
    images *= 255.
    images = images[:, [2, 1, 0], :, :]
    images = images.to(torch.uint8)
    images = images.permute(0, 2, 3, 1).numpy()
    from qd.process_image import show_images
    from qd.process_image import draw_rects
    #show_image(images[0, :])
    ims = []
    for i, t in enumerate(targets):
        im = images[i]
        if show_box:
            labels = t.get_field('labels').tolist()
            rects = t.bbox.tolist()
            rects = [{'rect': r, 'class': str(l)} for r, l in zip(rects, labels)]
            im = draw_rects(rects, im, add_label=True)
        ims.append(im)
    rows = int(math.ceil(len(ims) ** 0.5))
    show_images(ims, rows, rows)

def adapt_convbn_weight(weight, running_mean,
        curr_scale, curr_mean):
    # original: input: rgb, 0-1; first conv has no bias. The second layer is BN
    # convert: input should be 0-255 without norm and scale. the first conv's
    # weight and the running mean will be updated. The running mean can be seen
    # as a negated bias

    # make the input as bgr from rgb. note there is no need to update the bias
    # or running mean

    # no need to update bias because bias is only associated with the
    # output channels
    weight = weight[:, [2, 1, 0], :, :]
    curr_scale = torch.tensor(curr_scale)
    weight = weight / (255. * curr_scale[None, :, None, None])

    curr_mean = torch.tensor(curr_mean)
    x = curr_mean / curr_scale
    spatial_dim = weight.shape[2] * weight.shape[3]
    x = x.view((len(x), 1)).repeat((1, spatial_dim)).view((-1, 1))
    running_mean += torch.mm(weight.view((weight.shape[0], -1)), x).view(-1)
    return weight, running_mean

def load_scheduler_state(scheduler, state):
    for k, v in state.items():
        if k in scheduler.__dict__:
            curr = scheduler.__dict__[k]
            from qd.qd_common import float_tolorance_equal
            # if the parameter is about the old scheduling, we ignore it. We
            # prefer the current scheduling parameters, except the last_epoch
            # or some other states which relies on the iteration.
            if k in ['milestones', 'warmup_factor', 'warmup_iters', 'warmup_method',
                'base_lrs', 'gamma', '_last_lr'] or float_tolorance_equal(curr, v):
                continue
            elif k in ['last_epoch', '_step_count']:
                logging.info('updating {} from {} to {}'.format(k,
                    curr, v))
                scheduler.__dict__[k] = v
            else:
                raise NotImplementedError('unknown {}'.format(k))
        else:
            scheduler.__dict__[k] = v

def compare_caffeconverted_vs_pt(pt2, pt1):
    state1 = torch.load(pt1)
    state2 = torch.load(pt2)

    sd1 = state1['state_dict']
    for k in ['region_target.biases', 'region_target.seen_images']:
        sd1[k] = state1[k]
    sd2 = state2['state_dict']

    key_mismatched = []
    key_matched_size_inequal = []
    key_matched_value_equal = []
    key_matched_value_inequal = []
    for k2 in sd2:
        found = False
        k2_in_match = k2
        if k2_in_match in ['extra_conv19.weight', 'extra_conv20.weight']:
            k2_in_match = k2_in_match.replace('.', '/')
            k2_in_match = k2_in_match.replace('weight', 'conv.weight')
        for k1 in sd1:
            if not found:
                if k2_in_match in k1:
                    found = True
                    break
            else:
                assert k2_in_match not in k1
        if not found:
            key_mismatched.append(k2)
            continue
        v2 = sd2[k2].cpu().float()
        v1 = sd1[k1].cpu().float()
        assert len(v1.shape) == len(v2.shape)
        if not all(s1 == s2 for s1, s2 in zip(v1.shape, v2.shape)):
            key_matched_size_inequal.append(k2)
            continue
        d = (v1 - v2).norm()
        s = v1.norm()
        if d <= 0.00001 * s:
            key_matched_value_equal.append((k2, d, s, d / s))
        else:
            key_matched_value_inequal.append((k2, d, s, d / s))
    logging.info(pformat(key_mismatched))
    logging.info('key matched size inequal: \n{}'.format(pformat(key_matched_size_inequal)))
    logging.info('key matched value equal: \n {}'.format(pformat(key_matched_value_equal)))
    logging.info('key matched value not equal: \n{}'.format(
        pformat(key_matched_value_inequal)))

class ListCollator(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        return list(zip(*batch))

class OrderedSampler(torch.utils.data.Sampler):
    def __init__(self, idx):
        super(OrderedSampler, self).__init__(None)
        self.idx = list(idx)

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)

class DictTransformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, dict_data):
        for t in self.transforms:
            dict_data = t(dict_data)
        return dict_data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

def get_image_size(image):
    if isinstance(image, np.ndarray):
        return image.shape[1], image.shape[0]
    elif is_pil_image(image):
        return image.size
    else:
        raise NotImplementedError

class DictTransformAffineResize(object):
    def __init__(self, out_sizes,
            degrees=(-10, 10),
            scale=(.9, 1.1),
            shear=(-2, 2),
            border_value=(0., 0., 0.),
            ):
        self.out_sizes = [
                (s, s)
                if not (isinstance(s, tuple) or isinstance(s, list)) else s
                for s in out_sizes]
        self.degrees = degrees
        self.scale = scale
        self.shear = shear
        self.border_value = border_value

    def gen_out_size(self, dict_data):
        size = self.out_sizes[dict_data['iteration'] % len(self.out_sizes)]
        return size

    def gen_angle(self, dict_data):
        return random.random() * (self.degrees[1] - self.degrees[0]) + self.degrees[0]

    def gen_scale(self):
        return random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]

    def gen_shear_angle(self):
        return random.random() * (self.shear[1] - self.shear[0]) + self.shear[0]

    def gen_src2dst_info(self, dict_data):
        origin_image = dict_data['image']
        origin_width, origin_height = get_image_size(origin_image)

        # 1. change the scale, so that it is aligned to the network input size
        out_width, out_height = self.gen_out_size(dict_data)
        if random.random() < 0.5:
            align_scale = 1. * out_width / origin_width
        else:
            align_scale = 1. * out_height / origin_height
        M_align = np.eye(3)
        M_align[0, 0] = align_scale
        M_align[1, 1] = align_scale
        align_width = align_scale * origin_width
        align_height = align_scale * origin_height

        # 2. rotation and scale
        M_rotate = np.eye(3)
        random_angle = self.gen_angle(dict_data)
        random_scale = self.gen_scale()
        M_rotate[:2] = cv2.getRotationMatrix2D(angle=random_angle,
                center=(align_width / 2, align_height / 2), scale=random_scale)

        # 3. Shear
        M_shear = np.eye(3)
        shear_x_angle = self.gen_shear_angle()
        M_shear[0, 1] = math.tan(shear_x_angle * math.pi / 180)
        shear_y_angle = self.gen_shear_angle()
        M_shear[1, 0] = math.tan(shear_y_angle * math.pi / 180)

        # get the min_x, min_y, max_x, max_y for the whole image
        M = M_shear @ M_rotate @ M_align
        corners = np.ones((4, 3))
        corners[0, :2] = (0, 0)
        corners[1, :2] = (origin_width, 0)
        corners[2, :2] = (0, origin_height)
        corners[3, :2] = (origin_width, origin_height)
        aug_corners = corners @ M.T # 4x3
        aug_min_x = aug_corners[:, 0].min()
        aug_min_y = aug_corners[:, 1].min()
        aug_max_x = aug_corners[:, 0].max()
        aug_max_y = aug_corners[:, 1].max()

        # 4. random crop on the augmented images
        aug_width = aug_max_x - aug_min_x
        aug_height = aug_max_y - aug_min_y
        delta_x = random.random() * (aug_width - out_width)
        delta_y = random.random() * (aug_height - out_height)
        origin_x = aug_min_x + delta_x
        origin_y = aug_min_y + delta_y
        M_crop = np.eye(3)
        M_crop[0, 2] = -origin_x
        M_crop[1, 2] = -origin_y

        M = M_crop @ M

        return {'matrix': M,
                'out_width': out_width,
                'out_height': out_height,
                'angle': random_angle}

    def wrap_box(self, targets, info):
        '''
        targets: Nx4 (x1y1x2y2)
        '''
        n = targets.shape[0]
        points = targets
        M = info['matrix']
        width, height = info['out_width'], info['out_height']

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # apply angle-based reduction
        radians = info['angle'] * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        x1 = np.clip(xy[:,0], 0, width)
        y1 = np.clip(xy[:,1], 0, height)
        x2 = np.clip(xy[:,2], 0, width)
        y2 = np.clip(xy[:,3], 0, height)
        boxes = np.concatenate((x1, y1, x2, y2)).reshape(4, n).T
        return boxes

    def wrap_image(self, image, info):
        if isinstance(image, np.ndarray):
            def gen_warp_image_flag():
                if random.random() < 0.3:
                    return cv2.INTER_NEAREST
                else:
                    return cv2.INTER_LINEAR
            out_image = cv2.warpPerspective(image,
                    info['matrix'],
                    dsize=(info['out_width'], info['out_height']),
                    flags=gen_warp_image_flag(),
                    borderValue=self.border_value)  # BGR order borderValue
        elif is_pil_image(image):
            def gen_warp_image_flag():
                # PIL.Image.LANCZOS is not valid for transform
                r = random.random()
                if r < 0.3:
                    return PIL.Image.NEAREST
                elif r > 0.7:
                    return PIL.Image.BICUBIC
                else:
                    return PIL.Image.BILINEAR
            assert abs(info['matrix'][2, 0]) < 1e-5
            assert abs(info['matrix'][2, 1]) < 1e-5
            out_image = image.transform(size=(info['out_width'], info['out_height']),
                    method=PIL.Image.AFFINE,
                    data=np.linalg.inv(info['matrix']).flatten()[:6],
                    resample=gen_warp_image_flag())
        return out_image

    def __call__(self, dict_data):
        targets = dict_data['rects']
        info = self.gen_src2dst_info(dict_data)
        if targets is not None:
            assert targets.mode == 'xyxy'
            boxes = self.wrap_box(targets.bbox, info)
            targets.bbox = boxes
            valid_box = (boxes[:, 2] - boxes[:, 0] > 0) & (
                    boxes[:, 3] - boxes[:, 1] > 0)
            targets = targets[valid_box]
            targets.size = (info['out_width'], info['out_height'])
            dict_data['rects'] = targets

        image = dict_data['image']
        image = self.wrap_image(image, info)
        dict_data['image'] = image
        return dict_data

class DictTransformMaskRandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, dict_data):
        image, target = dict_data['image'], dict_data['rects']
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
            dict_data['image'] = image
            dict_data['rects'] = target
        return dict_data

class DictTransformResizeCrop(object):
    '''
    - only used for training
    - it first randomly select a crop size from all_crop_size
    - then randomly resize the input image to a target size. Let (a, b) be the
      original size; (c, d) be the crop  size. The scalar factor is randomly
      chosen from min(c/a, d/b) to max(c/a, d/b).
    - finally, we randomly crop a sub region.
    '''
    def __init__(self, all_crop_size, size_mode='random'):
        self.all_crop_size = all_crop_size
        self.size_mode = size_mode

    def get_size(self, image_size, iteration):
        if self.size_mode == 'random':
            w, h = image_size
            crop_size = self.all_crop_size[iteration % len(self.all_crop_size)]
            rw = 1. * crop_size / w
            rh = 1. * crop_size / h
            min_s = min(rw, rh)
            max_s = max(rw, rh)
            target_s = random.random() * (max_s - min_s) + min_s
            target_w = int(target_s * w)
            target_h = int(target_s * h)
            crop_w = crop_h = crop_size
        elif self.size_mode == 'max':
            w, h = image_size
            crop_size = self.all_crop_size[iteration % len(self.all_crop_size)]
            target_s = 1. * crop_size / max(w, h)
            target_w = int(target_s * w)
            target_h = int(target_s * h)
            crop_h = target_h
            crop_w = target_w
        elif self.size_mode.startswith('max_cut'):
            w, h = image_size
            # range(256:768)
            crop_size = self.all_crop_size[iteration % len(self.all_crop_size)]
            cut_size = int(self.size_mode[len('max_cut'):])
            target_s = 1. * crop_size / max(w, h)
            target_w = int(target_s * w)
            target_h = int(target_s * h)
            crop_h = min(target_h, cut_size)
            crop_w = min(target_w, cut_size)
        elif self.size_mode.startswith('mm_cut'):
            w, h = image_size
            # range(256:768)
            crop_size = self.all_crop_size[iteration % len(self.all_crop_size)]
            cut_size = int(self.size_mode[len('mm_cut'):])
            target_s = 1. * crop_size / min(w, h)
            target_w = int(target_s * w)
            target_h = int(target_s * h)
            crop_h = min(target_h, cut_size)
            crop_w = min(target_w, cut_size)
        return (target_h, target_w, crop_h, crop_w)

    def __call__(self, dict_data):
        num_trial = 50
        for idx_trial in range(num_trial):
            image, target = dict_data['image'], dict_data['rects']
            # we should not combine resize and crop here, since the resize will do
            # some anti-aliassing trick which helps if we need to downsample to a
            # large factor.
            target_h, target_w, crop_h, crop_w = self.get_size(image.size, dict_data['iteration'])

            def random_offset(origin, crop):
                if origin > crop:
                    return random.random() * (origin - crop)

            top = int(random.random() * (target_h - crop_h))
            left = int(random.random() * (target_w - crop_w))

            target = target.resize((target_w, target_h))
            target = target.crop((left, top, left + crop_w, top + crop_h))
            target = target.clip_to_image(remove_empty=True)

            if len(target) > 0:
                break

        image = F.resize(image, (target_h, target_w))
        image = image.crop((left, top, left + crop_w, top + crop_h))
        dict_data['image'] = image
        dict_data['rects'] = target
        return dict_data

class DictTransformMaskResize(object):
    '''
    maskrcnn-style resize
    '''
    def __init__(self, min_size,
                 max_size,
                 depends_on_iter=False,
                 treat_min_as_max=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.depends_on_iter = depends_on_iter
        self.treat_min_as_max = treat_min_as_max
        assert depends_on_iter, 'no need to set it as False'

    def gett_size_min_as_max(self, image_size, iteration):
        w, h = image_size
        max_size = self.min_size[iteration % len(self.min_size)]

        max_original_size = float(max((w, h)))
        scale = max_size / max_original_size

        return (int(h * scale), int(w * scale))

    def get_size(self, image_size, iteration):
        w, h = image_size
        if self.depends_on_iter:
            size = self.min_size[iteration % len(self.min_size)]
        else:
            size = random.choice(self.min_size)
        max_size = self.max_size

        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, dict_data):
        image, target = dict_data['image'], dict_data['rects']
        if self.treat_min_as_max:
            size = self.gett_size_min_as_max(image.size, dict_data['iteration'])
        else:
            size = self.get_size(image.size, dict_data['iteration'])
        image = F.resize(image, size)
        target = target.resize(image.size)
        dict_data['image'] = image
        dict_data['rects'] = target
        return dict_data

class DictTransformMaskToTensor(object):
    def __call__(self, dict_data):
        dict_data['image'] = F.to_tensor(dict_data['image'])
        return dict_data

class DictTransformMaskColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 adaptive=None,
                 ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.adaptive = None if adaptive == '' else adaptive

    def __call__(self, dict_data):
        if self.brightness in [0, None] and \
                self.contrast in [0, None] and \
                self.saturation in [0, None] and \
                self.hue in [0, None]:
            # used for testing mode
            return dict_data

        if self.adaptive is None:
            brightness = self.brightness
            contrast = self.contrast
            saturation = self.saturation
            hue = self.hue
        elif self.adaptive.startswith('sin'):
            exp_factor = self.adaptive[len('sin'):]
            exp_factor = 1 if exp_factor == '' else float(exp_factor)
            theta = 1. * dict_data['iteration'] / dict_data['max_iter'] * math.pi
            sin_th = math.sin(theta) ** exp_factor
            brightness = self.brightness * sin_th
            contrast = self.contrast * sin_th
            saturation = self.saturation * sin_th
            hue = self.hue * sin_th
        color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)
        dict_data['image'] = color_jitter(dict_data['image'])
        return dict_data

class DictTransformMaskNormalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, dict_data):
        image = dict_data['image']
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        dict_data['image'] = image
        return dict_data

class DictTransformCVImageBGR2RGB(object):
    def __call__(self, dict_data):
        im = dict_data['image']
        dict_data['image'] = im[:, :, [2, 1, 0]]
        return dict_data

class DictTransformCVImageToTensor(object):
    def __call__(self, dict_data):
        im = dict_data['image']
        dict_data['image'] = torchvision.transforms.functional.to_tensor(im)
        return dict_data

class DictTransformLabelMapIndex():
    def __init__(self, data):
        self.labelmap = TSVDataset(data).load_labelmap()
        self.label2idx = {l: i for i, l in enumerate(self.labelmap)}

    def __call__(self, dict_data):
        dict_data['rect'] = [r for r in dict_data['rects'] if
            r['class'] in self.label2idx]
        for r in dict_data['rects']:
            r['class_idx'] = self.label2idx[r['class']]
        return dict_data

class DictTransformCreateLabelCoordinatesTensor():
    def __call__(self, dict_data):
        class_idx = [r['class_idx'] for r in dict_data['rects']]
        coordinates = [r['rect'] for r in dict_data['rects']]
        class_idx = torch.tensor(class_idx).float().view((-1, 1))
        coordinates = torch.tensor(coordinates)
        dict_data['label_coordinates'] = torch.cat((class_idx, coordinates), dim=1)
        return dict_data

class DictTransformNormalizeRect():
    def __call__(self, dict_data):
        h, w = dict_data['h'], dict_data['w']
        for r in dict_data['rects']:
            x1, y1, x2, y2 = r['rect']
            r['rect'] = [1. * x1 / w, 1. * y1 / h, 1. * x2 / w, 1. * y2 / h]
        return dict_data

class TSVDatasetPt(Dataset):
    def __init__(self, data, split, version=0, cache_policy=None,
            transforms=None):
        self.hw = TSVSplitProperty(data, split, 'hw', version, cache_policy)
        self.image = TSVSplitProperty(data, split, t=None, cache_policy=cache_policy)
        self.label = TSVSplitProperty(data, split, t='label', version=version,
                cache_policy=cache_policy)
        self.transforms = transforms

    def __getitem__(self, index):
        image_key, _, str_im = self.image[index]
        im = img_from_base64(str_im)

        label_key, str_rects = self.label[index]
        rects = json.loads(str_rects)

        hw_key, str_hw = self.hw[index]
        h, w = map(int, str_hw.split(' '))

        dict_data = {
                'key': label_key,
                'image': im,
                'rects': rects,
                'original_height': h,
                'original_width': w,
                }
        if self.transforms:
            dict_data = self.transforms(dict_data)
        return dict_data

class TSVSplit(Dataset):
    '''
    prefer to use TSVSplitProperty, which is more general. One example is to
    read the hw property
    '''
    def __init__(self, data, split, version=0, cache_policy=None):
        self.data = data
        self.split = split
        self.tsv = TSVSplitProperty(data, split, t=None, version=version,
                cache_policy=cache_policy)
        self.label_tsv = TSVSplitProperty(data, split, t='label',
                version=version, cache_policy=cache_policy)

    def __getitem__(self, index):
        str_image = self.tsv[index][-1]
        key, str_label = self.label_tsv[index]
        return key, str_label, str_image

    def __len__(self):
        return len(self.tsv)

class DictDAPlacing(object):
    def __init__(self, all_crop_size, max_grid):
        self.all_crop_size = [c if
                isinstance(c, list) or isinstance(c, tuple)
                else (c, c)
                for c in all_crop_size]
        self.max_grid = (max_grid, max_grid)
        self.debug = True

    def _get_grid(self):
        grid_rows = random.randint(1, self.max_grid[0])
        grid_cols = random.randint(max(1, grid_rows // 2),
                                   min(grid_rows * 2, self.max_grid[1]))
        if random.random() < 0.5:
            t = grid_rows
            grid_rows = grid_cols
            grid_cols = t

        return grid_rows, grid_cols

    def __call__(self, dict_data):
        crop_size = self.all_crop_size[dict_data['iteration'] % len(self.all_crop_size)]
        canvas_transform = self.get_transform(crop_size)
        # take the current image as the canvas
        canvas = dict_data['image']
        canvas = canvas_transform(canvas)

        grid_rows, grid_cols = self._get_grid()

        canvas_width, canvas_height = crop_size

        grid_width = canvas_width // grid_cols
        grid_height = canvas_height // grid_rows
        rects = []
        num_total = len(dict_data['dataset'])
        for i in range(grid_rows):
            for j in range(grid_cols):
                scale = random.random() * (0.9 - 0.5) + 0.5
                image_width = int(grid_width * scale)
                image_height = int(grid_height * scale)

                region_transform = self.get_transform((image_width,
                    image_height))

                offset_w = (int(random.random() * (grid_width - image_width)) +
                    j * grid_width)

                offset_h = (int(random.random() * (grid_height - image_height)) +
                        i * grid_height)
                random_idx = random.randint(0, num_total - 1)
                im, anno = dict_data['dataset'].get_image_ann(random_idx)
                assert len(anno) == 1
                im = region_transform(im)
                rect = (offset_w, offset_h,
                    offset_w + image_width, offset_h + image_height)
                rects.append({'rect': rect, 'class': anno[0]['class']})
                canvas.paste(im, rect)
        boxes = torch.tensor([r['rect'] for r in rects]).reshape(-1, 4)
        from maskrcnn_benchmark.structures.bounding_box import BoxList
        boxlist = BoxList(boxes, image_size=(canvas_width, canvas_height), mode='xyxy')
        labels = torch.tensor([
            dict_data['dataset'].label_to_idx[r['class']] + 1
            for r in rects])
        boxlist.add_field('labels', labels)
        dict_data['image'] = canvas
        dict_data['rects'] = boxlist
        return dict_data

    def get_transform(self, crop_size):
        import torchvision.transforms as transforms
        if isinstance(crop_size, int):
            ratio = (3. / 4., 4. / 3.)
        else:
            # crop_size: w, h
            crop_w, crop_h = crop_size
            anchor_ratio = 1. * crop_w / crop_h
            ratio = (3. / 4 * anchor_ratio, 4. / 3 * anchor_ratio)
        all_trans = [
            transforms.RandomResizedCrop(scale=(0.25, 1.0),
                ratio=ratio,
                size=crop_size[::-1]),
            transforms.RandomHorizontalFlip(),
            ]
        data_augmentation = transforms.Compose(all_trans)
        return data_augmentation

class TSVSplitImage(TSVSplit):
    def __init__(self, data, split, version, transform=None,
            cache_policy=None, labelmap=None):
        super(TSVSplitImage, self).__init__(data, split, version,
                cache_policy)
        self.transform = transform
        # load the label map
        dataset = TSVDataset(data)
        if labelmap is None:
            labelmap = load_list_file(dataset.get_labelmap_file())
        elif type(labelmap) is str:
            labelmap = load_list_file(labelmap)
        assert type(labelmap) is list
        self.label_to_idx = {l: i for i, l in enumerate(labelmap)}

    def get_keys(self):
        return [self.label_tsv[i][0] for i in range(len(self.label_tsv))]

    def __getitem__(self, index):
        key, str_label, str_im = super(TSVSplitImage, self).__getitem__(index)
        img = img_from_base64(str_im)
        if self.transform is not None:
            img = self.transform(img)
        label = self._tsvcol_to_label(str_label)
        # before it returns 2 elements. now it is 3. some code might be broken,
        # adjust it accoordingly. The reason is that we want to save the
        # prediction result for each row. The key is required to identify which
        # image it is.
        return img, label, key

    def get_num_pos_labels(self):
        return len(self.label_to_idx)

    def _tsvcol_to_label(self, col):
        try:
            idx = int(col)
        except:
            info = json.loads(col)
            idx = self.label_to_idx.get(info[0]['class'], -1)
        label = torch.from_numpy(np.array(idx, dtype=np.int))
        return label

class TSVSplitImageDict(TSVSplitImage):
    def __init__(self, data, split, version, transform=None,
            cache_policy=None, labelmap=None):
        super().__init__(data, split, version, transform, cache_policy,
                         labelmap)

    def __getitem__(self, data_dict):
        result = {}
        if isinstance(data_dict, dict):
            index = data_dict['idx']
            result.update(data_dict)
        else:
            index = data_dict
        key, str_label, str_im = super(TSVSplitImage, self).__getitem__(index)
        img = img_from_base64(str_im)
        label = self._tsvcol_to_label(str_label)
        data = {
            'index': index,  # this is for backward compatibility
            'idx': index,
            'key': key,
            'image': img,
            'label': label,
        }
        result.update(data)
        if self.transform is not None:
            result = self.transform(result)
        return result

class TSVSplitImageSoftAssign(object):
    def __init__(self, data, split, temperature, transform, num_kmeans=None):
        self.image_tsv = TSVSplitProperty(data, split, t=None)
        self.feature_tsv = TSVSplitProperty(data, split, t='feature_ptr')
        self.feature_tsv = TSVFile(self.feature_tsv[0][0])
        self.data = data
        self.split = split
        self.transform = transform
        self._centers = None
        self.temperature = temperature
        if num_kmeans in [0, None]:
            num_kmeans = 1
        self.num_kmeans = num_kmeans

    @property
    def centers(self):
        if self._centers is None:
            all_centers = []
            for v in range(self.num_kmeans):
                tsv = TSVSplitProperty(
                    self.data,
                    self.split,
                    t='kmeans_center',
                    version=v)
                all_feat = []
                for i in range(len(tsv)):
                    row = tsv[i]
                    feat = [float(r) for r in row]
                    all_feat.append(feat)
                all_centers.append(torch.tensor(all_feat))
            self._centers = all_centers
        return self._centers

    def __getitem__(self, idx):
        image_row = self.image_tsv[idx]
        im = img_from_base64(image_row[-1])
        if self.transform is not None:
            im = self.transform(im)
        feat_row = self.feature_tsv[idx]
        assert feat_row[0] == image_row[0]
        feature = json.loads(feat_row[1])[0]['feature']
        feature = torch.tensor(feature)
        feature = torch.nn.functional.normalize(feature, dim=0)
        sims = [torch.matmul(c, feature.view(-1, 1)) / self.temperature
                for c in self.centers]
        sim = torch.cat(sims, dim=1)
        label = torch.nn.functional.softmax(sim.T, dim=1).squeeze()
        return im, label, image_row[0]

    def __len__(self):
        return len(self.image_tsv)

class TSVSplitCropImage(TSVSplitImage):
    def __getitem__(self, index):
        key, str_label, str_im = super(TSVSplitImage, self).__getitem__(index)
        img = img_from_base64(str_im)
        rects = json.loads(str_label)
        assert len(rects) == 1, (key, len(rects))
        rect = rects[0]
        x0, y0, x1, y1 = map(int, rect['rect'])
        h, w = img.shape[:2]
        y0 = max(0, min(y0, h))
        y1 = max(0, min(y1, h))
        x1 = max(0, min(x1, w))
        x0 = max(0, min(x0, w))
        assert y1 > y0 and x1 > x0, key
        img = img[y0:y1, x0:x1, :]
        str_label = rect['class']
        if self.transform is not None:
            img = self.transform(img)
        label = self.label_to_idx.get(rect['class'], -1)
        return img, label, key

class TSVSplitImageMultiLabel(TSVSplitImage):
    def __init__(self, data, split, version, transform=None,
            cache_policy=None, labelmap=None):
        super(TSVSplitImageMultiLabel, self).__init__(data, split, version,
                transform, cache_policy, labelmap)

    def _tsvcol_to_label(self, col):
        rects = json.loads(col)
        label = torch.zeros(len(self.label_to_idx))
        if type(rects) is int:
            label[rects] = 1
        else:
            all_cls = set(r['class'] for r in rects)
            # if it starts with -, it means negative
            all_idx = [self.label_to_idx[c] for c in all_cls if
                    not c.startswith('-') and c in self.label_to_idx]
            label[all_idx] = 1
        return label

class TSVSplitImageMultiLabelNeg(TSVSplitImage):
    def __init__(self, data, split, version, transform=None,
            cache_policy=None, labelmap=None):
        super(TSVSplitImageMultiLabelNeg, self).__init__(data, split, version,
                transform, cache_policy, labelmap)

    def _tsvcol_to_label(self, col):
        rects = json.loads(col)
        label = torch.zeros(len(self.label_to_idx))
        label[:] = -1
        if type(rects) is int:
            label[rects] = 1
        else:
            all_cls = set(r['class'] for r in rects)
            # if it starts with -, it means negative
            all_idx = [self.label_to_idx[c] for c in all_cls if
                    not c.startswith('-')]
            label[all_idx] = 1
            all_idx = [self.label_to_idx[c[1:]] for c in all_cls if
                    c.startswith('-')]
            label[all_idx] = 0
        return label

def calculate_ap_by_true_list(corrects, total):
    if total == 0:
        return 0
    corrects = corrects.cpu()
    total = total.cpu()
    precision = (1. * corrects.cumsum(dim=0)) / torch.range(1, len(corrects))
    return torch.sum(precision * corrects) / total

class TagMetric(object):
    def __init__(self):
        self.all_pred = []
        self.all_gt = []

    def update(self, pred, gt):
        '''
        pred: sample x number_of_tag: probability
        gt: sample x number_of_tag. 1/0
        '''
        self.all_pred.append(pred)
        self.all_gt.append(gt)

    def summarize(self):
        pred = torch.cat(self.all_pred)
        gt = torch.cat(self.all_gt)
        _, idx = torch.sort(pred, 0, descending=True)
        num_tag = gt.shape[-1]
        aps = torch.zeros(num_tag)
        for t in range(num_tag):
            is_correct = gt[idx[:, t], t]
            ap = calculate_ap_by_true_list(is_correct.cpu(), torch.sum(is_correct))
            aps[t] = ap
        return aps

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BGR2RGB(object):
    def __call__(self, im):
        return im[:, :, [2, 1, 0]]

class FixedPositionCrop(object):
    def __init__(self, pos, size):
        self.pos = pos
        assert isinstance(size, int)
        self.size = size

        if self.pos == 'top_left':
            self.get_pos = self.get_pos_for_top_left_crop
        elif self.pos == 'top_right':
            self.get_pos = self.get_pos_for_top_right_crop
        elif self.pos == 'bottom_left':
            self.get_pos = self.get_pos_for_bottom_left_crop
        else:
            assert self.pos == 'bottom_right'
            self.get_pos = self.get_pos_for_bottom_right_crop

    def get_pos_for_top_left_crop(self, img):
        return (0, 0)
    def get_pos_for_top_right_crop(self, img):
        w, h = img.size
        return (0, w - self.size)
    def get_pos_for_bottom_left_crop(self, img):
        w, h = img.size
        return (h - self.size, 0)
    def get_pos_for_bottom_right_crop(self, img):
        w, h = img.size
        return (h - self.size, w - self.size)

    def __call__(self, img):
        crop_top, crop_left = self.get_pos(img)
        from torchvision.transforms.functional import crop
        img = crop(img, crop_top, crop_left, self.size, self.size)
        return img

def create_crop_transform(crop_position, crop_size):
    if crop_position is None:
        crop_transform = transforms.CenterCrop(crop_size)
    else:
        crop_transform = FixedPositionCrop(crop_position, crop_size)
    return crop_transform

def get_test_transform(
    bgr2rgb=False,
    resize_size=256,
    crop_size=224,
    crop_position=None):
    normalize = get_data_normalize()
    all_trans = []
    if bgr2rgb:
        all_trans.append(BGR2RGB())
    crop_transform = create_crop_transform(crop_position, crop_size)
    all_trans.extend([
            transforms.ToPILImage(),
            transforms.Resize(resize_size),
            crop_transform,
            transforms.ToTensor(),
            normalize,
            ])
    return transforms.Compose(all_trans)

def get_train_transform(bgr2rgb=False, crop_size=224):
    normalize = get_data_normalize()
    totensor = transforms.ToTensor()
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    all_trans = []
    if bgr2rgb:
        all_trans.append(BGR2RGB())
    all_trans.extend([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(crop_size),
        color_jitter,
        transforms.RandomHorizontalFlip(),
        totensor,
        normalize,])
    data_augmentation = transforms.Compose(all_trans)
    return data_augmentation

def get_default_mean():
    return [0.485, 0.456, 0.406]

def get_default_std():
    return [0.229, 0.224, 0.225]

def get_data_normalize():
    normalize = transforms.Normalize(mean=get_default_mean(),
                                     std=get_default_std())
    return normalize

class MCEBLoss(nn.Module):
    # multi-hot cross entropy loss with background
    def __init__(self):
        super(MCEBLoss, self).__init__()

    def forward(self, feature_with_bkg, target):
        is_bkg = target.sum(dim=1) == 0
        target_with_bkg = torch.cat([is_bkg[:, None].float(), target.float()], dim=1)
        return multi_hot_cross_entropy(feature_with_bkg,
                target_with_bkg)

class IBCEWithLogitsNegLoss(nn.Module):
    def __init__(self, neg, pos,
            neg_pos_ratio=None,
            ignore_hard_neg_th=None,
            ignore_hard_pos_th=None,
            minus_2_weight=1.,
            correct_loss=None):
        super(IBCEWithLogitsNegLoss, self).__init__()
        self.neg = neg
        self.pos = pos
        self.neg_pos_ratio = neg_pos_ratio
        self.num_called = 0
        self.display = 100
        self.ignore_hard_neg_th = ignore_hard_neg_th
        self.ignore_hard_pos_th = ignore_hard_pos_th
        self.eps = 1e-5
        # -2 means it is a hard negative sample and we can pose more weight on
        # it
        self.minus_2_weight = minus_2_weight
        # correct_loss should always be 1, we keep it here only for back
        # campatibility
        self.correct_loss = correct_loss

    def forward(self, feature, target, weight=None, reduce=True):
        # currently, we do not support to customize teh weight. This field is
        # only used for mmdetection, where teh weight is used.
        # meanwhile, we only support reduce=True here
        assert reduce
        debug_infos = []
        print_debug = (self.num_called % self.display) == 0

        pos_position = target == 1
        # we assume it is a class level hard negative. That is, if any one of
        # the rows has -2 in that column; that colum will have -2 for all rows.
        minus_2_pos = ((target == -2).sum(dim=0, keepdim=True) > 0).expand_as(target)
        neg_position = ((target == 0) | minus_2_pos)
        ignore_position = target == -1

        target = target.float()

        weight = torch.ones_like(target)
        weight[minus_2_pos] = self.minus_2_weight
        target[minus_2_pos] = 0

        weight[ignore_position] = 0
        sig_feature = torch.sigmoid(feature)
        weight[(neg_position) & (sig_feature <= self.neg)] = 0
        weight[(pos_position) & (sig_feature >= self.pos)] = 0

        if print_debug:
            debug_infos.append('')
            debug_infos.append('ignore easy neg = {}; '.format(
                ((neg_position) & (sig_feature <= self.neg)).sum()))
            debug_infos.append('ignore easy pos = {}; '.format(
                ((pos_position) & (sig_feature >= self.pos)).sum()))
            if self.minus_2_weight != 1:
                debug_infos.append('#minus_2_pos = {}'.format(minus_2_pos.sum()))

        if self.ignore_hard_neg_th is not None and self.ignore_hard_neg_th <= 1:
            weight[(neg_position & (sig_feature >= self.ignore_hard_neg_th))] = 0
            if print_debug:
                debug_infos.append('ignore hard neg = {}; '.format((neg_position & (sig_feature >=
                    self.ignore_hard_neg_th)).sum()))

        if self.ignore_hard_pos_th is not None and self.ignore_hard_pos_th >= 0:
            weight[(pos_position & (sig_feature <= self.ignore_hard_pos_th))] = 0
            if print_debug:
                debug_infos.append('ignore hard pos = {}; '.format((pos_position &
                    (sig_feature <= self.ignore_hard_pos_th)).sum()))

        loss_weight = 1.
        if self.neg_pos_ratio is not None:
            pos_position_in_loss = pos_position & (weight > 0)
            neg_position_in_loss = neg_position & (weight > 0)
            if self.neg_pos_ratio > 0:
                num_pos_in_loss = weight[pos_position_in_loss].sum()
                num_neg_in_loss = weight[neg_position_in_loss].sum()
                if num_pos_in_loss != 0 and num_neg_in_loss != 0:
                    num_pos_in_loss = num_pos_in_loss.float()
                    num_neg_in_loss = num_neg_in_loss.float()
                    pos_weight = (num_pos_in_loss + num_neg_in_loss) / (
                            num_pos_in_loss * (self.neg_pos_ratio + 1))
                    neg_weight = (num_pos_in_loss + num_neg_in_loss) * self.neg_pos_ratio / (
                            num_neg_in_loss * (self.neg_pos_ratio + 1))
                    weight[pos_position_in_loss] *= pos_weight
                    weight[neg_position_in_loss] *= neg_weight
                elif num_pos_in_loss == 0 and num_neg_in_loss != 0:
                    loss_weight = 1. * self.neg_pos_ratio / (1 + self.neg_pos_ratio)
                elif num_pos_in_loss != 0 and num_neg_in_loss == 0:
                    loss_weight = 1. / (1 + self.neg_pos_ratio)
                if not self.correct_loss:
                    loss_weight = 1.
            elif self.neg_pos_ratio == -1:
                weight[pos_position_in_loss] = (target - sig_feature)[pos_position_in_loss]
        else:
            pos_position_in_loss, neg_position_in_loss = None, None

        weight_sum = torch.sum(weight)

        if print_debug:
            if pos_position_in_loss is None:
                pos_position_in_loss = pos_position & (weight > 0)
                neg_position_in_loss = neg_position & (weight > 0)
                num_pos_in_loss = pos_position_in_loss.sum()
                num_neg_in_loss = neg_position_in_loss.sum()
            if num_pos_in_loss == 0:
                avg_pos_in_loss = 0
            else:
                avg_pos_in_loss = sig_feature[pos_position_in_loss].sum() / num_pos_in_loss
            if num_neg_in_loss == 0:
                avg_neg_in_loss = 0
            else:
                avg_neg_in_loss = sig_feature[neg_position_in_loss].sum() / num_neg_in_loss
            debug_infos.append('sum of weight in pos position = {}; '
                    'sum of weight in neg position = {}'.format(
                    weight[pos_position].sum(),
                    weight[neg_position].sum(),
            ))
            debug_infos.append(('#pos_in_loss = {}, avg_pos_in_loss = {}, '
                          '#neg_in_loss = {}, avg_neg_in_loss = {}, '
                          ).format(
                          num_pos_in_loss, avg_pos_in_loss,
                          num_neg_in_loss, avg_neg_in_loss,
                          ))
            debug_infos.append(('ignore_hard_neg_th = {}, '
                    'ignore_hard_pos_th = {}, '
                    'neg = {}, '
                    'pos = {}, '
                    'minus_2_weight = {}').format(
                    self.ignore_hard_neg_th, self.ignore_hard_pos_th,
                    self.neg, self.pos, self.minus_2_weight))
            logging.info('\n'.join(debug_infos))

        self.num_called += 1

        if weight_sum == 0:
            loss = torch.tensor(0, device=feature.device, dtype=feature.dtype, requires_grad=True)
            # Do not use the following, since .backward may give the error of
            # element 0 of tensors does not require grad and does not have a
            # grad_fn
            #return torch.zeros((), device=feature.device, dtype=feature.dtype)
        else:
            criterion = nn.BCEWithLogitsLoss(weight, reduction='sum')
            loss = criterion(feature, target)
            loss = loss_weight * torch.sum(loss) / (weight_sum + self.eps)
        return loss

def mean_remove(x):
    assert x.dim() == 2
    return x - x.mean(dim=0)

def load_latest_parameters(folder):
    yaml_file = get_latest_parameter_file(folder)
    logging.info('using {}'.format(yaml_file))
    param = load_from_yaml_file(yaml_file)
    return param

def get_latest_parameter_file(folder):
    yaml_pattern = op.join(folder,
            'parameters_*.yaml')
    yaml_files = glob.glob(yaml_pattern)
    assert len(yaml_files) > 0, folder
    def parse_time(f):
        m = re.search('.*parameters_(.*)\.yaml', f)
        t = datetime.strptime(m.group(1), '%Y_%m_%d_%H_%M_%S')
        return t
    times = [parse_time(f) for f in yaml_files]
    fts = [(f, t) for f, t in zip(yaml_files, times)]
    fts.sort(key=lambda x: x[1], reverse=True)
    yaml_file = fts[0][0]
    return yaml_file

def parse_epoch(s):
    s = op.basename(s)
    results = re.match('^model_iter_(.*)e\.pth.tar$', s)
    return int(results.groups()[0])

def save_parameters(param, folder):
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    write_to_yaml_file(param, op.join(folder,
        'parameters_{}.yaml'.format(time_str)))
    # save the env parameters
    # convert it to dict for py3
    write_to_yaml_file(dict(os.environ), op.join(folder,
        'env_{}.yaml'.format(time_str)))

def ensure_create_evaluate_meta_file(evaluate_file):
    result = None
    simple_file = evaluate_file + '.map.json'
    if worth_create(evaluate_file, simple_file):
        if result is None:
            logging.info('data reading...')
            eval_result= read_to_buffer(evaluate_file)
            logging.info('json parsing...')
            result = json.loads(eval_result)
        s = {}
        for size_type in result:
            if size_type not in s:
                s[size_type] = {}
            for thresh in result[size_type]:
                if thresh not in s[size_type]:
                    s[size_type][thresh] = {}
                s[size_type][thresh]['map'] = \
                        result[size_type][thresh]['map']
        write_to_file(json.dumps(s, indent=4, sort_keys=True), simple_file)

    simple_file = evaluate_file + '.class_ap.json'
    if worth_create(evaluate_file, simple_file):
        if result is None:
            eval_result= read_to_buffer(evaluate_file)
            result = json.loads(eval_result)
        s = {}
        for size_type in result:
            if size_type not in s:
                s[size_type] = {}
            for thresh in result[size_type]:
                if thresh not in s[size_type]:
                    s[size_type][thresh] = {}
                s[size_type][thresh]['class_ap'] = \
                        result[size_type][thresh]['class_ap']
        write_to_file(json.dumps(s, indent=4, sort_keys=True), simple_file)

    simple_file = '{}.prec.threshold.tsv'.format(evaluate_file)
    if worth_create(evaluate_file, simple_file):
        if result is None:
            logging.info('data reading...')
            eval_result= read_to_buffer(evaluate_file)
            logging.info('json parsing...')
            result = json.loads(eval_result)
        _, max_key = max([(float(k), k) for k in result['overall']],
                key=lambda x: x[0])
        class_thresh = result['overall'][max_key]['class_thresh']
        precision_ths = None
        for l in class_thresh:
            precision_ths = class_thresh[l].keys()
            break
        if precision_ths:
            for precision_th in precision_ths:
                sub_simple_file = '{}.{}.prec{}.threshold.tsv'.format(
                        evaluate_file, max_key, precision_th)
                def gen_rows():
                    for l in class_thresh:
                        th_recall = class_thresh[l].get(precision_th, [1, 0])
                        yield l, th_recall[0], th_recall[1]
                tsv_writer(gen_rows(), sub_simple_file)
        from_file = '{}.{}.prec{}.threshold.tsv'.format(evaluate_file, max_key, 0.5)
        if op.isfile(from_file) and worth_create(from_file, simple_file):
            copyfile(from_file, simple_file)

def init_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def get_acc_for_plot(eval_file):
    if 'coco_box' in eval_file:
        return load_from_yaml_file(eval_file)
    elif 'top1' in eval_file:
        return load_from_yaml_file(eval_file)
    else:
        raise NotImplementedError()

def calc_neg_aware_gmap(data, split, predict_file,
        apply_nms_det=False,
        expand_label_det=False,
        apply_nms_gt=False):
    #from qd.evaluate.evaluate_openimages_google import evaluate
    from qd.evaluate.evaluate_openimages_google import parallel_evaluate as evaluate
    dataset = TSVDataset(data)
    truths = dataset.get_data(split, 'label')
    imagelabel_truths = dataset.get_data(split, 'imagelabel')
    assert op.isfile(truths), truths
    assert op.isfile(imagelabel_truths)
    start = time.time()
    result = evaluate(truths, imagelabel_truths, predict_file,
            json_hierarchy_file=op.join(dataset._data_root, 'hierarchy.json'),
            apply_nms_det=apply_nms_det,
            expand_label_det=expand_label_det,
            expand_label_gt=True,
            apply_nms_gt=apply_nms_gt,
            )
    logging.info(time.time() - start)
    from qd.qd_common import convert_to_yaml_friendly
    result = convert_to_yaml_friendly(result)
    return result

def get_precisie_bn_model_file(model_file):
    return op.splitext(model_file)[0] + '.PreciseBN.pt'

def freeze_last_bn_stats(model, num_bn):
    '''
    weight and bias can be updated without problems here
    '''
    bn_layers = [m for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    targets = bn_layers[-num_bn:]
    for t in targets:
        t.eval()

def freeze_all_parameters_except(model, freeze_all_except):
    found = False
    result = []
    names = []
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if freeze_all_except in n:
            assert not found
            logging.info('not freeze: {}'.format(n))
            found = True
        else:
            result.append(m)
            names.append(n)
    assert found
    freeze_parameters(result)

def freeze_parameters_by_last_name(model, last_fixed_param):
    fixed_modules, names = get_all_module_need_fixed(model, last_fixed_param)
    logging.info('fix the parameters of the following modules: {}'.format(
        pformat(names)))
    freeze_parameters(fixed_modules)

def get_all_module_need_fixed(model, last_fixed_param):
    found = False
    result = []
    names = []
    for n, m in model.named_modules():
        if len(list(m.children())) > 0:
            continue
        if not found:
            result.append(m)
            names.append(n)
            if last_fixed_param in n:
                found = True
                break
        else:
            assert last_fixed_param not in n
    assert found
    return result, names

def all_gather(x):
    if get_mpi_size() == 1:
        return x
    else:
        with torch.no_grad():
            all_x = [torch.zeros_like(x) for _ in range(get_mpi_size())]
            # note, all_rep should be treated as constent, which means no grad
            # will be propagated back through all_rep
            torch.distributed.all_gather(all_x, x)
        return torch.cat(all_x, dim=0)

class L2NormModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.iter = 0

    def forward(self, x):
        verbose = (self.iter % 100) == 0
        self.iter += 1
        if verbose:
            from qd.torch_common import describe_tensor
            logging.info(describe_tensor(x))
        return torch.nn.functional.normalize(x)

def replace_fc_with_mlp_(model, num=1, mlp_bn=False,
                         with_l2=True):
    while not hasattr(model, 'fc'):
        model = model.module
    module_list = []
    dim_mlp = model.fc.weight.shape[1]
    for i in range(num):
        module_list.append(nn.Linear(dim_mlp, dim_mlp))
        if mlp_bn:
            module_list.append(nn.BatchNorm1d(dim_mlp))
        module_list.append(nn.ReLU())
    module_list.append(model.fc)
    if with_l2:
        module_list.append(L2NormModule())
    model.fc = nn.Sequential(*module_list)

class TorchTrain(object):
    def __init__(self, **kwargs):
        if 'load_parameter' in kwargs and kwargs['load_parameter']:
            assert 'full_expid' in kwargs
            kwargs_f = load_latest_parameters(op.join('output',
                kwargs['full_expid']))
            for k in kwargs_f:
                if k not in kwargs:
                    # we can overwrite the parameter in the parameter file
                    kwargs[k] = kwargs_f[k]

        self.kwargs = kwargs

        self._default = {
            'dist_backend': 'nccl',
            'init_method_type': 'tcp',
            'log_step': 100,
            'evaluate_method': 'map',
            'test_split': 'test',
            'num_workers': 4,
            'test_normalize_module': 'softmax',
            'restore_snapshot_iter': -1,
            'ovthresh': [-1],
            'step_lr': 30,
            'base_lr': 0.1,
            'dataset_type': 'single',
            'max_iter': 10,
            # the default value was 5e-4, which is the default for yolo. We
            # add the default as 5e-4 in yolo_by_mask, and set it 1e-4 for
            # classification.
            'weight_decay': 1e-4,
            'effective_batch_size': 256,
            'pretrained': False,
            'dist_url_tcp_port': 23456,
            'random_seed': 6,
            'apply_nms_gt': True,
            'cudnn_benchmark': False,
            'use_hvd': False,
            'device': 'cuda',
            'test_mergebn': False,
            'bgr2rgb': False, # this should be True, but set it False for back compatibility
            'coco_eval_max_det': 100,
            'train_crop_size': 224,
            'test_crop_size': 224,
            'momentum': 0.9,
            'scheduler_type': 'step',
            'train_transform': 'inception',
            'cosine_warmup_iters': 500,
            'cosine_warmup_factor': 1. / 3,
            'rms_alpha': 0.99,
            'smooth_label_eps': 0.1,
            'pred_tsv_to_json_extra': 1,
            'mobilenetv3_dropout_ratio': 0.2,
            'cutout_factor': 4,
            'min_rel_lr_in_cosine': 0.,
            'dist_weight': 1.,
            'find_unused_parameters': False,
        }

        assert 'batch_size' not in kwargs, 'use effective_batch_size'

        self.data = kwargs.get('data', 'Unknown')
        self.net = kwargs.get('net', 'Unknown')
        self.expid = kwargs.get('expid', 'Unknown')

        self.full_expid = kwargs.get('full_expid',
                '_'.join(map(str, [self.data, self.net, self.expid])))
        self.output_folder = op.join('output', self.full_expid)
        self.model_folder = op.join(self.output_folder, 'snapshot')
        ensure_directory(self.model_folder)
        self.test_data = kwargs.get('test_data', self.data)
        self.test_batch_size = kwargs.get('test_batch_size',
                self.effective_batch_size)
        if self.max_epoch is None and \
                type(self.max_iter) is str and \
                self.max_iter.endswith('e'):
            # we will not use max_epoch gradually
            self.max_epoch = int(self.max_iter[: -1])

        self.mpi_rank = get_mpi_rank()
        self.mpi_size= get_mpi_size()
        self.mpi_local_rank = get_mpi_local_rank()
        self.mpi_local_size = get_mpi_local_size()
        # we can set device_id = 0 always for debugging distributed & you only
        # have 1 gpu
        self.device_id = (self.mpi_local_rank if not
                self.kwargs.get('debug_train') else 0)

        # the following two environements are used in init_dist_process if the
        # method is by env. we make sure the world is the same here
        if 'WORLD_SIZE' in os.environ:
            assert int(os.environ['WORLD_SIZE']) == self.mpi_size
        if 'RANK' in os.environ:
            assert int(os.environ['RANK']) == self.mpi_rank

        # we will always use distributed version even when world size is 1
        self.distributed = True
        # adapt the batch size based on the mpi_size
        self.is_master = self.mpi_rank == 0

        assert (self.test_batch_size % self.mpi_size) == 0, self.test_batch_size
        self.test_batch_size = self.test_batch_size // self.mpi_size
        self.train_dataset = TSVDataset(self.data)

        self.initialized = False

    def get_num_training_images(self):
        return self.train_dataset.num_rows('train')

    def get_num_classes(self):
        return len(self.train_dataset.load_labelmap())

    def demo(self, path):
        logging.info('not implemented')

    @property
    def batch_size_per_gpu(self):
        return self.effective_batch_size // self.mpi_size

    @property
    def batch_size(self):
        # do not run assert in __init__ because we may just want to run
        # inference and ignore the training
        assert (self.effective_batch_size % self.mpi_size) == 0, (self.effective_batch_size, self.mpi_size)
        return self.effective_batch_size // self.mpi_size

    def get_labelmap(self):
        if not self.labelmap:
            self.labelmap = self.train_dataset.load_labelmap()
        return self.labelmap

    def __getattr__(self, key):
        if key in self.kwargs:
            return self.kwargs[key]
        elif key in self._default:
            return self._default[key]

    def get_dist_url(self):
        init_method_type = self.init_method_type
        if init_method_type == 'file':
            dist_file = op.abspath(op.join('output', 'dist_sync'))
            if not op.isfile(dist_file):
                ensure_directory(op.dirname(dist_file))
                open(dist_file, 'a').close()
            dist_url = 'file://' + dist_file
        elif init_method_type == 'tcp':
            dist_url = 'tcp://{}:{}'.format(get_master_node_ip(),
                    self.dist_url_tcp_port)
        elif init_method_type == 'env':
            dist_url = 'env://'
        else:
            raise ValueError('unknown init_method_type = {}'.format(init_method_type))
        return dist_url

    def model_surgery(self, model):
        from qd.layers.group_batch_norm import GroupBatchNorm, get_normalize_groups
        if self.convert_bn == 'L1':
            raise NotImplementedError
        elif self.convert_bn == 'L2':
            raise NotImplementedError
        elif self.convert_bn == 'GN':
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: torch.nn.GroupNorm(32, m.num_features),
                    )
        elif self.convert_bn == 'LNG': # layer norm by group norm
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: torch.nn.GroupNorm(1, m.num_features))
        elif self.convert_bn == 'ING': # Instance Norm by group norm
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: torch.nn.GroupNorm(m.num_features, m.num_features))
        elif self.convert_bn == 'GBN':
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: GroupBatchNorm(get_normalize_groups(m.num_features, self.normalization_group,
                        self.normalization_group_size), m.num_features))
        elif self.convert_bn == 'SBN':
            if self.distributed or True:
                model = replace_module(model,
                        lambda m: isinstance(m, torch.nn.BatchNorm2d) or
                            isinstance(m, torch.nn.BatchNorm1d),
                        lambda m: torch.nn.SyncBatchNorm(m.num_features,
                            eps=m.eps,
                            momentum=m.momentum,
                            affine=m.affine,
                            track_running_stats=m.track_running_stats))
                from qd.layers.batch_norm import FrozenBatchNorm2d
                model = replace_module(model,
                        lambda m: isinstance(m, FrozenBatchNorm2d),
                        lambda m: torch.nn.SyncBatchNorm(m.num_features,
                            eps=m.eps))

        elif self.convert_bn == 'FBN': # frozen batch norm
            def set_eval_return(m):
                m.eval()
                return m
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d) or
                                   isinstance(m, torch.nn.BatchNorm1d),
                    lambda m: set_eval_return(m))
        #elif self.convert_bn == 'NSBN':
            #if self.distributed:
                #from qd.layers.batch_norm import NaiveSyncBatchNorm
                #model = replace_module(model,
                        #lambda m: isinstance(m, torch.nn.BatchNorm2d),
                        #lambda m: NaiveSyncBatchNorm(m.num_features,
                            #eps=m.eps,
                            #momentum=m.momentum,
                            #affine=m.affine,
                            #track_running_stats=m.track_running_stats))
        elif self.convert_bn == 'CBN':
            from qd.layers.batch_norm import ConvergingBatchNorm
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: ConvergingBatchNorm(
                        policy=self.cbn_policy,
                        max_iter=self.max_iter,
                        gamma=self.cbn_gamma,
                        num_features=m.num_features,
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=True,
                        track_running_stats=m.track_running_stats,
                        ))
        else:
            assert self.convert_bn is None, self.convert_bn
        if self.fc_as_mlp:
            # this is used, normally for self-supervised learning scenarios
            replace_fc_with_mlp_(model)
        if self.hswish2relu6:
            from qd.layers.mitorch_models.modules.activation import HardSwish
            model = replace_module(model,
                    lambda m: isinstance(m,
                                         HardSwish),
                    lambda m: torch.nn.ReLU6(inplace=True))
        if self.vis_adaptive_global_pool:
            from qd.layers.adapt_avg_pool2d import VisAdaptiveAvgPool2d
            model = replace_module(model,
                    lambda m: isinstance(m,
                                         nn.AdaptiveAvgPool2d),
                    lambda m: VisAdaptiveAvgPool2d())
        if self.freeze_bn:
            from qd.torch_common import freeze_bn_
            freeze_bn_(model)
        if self.standarize_conv2d:
            from qd.layers.standarized_conv import convert_conv2d_to_standarized_conv2d
            model = convert_conv2d_to_standarized_conv2d(model)
        if self.bn_momentum:
            from qd.torch_common import update_bn_momentum
            update_bn_momentum(model, self.bn_momentum)
        if self.c_bias_sigmoid_small is not None:
            from qd.torch_common import query_modules_by_name
            modules = query_modules_by_name(model, '.fc')
            assert len(modules) == 1
            fc = modules[0]
            nn.init.constant_(fc.bias, -math.log(1. / self.c_bias_sigmoid_small - 1))

        # assign a name to each module so that we can use it in each module to
        # print debug information
        from qd.torch_common import attach_module_name_
        attach_module_name_(model)
        return model

    def _ensure_initialized(self):
        if self._initialized:
            return

        if self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        # in torch 0.4, torch.randperm only supports cpu. if we set it as
        # cuda.Float by default, it will crash there
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #torch.set_default_tensor_type('torch.FloatTensor')
        if self.use_hvd:
            # work with world size == 1, also
            import horovod.torch as hvd
            hvd.init()
            torch.cuda.set_device(hvd.local_rank())
            assert hvd.rank() == self.mpi_rank
            assert hvd.size() == self.mpi_size
            assert hvd.local_size() == self.mpi_local_size
        if self.distributed:
            if not self.use_hvd:
                dist_url = self.get_dist_url()
                init_param = {'backend': self.dist_backend,
                        'init_method': dist_url,
                        'rank': self.mpi_rank,
                        'world_size': self.mpi_size}
                # always set the device at the very beginning
                torch.cuda.set_device(self.device_id)
                logging.info('init param: \n{}'.format(pformat(init_param)))
                if not dist.is_initialized():
                    dist.init_process_group(**init_param)
                # sometimes, the init hangs, and thus we print some logs for
                # verification
                logging.info('initialized')
                # we need to synchronise before exit here so that all workers can
                # finish init_process_group(). If not, worker A might exit the
                # whole program first, but worker B still needs to talk with A. In
                # that case, worker B will never return and will hang there
                synchronize()
        init_random_seed(self.random_seed)
        self._initialized = True

    def get_train_transform(self):
        if self.train_transform == 'inception':
            data_augmentation = get_train_transform(self.bgr2rgb,
                        crop_size=self.train_crop_size)
            if self.dict_trainer:
                data_augmentation = ImageTransform2Dict(data_augmentation)
        elif self.train_transform == 'cutout':
            normalize = get_data_normalize()
            totensor = transforms.ToTensor()
            color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
            all_trans = []
            if self.bgr2rgb:
                all_trans.append(BGR2RGB())
            all_trans.extend([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.train_crop_size),
                color_jitter,
                transforms.RandomHorizontalFlip(),
                totensor,
                normalize,
                transforms.RandomApply([ImageCutout(1./self.cutout_factor)], p=0.5),
            ])
            data_augmentation = transforms.Compose(all_trans)
            if self.dict_trainer:
                data_augmentation = ImageTransform2Dict(data_augmentation)

        elif self.train_transform == 'no_color':
            normalize = get_data_normalize()
            totensor = transforms.ToTensor()
            all_trans = []
            if self.bgr2rgb:
                all_trans.append(BGR2RGB())
            all_trans.extend([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.train_crop_size),
                transforms.RandomHorizontalFlip(),
                totensor,
                normalize,])
            data_augmentation = transforms.Compose(all_trans)
            if self.dict_trainer:
                data_augmentation = ImageTransform2Dict(data_augmentation)
        elif self.train_transform == 'simclr':
            normalize = get_data_normalize()
            totensor = transforms.ToTensor()
            all_trans = []
            if self.bgr2rgb:
                all_trans.append(BGR2RGB())
            gaussian_kernel_size = int(0.1 * self.train_crop_size) // 2 * 2 + 1
            color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            from qd.data_layer.transform import SimCLRGaussianBlur
            all_trans.extend([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.train_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                SimCLRGaussianBlur(kernel_size=gaussian_kernel_size),
                totensor,
                normalize,])
            data_augmentation = transforms.Compose(all_trans)
            if self.dict_trainer:
                data_augmentation = ImageTransform2Dict(data_augmentation)
        elif self.train_transform == 'aa':
            normalize = get_data_normalize()
            totensor = transforms.ToTensor()
            all_trans = []
            if self.bgr2rgb:
                all_trans.append(BGR2RGB())
            from qd.data_layer.autoaugmentation import ImageNetPolicy
            all_trans.extend([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.train_crop_size),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                totensor,
                normalize,])
            data_augmentation = transforms.Compose(all_trans)
            if self.dict_trainer:
                data_augmentation = ImageTransform2Dict(data_augmentation)
        elif self.train_transform == 'rand_aug':
            normalize = get_data_normalize()
            totensor = transforms.ToTensor()
            all_trans = []
            if self.bgr2rgb:
                all_trans.append(BGR2RGB())
            from qd.data_layer.rand_augmentation import rand_augment_transform

            # this is default
            config_str = 'rand-m9-mstd0.5'
            fillcolor = [0.5, 0.5, 0.5]
            hparams = dict(
                translate_const=int(self.train_crop_size * 0.45),
                img_mean=tuple([min(255, round(255 * x)) for x in fillcolor]),
            )

            all_trans.extend([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.train_crop_size),
                transforms.RandomHorizontalFlip(),
                rand_augment_transform(config_str, hparams),
                totensor,
                normalize,])
            data_augmentation = transforms.Compose(all_trans)
        elif self.train_transform == 'feature':
            # this should be paired with io dataset
            from qd.data_layer.transform import (
                CvImageDecodeDict,
                DecodeFeatureDict,
                RemoveUselessKeys,
            )
            decode_transform = [
                CvImageDecodeDict(),
                DecodeFeatureDict(),
                RemoveUselessKeys(),
            ]
            decode_transform = transforms.Compose(decode_transform)
            data_augmentation = get_train_transform(self.bgr2rgb, crop_size=self.train_crop_size)
            data_augmentation = ImageTransform2Dict(data_augmentation)
            data_augmentation = transforms.Compose([
                decode_transform,
                data_augmentation])
        elif self.train_transform == 'rand_cut':
            normalize = get_data_normalize()
            totensor = transforms.ToTensor()
            if self.min_size_range32 is None:
                all_trans = []
                if self.bgr2rgb:
                    all_trans.append(BGR2RGB())
                from qd.data_layer.rand_augmentation import rand_augment_transform

                # this is default
                config_str = 'rand-m9-mstd0.5'
                fillcolor = [0.5, 0.5, 0.5]
                hparams = dict(
                    translate_const=int(self.train_crop_size * 0.45),
                    img_mean=tuple([min(255, round(255 * x)) for x in fillcolor]),
                )

                all_trans.extend([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(self.train_crop_size),
                    transforms.RandomHorizontalFlip(),
                    rand_augment_transform(config_str, hparams),
                    totensor,
                    normalize,
                    transforms.RandomApply([ImageCutout(1./self.cutout_factor)], p=0.5),
                ])
                data_augmentation = transforms.Compose(all_trans)
                if self.dict_trainer:
                    data_augmentation = ImageTransform2Dict(data_augmentation)
            else:
                first_trans = []
                if self.bgr2rgb:
                    first_trans.append(BGR2RGB())
                from qd.data_layer.rand_augmentation import rand_augment_transform

                # this is default
                config_str = 'rand-m9-mstd0.5'
                fillcolor = [0.5, 0.5, 0.5]
                hparams = dict(
                    translate_const=int(self.train_crop_size * 0.45),
                    img_mean=tuple([min(255, round(255 * x)) for x in fillcolor]),
                )

                first_trans.extend([
                    transforms.ToPILImage(),
                ])
                first_trans = transforms.Compose(first_trans)
                first_trans = ImageTransform2Dict(first_trans)
                from qd.data_layer.transform import RandomResizedCropMultiSize
                all_size = list(range(self.min_size_range32[0], self.min_size_range32[1] + 31, 32))
                if self.train_crop_size not in all_size:
                    all_size.append(self.train_crop_size)
                second_trans = RandomResizedCropMultiSize(all_size)
                third_trans = [
                    transforms.RandomHorizontalFlip(),
                    rand_augment_transform(config_str, hparams),
                    totensor,
                    normalize,
                    transforms.RandomApply([ImageCutout(1./self.cutout_factor)], p=0.5),
                ]
                third_trans = transforms.Compose(third_trans)
                third_trans = ImageTransform2Dict(third_trans)
                data_augmentation = transforms.Compose([
                    first_trans, second_trans, third_trans])
        else:
            raise NotImplementedError(self.train_transform)

        logging.info(data_augmentation)
        return data_augmentation

    def get_transform(self, stage):
        if stage == 'train':
            transform = self.get_train_transform()
        else:
            transform = self.get_test_transform()
        logging.info(transform)
        return transform

    def get_test_transform(self):
        resize_size = self.test_resize_size
        if resize_size is None:
            resize_size = 256 * self.test_crop_size // 224
        transform = get_test_transform(
            self.bgr2rgb,
            resize_size=resize_size,
            crop_size=self.test_crop_size,
            crop_position=self.test_crop_position)
        if self.dict_trainer:
            transform = ImageTransform2Dict(transform)
        return transform

    def _get_dataset(self, data, split, stage, labelmap, dataset_type):
        if not self.bgr2rgb:
            logging.warn('normally bgr2rgb should be true.')

        transform = self.get_transform(stage)

        return self.create_dataset_with_transform(data, split, stage, labelmap, dataset_type,
                                   transform)

    def create_dataset_with_transform(self, data, split, stage, labelmap, dataset_type,
                       transform):
        if dataset_type == 'single':
            return TSVSplitImage(data, split,
                    version=0,
                    transform=transform,
                    labelmap=labelmap,
                    cache_policy=self.cache_policy)
        elif dataset_type == 'single_dict':
            return TSVSplitImageDict(data, split,
                    version=0,
                    transform=transform,
                    labelmap=labelmap,
                    cache_policy=self.cache_policy)
        elif dataset_type == 'io':
            # this is the recommended setting. all others can be implemented in
            # transform
            from qd.data_layer.dataset import IODataset, DatasetPlusTransform
            io_set = IODataset(data, split, version=0)
            return DatasetPlusTransform(io_set, transform)
        elif dataset_type == 'soft_assign':
            return TSVSplitImageSoftAssign(data, split,
                    transform=transform,
                    temperature=self.temperature,
                    num_kmeans=self.heads,
                    )
        if dataset_type == 'crop':
            return TSVSplitCropImage(data, split,
                    version=0,
                    transform=transform,
                    labelmap=labelmap,
                    cache_policy=self.cache_policy)
        elif dataset_type == 'multi_hot':
            return TSVSplitImageMultiLabel(data, split,
                    version=0,
                    transform=transform,
                    cache_policy=self.cache_policy,
                    labelmap=labelmap)
        elif dataset_type == 'multi_hot_neg':
            return TSVSplitImageMultiLabelNeg(data, split, version=0,
                    transform=transform,
                    cache_policy=self.cache_policy,
                    labelmap=labelmap)
        else:
            raise ValueError('unknown {}'.format(self.dataset_type))

    def _get_criterion(self):
        if self.dataset_type in ['single', 'crop', 'single_dict',
                                 'io']:
            if self.loss_type == 'NTXent':
                from qd.layers.ntxent_loss import NTXentLoss
                criterion = NTXentLoss(self.temperature, self.correct_loss)
            elif self.loss_type == 'NTXentQueue':
                from qd.layers.ntxent_loss import NTXentQueueLoss
                criterion = NTXentQueueLoss(self.temperature,
                                            self.queue_size,
                                            self.out_dim,
                                            self.queue_alpha,
                                            alpha_max=self.queue_alpha_max,
                                            alpha_policy=self.queue_alpha_policy,
                                            max_iter=self.max_iter,
                                            criterion_type=self.criterion_type,
                                            denominator_ce_factor=self.denominator_ce_factor
                                            )
            elif self.loss_type == 'SwAV':
                from qd.layers.ntxent_loss import SwAVQueueLoss
                criterion = SwAVQueueLoss(
                    self.temperature,
                    cluster_size=self.cluster_size,
                    queue_size=self.queue_size,
                    involve_queue_after=self.involve_queue_after,
                    dim=self.out_dim)
            elif self.loss_type == 'SimpleQueue':
                from qd.layers.ntxent_loss import SimpleQueueLoss
                criterion = SimpleQueueLoss(self.temperature,
                                            self.queue_size,
                                            self.out_dim,
                                            self.queue_alpha,
                                            alpha_max=self.queue_alpha_max,
                                            alpha_policy=self.queue_alpha_policy,
                                            max_iter=self.max_iter,
                                            criterion_type=self.criterion_type,
                                            denominator_ce_factor=self.denominator_ce_factor
                                            )
            elif self.loss_type == 'NoisyDis':
                from qd.layers.ntxent_loss import NoisyDiscriminator
                criterion = NoisyDiscriminator(self.out_dim)
            elif self.loss_type == 'multi_ce':
                from qd.layers.loss import MultiCrossEntropyLoss
                criterion = MultiCrossEntropyLoss(weights=self.multi_ce_weights)
            elif self.loss_type == 'dist_ce':
                from qd.layers.loss import DistilCrossEntropyLoss
                criterion = DistilCrossEntropyLoss(self.dist_ce_weight)
            elif self.loss_type == 'smooth_ce':
                from qd.layers.loss import SmoothLabelCrossEntropyLoss
                criterion = SmoothLabelCrossEntropyLoss(eps=self.smooth_label_eps)
            elif self.loss_type == 'ExCE':
                from qd.layers.loss import ExclusiveCrossEntropyLoss
                criterion = ExclusiveCrossEntropyLoss(2.)
            elif self.loss_type == 'kl_ce':
                from qd.layers.loss import KLCrossEntropyLoss
                criterion = KLCrossEntropyLoss()
            elif self.loss_type == 'l2':
                from qd.layers.loss import L2Loss
                criterion = L2Loss()
            elif self.loss_type == 'mo_dist_ce':
                from qd.layers.loss import DistillCrossEntropyLoss
                criterion = DistillCrossEntropyLoss(
                    num_image=self.get_num_training_images(),
                    num_class=self.get_num_classes(),
                    momentum=self.dist_ce_momentum,
                    dist_weight=self.dist_weight,
                )
            elif self.loss_type == 'eff_fpn_ce':
                from qd.layers.loss import EfficientDetCrossEntropy
                criterion = EfficientDetCrossEntropy(
                    no_reg=self.no_reg,
                    sep=self.sep,
                )
            else:
                criterion = nn.CrossEntropyLoss().cuda()
        elif self.dataset_type in ['soft_assign']:
            from qd.layers.kl_div_logit_loss import KLDivLogitLoss
            criterion = KLDivLogitLoss()
        elif self.dataset_type == 'multi_hot':
            if self.loss_type == 'BCEWithLogitsLoss':
                criterion = nn.BCEWithLogitsLoss().cuda()
            elif self.loss_type == 'MultiHotCrossEntropyLoss':
                criterion = MultiHotCrossEntropyLoss()
            else:
                raise Exception('not support value {}'.format(self.loss_type))
        elif self.dataset_type == 'multi_hot_neg':
            assert self.loss_type == 'BCEWithLogitsNegLoss'
            criterion = BCEWithLogitsNegLoss()
        else:
            raise NotImplementedError
        return criterion

    def _get_checkpoint_file(self, epoch=None, iteration=None):
        if iteration is None and epoch is None and self.model_file is not None:
            return self.model_file
        assert epoch is None, 'not supported'
        if iteration is None:
            iteration = self.max_iter
        iteration = self.parse_iter(iteration)
        return op.join(self.output_folder, 'snapshot',
                "model_iter_{:07d}.pt".format(iteration))

    def _get_model(self, pretrained, num_class):
        if self.net in ['MobileNetV3', 'MobileNetV3Small']:
            from qd.layers.mitorch_models import ModelFactory
            model = ModelFactory.create(self.net,
                                        num_class,
                                        dropout_ratio=self.mobilenetv3_dropout_ratio)
            assert not pretrained
        elif self.net.startswith('efficientdet'):
            # used for pre-training
            assert not pretrained
            compound = int(self.net[-1])
            from qd.layers.efficient_det import EfficientDetBackbone
            model = EfficientDetBackbone(num_classes=num_class,
                                            compound_coef=compound,
                                            ratios=[(1., 1.)],
                                            scales=[2 ** 0],
                                            prior_prob=0.5,
                                            adaptive_up=True,
                                            anchor_scale=3,
                                            drop_connect_rate=None,
                                            box_dim=num_class,
                                            )
        elif self.net.startswith('efficientnet'):
            if pretrained:
                assert ValueError('not tested')
                from efficientnet_pytorch import EfficientNet
                model = EfficientNet.from_pretrained(
                    self.net,
                    num_class)
            else:
                from qd.layers import efficient_det
                if self.efficient_net_simple_padding:
                    efficient_det.g_simple_padding = True
                else:
                    efficient_det.g_simple_padding = False
                from qd.layers.efficient_det import EfficientNet
                model = EfficientNet.from_name(
                    self.net, # efficientnet-b0
                    override_params={'num_classes': num_class})
                assert not pretrained
        elif self.net.startswith('yolov5'):
            from qd.layers.yolov5 import ClassificationModel, get_model_cfg
            model = ClassificationModel(cfg=get_model_cfg(self.net))
        elif self.net.startswith('resnet'):
            dict_data = {
                'from': 'qd.layers.resnet',
                'import': self.net,
                'param': {
                    'pretrained': self.pretrained,
                    'stem_stride2': self.stem_stride2,
                },
            }
            from qd.qd_common import execute_func
            model = execute_func(dict_data)
            if model.fc.weight.shape[0] != num_class:
                model.fc = nn.Linear(model.fc.weight.shape[1], num_class)
                torch.nn.init.xavier_uniform_(model.fc.weight)
            if self.moco_finetune_init_last_linear:
                logging.info('moco fine-tune init last linear')
                model.fc.weight.data.normal_(mean=0.0, std=0.01)
                model.fc.bias.data.zero_()
            if self.zero_init_last_linear:
                logging.info('zero out last linear')
                model.fc.weight.data.zero_()
                model.fc.bias.data.zero_()
        else:
            model = models.__dict__[self.net](pretrained=pretrained)
            if model.fc.weight.shape[0] != num_class:
                model.fc = nn.Linear(model.fc.weight.shape[1], num_class)
                torch.nn.init.xavier_uniform_(model.fc.weight)
            if self.moco_finetune_init_last_linear:
                logging.info('moco fine-tune init last linear')
                model.fc.weight.data.normal_(mean=0.0, std=0.01)
                model.fc.bias.data.zero_()
            if self.zero_init_last_linear:
                logging.info('zero out last linear')
                model.fc.weight.data.zero_()
                model.fc.bias.data.zero_()

        if self.dict_trainer:
            model = InputAsDict(model)
        return model

    def _data_parallel_wrap(self, model):
        if self.distributed:
            model.cuda()
            #if self.mpi_local_size > 1:
            if not self.use_hvd:
                model = torch.nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.device_id],
                    # used for effiicient-net + faster-rcnn
                    find_unused_parameters=self.find_unused_parameters,
                )
            #else:
                #if not self.use_hvd:
                    #model = torch.nn.parallel.DistributedDataParallel(model)
        #else:
            #model = torch.nn.parallel.DistributedDataParallel(model)
            #model = torch.nn.DataParallel(model).cuda()
        return model

    def _get_train_data_loader(self, start_iter=0):
        train_dataset = self._get_dataset(self.data,
                'train',
                stage='train',
                labelmap=self.train_dataset.load_labelmap(),
                dataset_type=self.dataset_type)

        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            self.train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=(self.train_sampler==None),
            num_workers=self.num_workers, pin_memory=True,
            sampler=self.train_sampler)

        return train_dataset, train_loader

    def _get_test_data_loader(self, test_data, test_split, labelmap):
        if self.test_dataset_type is None:
            self.test_dataset_type = self.dataset_type

        test_dataset = self._get_dataset(test_data, test_split, stage='test',
                labelmap=labelmap, dataset_type=self.test_dataset_type)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.test_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True)

        return test_dataset, test_loader

    def _save_parameters(self):
        save_parameters(self.kwargs, self.output_folder)

    def _setup_logging(self):
        # all ranker outputs the log to a file
        # only rank 0 print the log to console
        log_file = op.join(self.output_folder,
            'log_{}_rank{}.txt'.format(
                datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                self.mpi_rank))
        ensure_directory(op.dirname(log_file))
        file_handle = logging.FileHandler(log_file)
        logger_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s')
        file_handle.setFormatter(fmt=logger_fmt)

        root = logging.getLogger()
        root.handlers = []
        root.setLevel(logging.INFO)
        root.addHandler(file_handle)

        if self.mpi_rank == 0:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logger_fmt)
            root.addHandler(ch)

    def get_latest_checkpoint(self):
        all_snapshot = glob.glob(op.join(self.output_folder, 'snapshot',
            'model_iter_*e.pth.tar'))
        if len(all_snapshot) == 0:
            return
        snapshot_epochs = [(s, parse_epoch(s)) for s in all_snapshot]
        s, _ = max(snapshot_epochs, key=lambda x: x[1])
        return s

    def ensure_train(self):
        self._ensure_initialized()

        last_model_file = self._get_checkpoint_file()
        logging.info('last model file = {}'.format(last_model_file))
        if op.isfile(last_model_file) and not self.force_train:
            logging.info('skip to train')
            return

        ensure_directory(op.join(self.output_folder, 'snapshot'))

        if self.mpi_rank == 0:
            self._save_parameters()

        ensure_directory(self.output_folder)

        logging.info(pformat(self.kwargs))
        from qd.torch_common import get_torch_version_info
        logging.info('torch info = {}'.format(
            pformat(get_torch_version_info())))

        if self.mpi_rank == 0 and not self.debug_train:
            from qd.qd_common import zip_qd, try_delete
            # we'd better to delete it since it seems like zip will read/write
            # if there is
            try_delete(op.join(self.output_folder, 'source_code.zip'))
            zip_qd(op.join(self.output_folder, 'source_code'))
        synchronize()

        self._setup_logging()
        train_result = self.train()

        synchronize()

        return train_result

    def init_model(self, model):
        if self.init_from:
            if self.init_from['type'] == 'best_model':
                c = TorchTrain(full_expid=self.init_from['full_expid'],
                        load_parameter=True)
                model_file = c._get_checkpoint_file()
                logging.info('loading from {}'.format(model_file))
                checkpoint = torch_load(model_file)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError('unknown init type = {}'.format(self.init_from['type']))

    def get_optimizer(self, model):
        parameters = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.base_lr
            weight_decay = self.weight_decay
            if self.bias_no_weight_decay and "bias" in key:
                weight_decay = 0.
            if self.conv_no_weight_decay and key.endswith('conv.weight'):
                weight_decay = 0.
            logging.info('{}: lr = {}; weight_decay = {}'.format(
                key, lr, weight_decay
            ))
            parameters += [{"params": [value],
                            "lr": lr,
                            "weight_decay": weight_decay,
                            'param_names': [key]}]

        if self.optimizer_type in [None, 'SGD', 'LARS']:
            from qd.opt.sgd import SGDVerbose
            optimizer = SGDVerbose(parameters,
                                   self.base_lr,
                                   momentum=self.momentum,
                                   # this is default decay, and will be
                                   # overwritten if we specified it in
                                   # parameters.
                                   weight_decay=self.weight_decay,
                                   nesterov=self.sgd_nesterov,
                                   )
        elif self.optimizer_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(
                parameters,
                self.base_lr,
                momentum=self.momentum,
                alpha=self.rms_alpha,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type in ['Adam']:
            optimizer = torch.optim.Adam(parameters, self.base_lr,
                                        weight_decay=self.weight_decay)
        elif self.optimizer_type in ['AdamW']:
            optimizer = torch.optim.AdamW(parameters,
                                          self.base_lr,
                                          weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(self.optimizer_type)
        if self.optimizer_type in ['LARS']:
            from torchlars import LARS
            optimizer = LARS(optimizer=optimizer)
        if self.ema_optimizer:
            from qd.opt.ema_optimizer import EMAOptimizer
            optimizer = EMAOptimizer(optimizer=optimizer)
        return optimizer

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        if self.scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                    step_size=self.parse_iter(self.step_lr),
                    last_epoch=last_epoch)
        elif self.scheduler_type == 'multi_step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=[self.parse_iter(i) for i in self.stageiter],
                    gamma=0.1,
                    last_epoch=last_epoch)
        elif self.scheduler_type == 'cosine':
            from qd.opt.WarmupCosineAnnealingLR import WarmupCosineAnnealingLR
            assert isinstance(self.max_iter, int)
            scheduler = WarmupCosineAnnealingLR(optimizer,
                    max_iter=self.max_iter,
                    last_epoch=last_epoch,
                    min_lr=self.min_rel_lr_in_cosine * self.base_lr,
                    warmup_factor=self.cosine_warmup_factor,
                    warmup_iters=self.parse_iter(self.cosine_warmup_iters),
                    cosine_restart_after_warmup=self.cosine_restart_after_warmup
                    )
        elif self.scheduler_type == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                    T_max=self.max_iter,
                    last_epoch=last_epoch)
        elif self.scheduler_type == 'ReduceLROnPlateau':
            assert isinstance(self.max_iter, int)
            patience = 3 * self.max_iter // self.effective_batch_size
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=patience, verbose=True)
        else:
            raise NotImplementedError(self.scheduler_type)
        return scheduler

    def train(self):
        train_dataset, train_loader = self._get_train_data_loader()
        num_class = train_dataset.get_num_pos_labels()

        model = self._get_model(self.pretrained, num_class)
        model = self._data_parallel_wrap(model)

        self.init_model(model)

        optimizer = torch.optim.SGD(model.parameters(), self.base_lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)
        if self.use_hvd:
            import horovod.torch as hvd
            optimizer = hvd.DistributedOptimizer(optimizer,
                    model.named_parameters())

        start_epoch = 0
        if self.restore_snapshot_iter == -1:
            last_checkpoint = self.get_latest_checkpoint()
            if last_checkpoint:
                checkpoint = torch.load(last_checkpoint)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']

        cudnn.benchmark = True
        max_iter_each_epoch = self.max_iter_each_epoch
        logging.info('start to train')
        for epoch in range(start_epoch, self.max_epoch):
            if self.distributed:
                # this is optional. need experiment to verify if it helps
                self.train_sampler.set_epoch(epoch)

            self._adjust_learning_rate(optimizer, epoch)

            self.train_epoch(train_loader, model, optimizer, epoch,
                    max_iter_each_epoch)

            if self.is_master:
                torch_save({
                    'epoch': epoch + 1,
                    'net': self.net,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, self._get_checkpoint_file(iteration='{}e'.format(epoch + 1)))

    def train_epoch(self, train_loader, model, optimizer, epoch,
            max_iter=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        # switch to train mode
        model.train()
        iter_start = time.time()
        log_start = iter_start
        log_step = self.log_step
        criterion = self._get_criterion()
        for i, (input, target, _) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - iter_start)

            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - iter_start)

            if i % log_step == 0:
                speed = self.mpi_size * log_step * target.shape[0] / (time.time() - log_start)
                logging.info('Epoch: [{0}][{1}/{2}], ' \
                      'Speed: {speed:.2f} samples/sec, ' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f}), '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f}), '
                      'Loss {loss.val:.4f} ({loss.avg:.4f}).'.format(
                       epoch, i, len(train_loader), speed=speed,
                       batch_time=batch_time,
                       data_time=data_time,
                       loss=losses))
                log_start = time.time()
                if max_iter and i > max_iter:
                    logging.info('break')
                    break
            iter_start = time.time()

    def _adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.base_lr * (0.1 ** (epoch // self.step_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _get_predict_file(self, model_file=None):
        if model_file is None:
            model_file = self._get_checkpoint_file(iteration=self.max_iter)
        cc = [model_file, self.test_data, self.test_split]
        self.append_predict_param(cc)
        if self.test_max_iter is not None:
            # this is used for speed test
            if self.test_mergebn:
                cc.append('mergebn')
            cc.append('max_iter{}'.format(self.test_max_iter))
            # we explicitly log the batch size here so that we can make sure it
            # is 1 or batch processing
            cc.append('BS{}'.format(self.test_batch_size))
            cc.append(self.device)
            if self.device == 'cpu' and self.cpu_num_threads:
                torch.set_num_threads(self.cpu_num_threads)
                cc.append('thread{}'.format(self.cpu_num_threads))
        if self.flush_denormal and self.device == 'cpu':
            # gpu is not supported
            r = torch.set_flush_denormal(True)
            assert r, 'not supported'
            cc.append('flush_denormal')
        if self.pred_file_hint is not None:
            cc.append(self.pred_file_hint)
        if self.test_crop_size != 224 and self.test_crop_size:
            cc.append('crop{}'.format(self.test_crop_size))
        cc.append('predict')
        cc.append('tsv')
        return '.'.join(cc)

    def append_predict_param(self, cc):
        if self.test_normalize_module != 'softmax':
            cc.append('NormBy{}'.format(self.test_normalize_module))
        if self.predict_extract:
            s = self.predict_extract
            if isinstance(self.predict_extract, list):
                s = '.'.join(self.predict_extract)
            cc.append('Extract{}'.format(s))
        if self.test_crop_position:
            cc.append(self.test_crop_position)
        if self.test_resize_size:
            cc.append('r{}'.format(self.test_resize_size))
        if self.predict_ema_decay:
            cc.append('ema{}'.format(self.predict_ema_decay))

    def monitor_train(self):
        self._ensure_initialized()
        while True:
            self.standarize_intermediate_models()
            need_wait_models = self.pred_eval_intermediate_models()
            all_step = self.get_all_steps()
            all_eval_file = [self._get_evaluate_file(self._get_predict_file(self._get_checkpoint_file(iteration=i)))
                for i in all_step]
            iter_to_eval = dict((i, get_acc_for_plot(eval_file))
                    for i, eval_file in zip(all_step, all_eval_file) if
                        op.isfile(eval_file))
            self.update_acc_iter(iter_to_eval)
            if need_wait_models == 0:
                break
            time.sleep(5)

        if self.mpi_rank == 0:
            self.save_to_tensorboard()
        synchronize()

    def save_to_tensorboard(self):
        all_step = self.get_all_steps()
        all_eval_file = [self._get_evaluate_file(self._get_predict_file(self._get_checkpoint_file(iteration=s)))
            for s in all_step]
        all_step_eval_result = [(s, get_acc_for_plot(e)) for s, e in zip(all_step,
            all_eval_file) if op.isfile(e)]

        tensorboard_folder = op.join('output', self.full_expid, 'tensorboard_data')
        from torch.utils.tensorboard import SummaryWriter
        ensure_remove_dir(tensorboard_folder)
        wt = SummaryWriter(log_dir=tensorboard_folder)
        tag_prefix = '{}_{}'.format(self.test_data, self.test_split)
        for step, eval_result in all_step_eval_result:
            for k in eval_result:
                wt.add_scalar(tag='{}_{}'.format(tag_prefix, k),
                        scalar_value=eval_result[k],
                        global_step=step)
        wt.close()

    def is_train_finished(self):
        last_model = self._get_checkpoint_file()
        logging.info('checking if {} exists'.format(last_model))
        return op.isfile(last_model) or op.islink(last_model)

    def update_acc_iter(self, iter_to_eval):
        if self.mpi_rank == 0:
            xys = list(iter_to_eval.items())
            xys = sorted(xys, key=lambda x: x[0])
            xs = [x for x, _ in xys]
            if all('all-all' in y for _, y in xys):
                # coco accuracy
                ys = [y['all-all'] for _, y in xys]
            elif all('top1' in y for _, y in xys):
                ys = [y['top1'] for _, y in xys]

            if len(xs) > 0:
                out_file = os.path.join(
                    self.output_folder,
                    'map_{}_{}.png'.format(self.test_data,
                        self.test_split))
                logging.info('create {}'.format(out_file))
                if op.isfile(out_file):
                    os.remove(out_file)
                plot_to_file(xs, ys, out_file)
            else:
                logging.info('nothing plotted')
        synchronize()

    def get_all_steps(self):
        steps = self.get_snapshot_steps()
        curr = 0
        all_step = []
        while True:
            curr += steps
            if curr >= self.max_iter:
                all_step.append(self.max_iter)
                break
            all_step.append(curr)
        return all_step

    def get_intermediate_model_status(self):
        ready_predict = []
        all_step = self.get_all_steps()
        for step in all_step[:-1]:
            model_file = self._get_checkpoint_file(iteration=step)
            if not op.isfile(model_file):
                ready_predict.append(0)
                continue
            predict_result_file = self._get_predict_file(model_file)
            eval_file = self._get_evaluate_file(predict_result_file)
            if not worth_create(model_file, predict_result_file) and \
                    not worth_create(predict_result_file, eval_file):
                ready_predict.append(-1)
                continue
            ready_predict.append(1)
        if self.mpi_size > 1:
            # by default, we use nccl backend, which only supports gpu. Thus,
            # we should not use cpu here.
            ready_predict = torch.tensor(ready_predict).cuda()
            dist.broadcast(ready_predict, src=0)
            ready_predict = ready_predict.tolist()
        return ready_predict

    def pred_eval_intermediate_models(self):
        ready_predict = self.get_intermediate_model_status()
        all_step = self.get_all_steps()[:-1]
        all_ready_predict_step = [step for step, status in zip(all_step, ready_predict) if status == 1]
        for step in all_ready_predict_step:
            model_file = self._get_checkpoint_file(iteration=step)
            pred = self.ensure_predict(model_file=model_file)
            self.ensure_evaluate(pred)
        return len([x for x in ready_predict if x == 0])

    def standarize_intermediate_models(self):
        # hack the original file format
        if self.mpi_rank == 0:
            steps = self.get_snapshot_steps()
            curr = 0
            while True:
                curr += steps
                if curr >= self.max_iter:
                    break
                original_model = self._get_old_check_point_file(curr)
                new_model = self._get_checkpoint_file(iteration=curr)
                # use copy since the continous training requires the original
                # format in maskrcnn
                from qd.qd_common import ensure_copy_file
                if original_model != new_model and \
                        op.isfile(original_model) and \
                        not op.isfile(new_model):
                    ensure_copy_file(original_model, new_model)
        synchronize()

    def get_snapshot_steps(self):
        return 5000

    def ensure_predict(self, epoch=None, iteration=None, model_file=None):
        if self.ignore_predict:
            logging.info('ignore to predict as instructed')
            return

        # deprecate epoch and iteration. use model_file, gradually
        self._ensure_initialized()
        if epoch is not None or iteration is not None:
            assert model_file is None
            logging.warn('use model_file rather than epoch or iteration, pls')
            if epoch is None:
                epoch = self.max_epoch
            model_file = self._get_checkpoint_file(iteration=iteration)
        else:
            if model_file is None:
                model_file = self._get_checkpoint_file()
            assert model_file is not None
        predict_result_file = self._get_predict_file(model_file)
        if not op.isfile(model_file):
            logging.info('ignore to run predict since {} does not exist'.format(
                model_file))
            return predict_result_file
        if not worth_create(model_file, predict_result_file) and not self.force_predict:
            logging.info('ignore to do prediction {}'.format(predict_result_file))
            return predict_result_file

        self.predict(model_file, predict_result_file)

        return predict_result_file

    def _get_test_normalize_module(self):
        if self.test_normalize_module == 'softmax':
            func = torch.nn.Softmax(dim=1)
        elif self.test_normalize_module == 'sigmoid':
            func = torch.nn.Sigmoid()
        else:
            raise Exception('unknown {}'.format(self.test_normalize_module))
        return func

    def predict(self, model_file, predict_result_file):
        if self.mpi_rank == 0:
            train_dataset = TSVDataset(self.data)
            labelmap = train_dataset.load_labelmap()

            model = self._get_model(pretrained=False, num_class=len(labelmap))
            model = self._data_parallel_wrap(model)

            checkpoint = torch_load(model_file)

            model.load_state_dict(checkpoint['state_dict'])

            test_dataset, dataloader = self._get_test_data_loader(self.test_data,
                    self.test_split, labelmap)

            softmax_func = self._get_test_normalize_module()

            # save top 50
            topk = 50
            model.eval()
            def gen_rows():
                pbar = tqdm(total=len(test_dataset))
                for i, (inputs, labels, keys) in enumerate(dataloader):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    output = model(inputs)
                    output = softmax_func(output)

                    all_tops, all_top_indexes = output.topk(topk, dim=1,
                            largest=True, sorted=False)

                    for key, tops, top_indexes in zip(keys, all_tops, all_top_indexes):
                        all_tag = [{'class': labelmap[i], 'conf': float(t)} for t, i in
                                zip(tops, top_indexes)]
                        yield key, json.dumps(all_tag)
                    pbar.update(len(keys))

            tsv_writer(gen_rows(), predict_result_file)
        synchronize()
        return predict_result_file

    def _get_evaluate_file(self, predict_file=None):
        if predict_file is None:
            predict_file = self._get_predict_file()
        assert predict_file.endswith('.tsv')
        cc = [op.splitext(predict_file)[0]]
        if self.evaluate_method != 'map':
            if self.evaluate_method is None:
                return
            cc.append(self.evaluate_method)
        if self.evaluate_method == 'neg_aware_gmap':
            if not self.apply_nms_gt:
                cc.append('noNMSGt')
            if not self.apply_nms_det:
                cc.append('noNMSDet')
            if not self.expand_label_det:
                cc.append('noExpandDet')
        if self.test_version:
            if self.test_version == -1:
                latest_version = TSVDataset(self.test_data).get_latest_version(
                        self.test_split, 'label')
                self.test_version = latest_version
                logging.info('inferred the latest version is {}'.format(
                    latest_version))
            cc.append('v{}'.format(self.test_version))
        if self.coco_eval_max_det is not None and self.coco_eval_max_det != 100:
            cc.append('MaxDet{}'.format(self.coco_eval_max_det))
        if self.pred_tsv_to_json_extra != 1 and \
                self.evaluate_method == 'coco_box':
            cc.append('{}'.format(self.pred_tsv_to_json_extra))
        cc.append('report')
        return '.'.join(cc)

    def ensure_evaluate(self, predict_file=None):
        if self.mpi_rank != 0:
            logging.info('skip because the rank {} != 0'.format(self.mpi_rank))
            return

        # if prediction is disabled, we will not proceed either.
        if self.ignore_evaluate or self.ignore_predict:
            logging.info('ignore evaluate as instructed')
            return

        # not other rank will exit and initalizing distributed will not go
        # through. No need to run initilaization here actually.
        #self._ensure_initialized()
        if not predict_file:
            model_file = self._get_checkpoint_file()
            predict_file = self._get_predict_file(model_file)
        evaluate_file = self._get_evaluate_file(predict_file)
        if evaluate_file is None:
            return
        if not worth_create(predict_file, evaluate_file) and not self.force_evaluate:
            logging.info('ignore {}'.format(evaluate_file))
        else:
            self.evaluate(predict_file, evaluate_file)

        # create index
        self.ensure_create_evaluate_meta_file(evaluate_file)
        return evaluate_file

    def ensure_create_evaluate_meta_file(self, evaluate_file):
        if self.evaluate_method == 'map':
            ensure_create_evaluate_meta_file(evaluate_file)

    def evaluate(self, predict_file, evaluate_file):
        dataset = TSVDataset(self.test_data)

        if self.evaluate_method == 'map':
            from qd.deteval import deteval_iter
            other_param = copy.deepcopy(self.kwargs)
            if 'ovthresh' in other_param:
                del other_param['ovthresh']
            deteval_iter(
                    dataset.iter_data(self.test_split, 'label',
                        version=self.test_version),
                    predict_file,
                    report_file=evaluate_file,
                    ovthresh=self.ovthresh, # this is in self.kwargs already
                    **other_param)
        elif self.evaluate_method == 'coco_box':
            from qd.cocoeval import convert_gt_to_cocoformat
            from qd.cocoeval import convert_to_cocoformat
            from qd.cocoeval import coco_eval_json
            pred_tsv_to_json_extra = self.pred_tsv_to_json_extra
            gt_json = dataset.get_data(self.test_split, 'label.cocoformat',
                    version=self.test_version) + '.json'
            gt_iter = dataset.iter_data(self.test_split, 'label',
                        version=self.test_version)

            if not op.isfile(gt_json) or self.force_evaluate:
                convert_gt_to_cocoformat(gt_iter, gt_json)
            if pred_tsv_to_json_extra == 1:
                predict_json = predict_file + '.cocoformat.json'
            else:
                assert pred_tsv_to_json_extra == 0
                predict_json = predict_file + '.cocoformat.0.json'
            is_empty = False
            if worth_create(predict_file, predict_json) or self.force_evaluate:
                annotations = convert_to_cocoformat(predict_file, predict_json,
                                                    extra=pred_tsv_to_json_extra)
                if len(annotations) == 0:
                    is_empty = True
            else:
                from qd.qd_common import get_file_size
                if get_file_size(predict_json) < 100 and \
                        len(json.loads(read_to_buffer(predict_json))) == 0:
                    is_empty = True
            if is_empty:
                result = {'0.5-all': 0,
                        '0.75-all': 0,
                        'AR-all': 0,
                        'AR-all-1': 0,
                        'AR-all-10': 0,
                        'AR-large': 0,
                        'AR-medium': 0,
                        'AR-small': 0,
                        'all-all': 0,
                        'all-large': 0,
                        'all-medium': 0,
                        'all-small': 0}
            else:
                result = coco_eval_json(predict_json, gt_json,
                        maxDet=self.coco_eval_max_det)

            write_to_yaml_file(result, evaluate_file)
        elif self.evaluate_method == 'top1':
            iter_label = dataset.iter_data(self.test_split, 'label',
                    self.test_version)
            top1 = evaluate_topk(tsv_reader(predict_file), iter_label)
            logging.info('top1 = {}'.format(top1))
            write_to_yaml_file({'top1': top1}, evaluate_file)
        elif self.evaluate_method == 'neg_aware_gmap':
            from qd.evaluate.evaluate_openimages_google import evaluate
            truths = dataset.get_data(self.test_split, 'label')
            imagelabel_truths = dataset.get_data(self.test_split, 'imagelabel')
            assert op.isfile(truths), truths
            assert op.isfile(imagelabel_truths)
            result = evaluate(truths, imagelabel_truths, predict_file,
                    json_hierarchy_file=op.join(dataset._data_root, 'hierarchy.json'),
                    apply_nms_det=self.apply_nms_det,
                    expand_label_det=self.expand_label_det,
                    expand_label_gt=True,
                    apply_nms_gt=self.apply_nms_gt,
                    )
            from qd.qd_common import convert_to_yaml_friendly
            result = convert_to_yaml_friendly(result)
            logging.info(pformat(result))
            logging.info('mAP = {}'.format(result['map']))
            write_to_yaml_file(result, evaluate_file)
        else:
            logging.info('unknown evaluate method = {}'.format(self.evaluate_method))

    def parse_iter(self, i):
        def to_iter(e):
            if type(e) is str and e.endswith('e'):
                if self.num_train_images is None:
                    self.num_train_images = TSVDataset(self.data).num_rows('train')
                iter_each_epoch = 1. * self.num_train_images / self.effective_batch_size
                return int(float(e[:-1]) * iter_each_epoch)
            else:
                return int(e)
        return to_iter(i)

class ModelPipeline(TorchTrain):
    pass

def evaluate_topk(iter_pred_tsv, iter_label_tsv):
    correct = 0
    total = 0
    for (key, str_rects), (key_pred, str_pred) in zip(iter_label_tsv, iter_pred_tsv):
        total = total + 1
        assert key == key_pred
        curr_predict = json.loads(str_pred)
        if len(curr_predict) == 0:
            continue
        curr_gt_rects = json.loads(str_rects)
        if type(curr_gt_rects) is int:
            # imagenet data
            curr_gt_rects = [{'class': str(curr_gt_rects)}]
        curr_pred_best = curr_predict[max(range(len(curr_predict)), key=lambda i: curr_predict[i]['conf'])]['class']
        if any(g['class'] == str(curr_pred_best) for g in curr_gt_rects):
            correct = correct + 1
    return 1. * correct / total

def load_model_state_ignore_mismatch(model, init_dict):
    real_init_dict = {}
    name_to_param = dict(model.named_parameters())
    name_to_param.update(dict(model.named_buffers()))

    def same_shape(a, b):
        return len(a.shape) == len(b.shape) and \
                all(x == y for x, y in zip(a.shape, b.shape))

    num_ignored = 0
    unique_key_in_init_dict = []
    for k in init_dict:
        if k in name_to_param:
            if same_shape(init_dict[k], name_to_param[k]):
                real_init_dict[k] = init_dict[k]
            else:
                logging.info('{} shape is not consistent, expected: {}; got '
                             '{}'.format(k, name_to_param[k].shape, init_dict[k].shape))
        else:
            unique_key_in_init_dict.append(k)
            num_ignored = num_ignored + 1

    logging.info('unique keys in loaded model = {}; total = {}'.format(
        pformat(unique_key_in_init_dict), len(unique_key_in_init_dict),
    ))

    result = model.load_state_dict(real_init_dict, strict=False)
    logging.info('unique key (not initialized) in current model = {}'.format(
        pformat(result.missing_keys),
    ))

    logging.info('loaded key = {}'.format(
        pformat(list(real_init_dict.keys()))))

if __name__ == '__main__':
    init_logging()

    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

