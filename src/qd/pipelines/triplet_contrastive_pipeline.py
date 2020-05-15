import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from qd.qd_pytorch import batch_shuffle_ddp, batch_unshuffle_ddp
import logging
import cv2
import PIL
import torchvision


def calculate_giou_x1y1x2y2(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    a = (ax2 - ax1) * (ay2 - ay1)
    b = (bx2 - bx1) * (by2 - by1)
    max_x1 = max(ax1, bx1)
    max_y1 = max(ay1, by1)
    min_x2 = min(ax2, bx2)
    min_y2 = min(ay2, by2)
    if min_x2 > max_x1 and min_y2 > max_y1:
        inter = (min_x2 - max_x1) * (min_y2 - max_y1)
    else:
        inter = 0
    min_x1 = min(ax1, bx1)
    min_y1 = min(ay1, by1)
    max_x2 = max(ax2, by2)
    max_y2 = max(ay2, by2)
    cover = (max_x2 - min_x1) * (max_y2 - min_y1)
    union = a + b - inter
    iou = inter / union
    giou = iou - (cover - union) / cover
    return giou

def y1x1hw_to_x1y1x2y2(box):
    y, x, h, w = box
    return x, y, x + w, y + h

def calculate_giou_y1x1hw(box1, box2):
    box1 = y1x1hw_to_x1y1x2y2(box1)
    box2 = y1x1hw_to_x1y1x2y2(box2)

    return calculate_giou_x1y1x2y2(box1, box2)

class TripletRandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=PIL.Image.BILINEAR, num_trial=50):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.not_found = 0
        self.num_trial = num_trial

    def __repr__(self):
        return ('TripletRandomResizedCrop(size={}, scale={}, ratio={}, '
                'interpolation={}, num_trial={})'.format(
                    self.size,
                    self.scale,
                    self.ratio,
                    self.interpolation,
                    self.num_trial))

    def get_params(self, img):
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
            import random, math
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

    def __call__(self, img):
        anchor = self.get_params(img)
        boxes = [self.get_params(img) for _ in range(self.num_trial)]
        gious = [calculate_giou_y1x1hw(b, anchor) for b in boxes]
        idx_giou = list(enumerate(gious))
        idx_giou = sorted(idx_giou, key=lambda x: x[1])
        img0 = torchvision.transforms.functional.resized_crop(img, anchor[0], anchor[1], anchor[2], anchor[3],
                              self.size, self.interpolation)
        i1, j1, h1, w1 = boxes[idx_giou[-1][0]]
        img1 = torchvision.transforms.functional.resized_crop(img, i1, j1, h1, w1, self.size, self.interpolation)
        i2, j2, h2, w2 = boxes[idx_giou[0][0]]
        img2 = torchvision.transforms.functional.resized_crop(img, i2, j2, h2, w2, self.size, self.interpolation)

        #i, j, h, w = anchor
        #from PIL import ImageDraw
        #img_draw = ImageDraw.Draw(img)
        #img_draw.rectangle([j, i, j + w, i + h], outline='red')
        #img_draw.rectangle([j1, i1, j1 + w1, i1 + h1], outline='black')
        #img_draw.rectangle([j2, i2, j2 + w2, i2 + h2])
        #img.show()
        #img0.show()
        #img1.show()
        #img2.show()
        #import ipdb;ipdb.set_trace(context=15)
        return img0, img1, img2

class MultiCropsTransformX():
    def __init__(self, num, aug1, aug_join, aug2):
        self.aug1 = aug1
        self.aug_join = aug_join
        self.aug2 = aug2
        self.num = num

    def __repr__(self):
        s = 'TwoCropsTransformX(num={}, aug1={}, aug_join={}, aug2={})'.format(
            self.num,
            self.aug1, self.aug_join, self.aug2,
        )
        return s

    def __call__(self, x):
        y = self.aug1(x)
        ys = self.aug_join(y)
        ys = [self.aug2(y) for y in ys]
        return ys

class ResNetTripletContrast(nn.Module):
    def __init__(self, base_model, out_dim, temperature):
        super().__init__()
        #self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            #"resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model, out_dim)
        num_ftrs = resnet.fc.in_features

        resnet.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
                                  nn.BatchNorm1d(num_ftrs),
                                  nn.ReLU(),
                                  resnet.fc)
        self.module = resnet
        self.temperature = temperature

    def _get_basemodel(self, model_name, out_dim):
        model = models.__dict__[model_name](num_classes=out_dim)
        return model

    def forward_one(self, x):
        x = self.module(x)
        x = F.normalize(x, dim=1)
        return x

    def forward(self, x, y, z):
        # by default, it is False
        z = torch.cat((x, y, z), dim=0)
        z = self.forward_one(z)
        s = z.shape[0] // 3
        x, y, z = torch.split(z, [s, s, s])
        from qd.qd_pytorch import all_gather_grad_curr
        x = all_gather_grad_curr(x)
        y = all_gather_grad_curr(y)
        z = all_gather_grad_curr(z)
        xyz = torch.cat([x, y, z], dim=0)
        sim = torch.matmul(x, xyz.T)
        sim /= self.temperature
        sim.fill_diagonal_(-float('inf'))
        ebs = x.shape[0]
        label = torch.arange(ebs, ebs * 2).to(x.device)

        sim2 = sim.clone()
        x_idx = torch.arange(ebs)
        y_idx = label
        sim2[x_idx, y_idx] = -float('inf')
        label2 = torch.arange(ebs * 2, ebs * 3).to(x.device)

        return sim, label, sim2, label2

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


class TripletContrastive(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        curr_default = {
            's': 1.,
            'convert_bn': 'SBN',
            'out_dim': 128,
            'weight_decay': 1e-6,
            # the best result is supposed to be achieved with 4096 batch
            # size and linearly scale up the learning rate
            'base_lr': 0.3,
            'effective_batch_size': 256,
            'max_iter': '100e',
            'bias_no_weight_decay': True,
            'optimizer_type': 'LARS',
            'scheduler_type': 'cosine',
            'cosine_warmup_iters': '10e',
            'cosine_warmup_factor': 1.,
            'temperature': 0.1,
            'num_workers': 16,
            'loss_type': 'multi_ce',
        }
        #from qd.qd_common import max_iter_mult
        #curr_default['stageiter'] = [max_iter_mult(self.max_iter, x / 200.)
                #for x in [120, 160]]
        self._default.update(curr_default)
        if self.shuffle_bn:
            logging.info('set convert_bn as None with shuffle_bn on')
            self.convert_bn = None

    def get_train_transform(self, start_iter=0):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        from qd.qd_pytorch import BGR2RGB
        gaussian_kernel_size = int(0.1 * self.train_crop_size) // 2 * 2 + 1
        aug1 = [
            BGR2RGB(),
            transforms.ToPILImage(),
        ]
        aug1 = transforms.Compose(aug1)
        aug_crop = TripletRandomResizedCrop(self.train_crop_size, scale=(0.2, 1.))
        aug2 = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            SimCLRGaussianBlur(kernel_size=gaussian_kernel_size),
            transforms.ToTensor(),
            normalize,
        ]
        aug2 = transforms.Compose(aug2)
        return MultiCropsTransformX(3, aug1, aug_crop, aug2)

    def get_train_model(self):
        model = ResNetTripletContrast(
            base_model=self.net,
            out_dim=self.out_dim,
            temperature=self.temperature,
        ).to(self.device)
        model = self.model_surgery(model)

        criterion = self._get_criterion()
        # we need wrap model output and criterion into one model, to re-use
        # maskrcnn trainer
        from qd.layers.loss import UnsupervisedLoss
        model = UnsupervisedLoss(model, criterion)

        return model

    def ensure_predict(self):
        pass

    def ensure_evaluate(self):
        pass


