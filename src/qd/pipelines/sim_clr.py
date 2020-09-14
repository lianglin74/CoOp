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
from qd.torch_common import describe_tensor


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim, shuffle=False):
        super(ResNetSimCLR, self).__init__()
        #self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            #"resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model, out_dim)
        num_ftrs = resnet.fc.in_features

        resnet.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
                                  nn.BatchNorm1d(num_ftrs),
                                  nn.ReLU(),
                                  resnet.fc)
        self.module = resnet
        self.shuffle = shuffle
        self.iter = 0

    def _get_basemodel(self, model_name, out_dim):
        model = models.__dict__[model_name](num_classes=out_dim)
        return model

    def forward_one(self, x, verbose=False):
        x = self.module(x)
        if verbose:
            with torch.no_grad():
                logging.info('before norm = {}'.format(
                    describe_tensor(x)
                ))
        x = F.normalize(x, dim=1)
        return x

    def forward(self, x, y):
        verbose = (self.iter % 10 == 0)
        self.iter += 1
        # by default, it is False
        if self.shuffle:
            x = self.forward_one(x)
            y, idx_unshuffle = batch_shuffle_ddp(y)
            y = self.forward_one(y)
            y = batch_unshuffle_ddp(y, idx_unshuffle)
            return x, y
        else:
            z = torch.cat((x, y), dim=0)
            z = self.forward_one(z, verbose)
            x, y = torch.split(z, x.shape[0])
            if verbose:
                with torch.no_grad():
                    sim = torch.matmul(x, y.T)
                    sim.fill_diagonal_(0)
                    logging.info('sim = {}'.format(
                        describe_tensor(sim)))
            return x, y


class SimCLRPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        curr_default = {
            's': 1.,
            'shuffle_bn': False,
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
            'loss_type': 'NTXent',
            'temperature': 0.1,
            'num_workers': 16,

            'queue_alpha': 1.,
            'queue_alpha_policy': None,
            'queue_alpha_max': 1.,

            'criterion_type': None,
            'denominator_ce_factor': 2,

            'cluster_size': 3000,
            'involve_queue_after': 100,
        }
        #from qd.qd_common import max_iter_mult
        #curr_default['stageiter'] = [max_iter_mult(self.max_iter, x / 200.)
                #for x in [120, 160]]
        self._default.update(curr_default)
        if self.shuffle_bn:
            logging.info('set convert_bn as None with shuffle_bn on')
            self.convert_bn = None

    def get_train_model(self):
        model = ResNetSimCLR(
            base_model=self.net,
            out_dim=self.out_dim,
            shuffle=self.shuffle_bn).to(self.device)
        model = self.model_surgery(model)

        criterion = self._get_criterion()
        # we need wrap model output and criterion into one model, to re-use
        # maskrcnn trainer
        from qd.layers.loss import UnsupervisedLoss
        model = UnsupervisedLoss(model, criterion)

        return model

    def get_train_transform(self, start_iter=0):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        from qd.qd_pytorch import BGR2RGB
        gaussian_kernel_size = int(0.1 * self.train_crop_size) // 2 * 2 + 1
        from qd.data_layer.transform import SimCLRGaussianBlur
        data_transforms = transforms.Compose([
            BGR2RGB(),
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=self.train_crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            SimCLRGaussianBlur(kernel_size=gaussian_kernel_size),
            transforms.ToTensor(),
            normalize,
            ])
        from qd.qd_pytorch import TwoCropsTransform
        return TwoCropsTransform(data_transforms)

    def get_test_model(self):
        # used for feasture extraction
        model = models.__dict__[self.net](num_classes=self.out_dim)
        num_ftrs = model.fc.weight.shape[1]
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.BatchNorm1d(num_ftrs),
            nn.ReLU(),
            model.fc)
        if self.dict_trainer:
            from qd.qd_pytorch import InputAsDict
            model = InputAsDict(model)
        return model

    def ensure_evaluate(self):
        pass

