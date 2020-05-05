from qd.qd_pytorch import TwoCropsTransform
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import os.path as op
import logging
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import ImageFilter
from qd.qd_common import print_frame_info


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07,
                 mlp=False, mlp_bn=False, mlp_num=1,
                 with_sim_clr=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        print_frame_info()

        self.K = K
        self.m = m
        self.T = T
        self.with_sim_clr = with_sim_clr

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        # the purpose is to have a key with module, so that our classification
        # module can load such model
        self.module = self.encoder_q

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            def replace_fc(encoder):
                module_list = []
                for i in range(mlp_num):
                    module_list.append(nn.Linear(dim_mlp, dim_mlp))
                    if mlp_bn:
                        module_list.append(nn.BatchNorm1d(dim_mlp))
                    module_list.append(nn.ReLU())
                module_list.append(encoder.fc)
                encoder.fc = nn.Sequential(*module_list)
            replace_fc(self.encoder_q)
            replace_fc(self.encoder_k)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        from qd.qd_common import get_mpi_size, get_mpi_rank
        self.mpi_size = get_mpi_size()
        self.mpi_rank = get_mpi_rank()
        self._use_hvd = None

    @property
    def use_hvd(self):
        if self._use_hvd is None:
            from qd.qd_common import is_hvd_initialized
            self._use_hvd = is_hvd_initialized()
        return self._use_hvd

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, gather=True):
        if gather:
            # gather keys before updating queue
            keys = self.concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def broadcast(self, x):
        if not self.use_hvd:
            if not self.use_hvd:
                torch.distributed.broadcast(x, src=0)
            else:
                from horovod import hvd
                hvd.torch.broadcast(x, src=0)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        if self.mpi_size > 1:
            # broadcast to all gpus
            self.broadcast(idx_shuffle)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        idx_this = idx_shuffle.view(num_gpus, -1)[self.mpi_rank]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = self.concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        idx_this = idx_unshuffle.view(num_gpus, -1)[self.mpi_rank]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        if self.with_sim_clr:
            from qd.qd_pytorch import all_gather_grad_curr, all_gather
            q = all_gather_grad_curr(q)
            k = all_gather(k)
            qk = torch.cat([q, k], dim=0)
            qk_queue = torch.cat([qk, self.queue.detach().T], dim=0)

            bs = q.shape[0]

            sim = torch.matmul(qk, qk_queue.T)
            sim /= self.T

            sim.fill_diagonal_(-float('inf'))
            label1 = torch.arange(bs, bs + bs).to(sim.device)
            label2 = torch.arange(bs).to(sim.device)
            label = torch.cat([label1, label2], dim=0)

            self._dequeue_and_enqueue(k, gather=False)
            return sim, label
        else:
            return self.forward_moco(q, k)

    def forward_moco(self, q, k):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


    # utils
    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        if self.mpi_size == 1:
            return tensor
        if not self.use_hvd:
            tensors_gather = [torch.ones_like(tensor)
                for _ in range(self.mpi_size)]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
            return output
        else:
            import horovod as hvd
            output = hvd.torch.allgather(tensor)
            return output

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class MocoPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        curr_default = {
            'data': 'imagenet2012',
            'net': 'resnet50',
            #'epochs': 200,
            'start_epoch': 0,
            #'batch_size': 256,
            'learning_rate': 0.03,
            'scheduler_type': 'multi_step',
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'moco_dim': 128,
            'moco_k': 65536,
            'moco_m': 0.999,
            'moco_t': 0.07,
            'mlp': False,
            'aug_plus': False,
            'cos': False,
            'log_step': 100,
            'ignore_predict': True,
            'mlp_num': 1,
        }

        from qd.qd_common import max_iter_mult
        curr_default['stageiter'] = [max_iter_mult(self.max_iter, x / 200.)
                for x in [120, 160]]
        self._default.update(curr_default)

    def get_train_model(self):
        model = MoCo(
            models.__dict__[self.net],
            self.moco_dim,
            self.moco_k,
            self.moco_m,
            self.moco_t,
            self.mlp,
            mlp_bn=self.mlp_bn,
            mlp_num=self.mlp_num,
            with_sim_clr=self.with_sim_clr)

        criterion = self._get_criterion()
        # we need wrap model output and criterion into one model, to re-use
        # maskrcnn trainer
        from qd.layers.loss import UnsupervisedLoss
        model = UnsupervisedLoss(model, criterion)
        return model

    def get_test_model(self):
        model = models.__dict__[self.net](num_classes=self.moco_dim)
        if self.mlp:
            model.fc = nn.Sequential(
                nn.Linear(
                    model.fc.weight.shape[1],
                    model.fc.weight.shape[1]),
                nn.ReLU(),
                model.fc)
        return model

    def get_train_transform(self, start_iter=0):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        from qd.qd_pytorch import BGR2RGB
        if self.aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            augmentation = [
                BGR2RGB(),
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:
            # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
            augmentation = [
                BGR2RGB(),
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]

        return TwoCropsTransform(transforms.Compose(augmentation))

    def ensure_evaluate(self):
        pass
