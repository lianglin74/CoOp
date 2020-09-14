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
from qd.qd_pytorch import GaussianBlur
from qd.data_layer.transform import ImageToImageDictTransform


__debug_info = {}
class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07,
                 mlp=False, mlp_bn=False, mlp_num=1,
                 key_mlp_num=None,
                 with_sim_clr=None,
                 multi_crop=False, dim_mlp=None):
        super(MoCo, self).__init__()
        print_frame_info()

        self.K = K
        self.m = m
        self.T = T
        self.with_sim_clr = with_sim_clr
        assert self.with_sim_clr is None
        self.multi_crop = multi_crop

        # create the encoders
        # num_classes is the output fc dimension
        encoder_q = base_encoder(num_classes=dim)
        encoder_k = base_encoder(num_classes=dim)

        if mlp:
            fc_in_dim = encoder_q.fc.weight.shape[1]
            if dim_mlp is None:
                dim_mlp = fc_in_dim
            out_dim = encoder_q.fc.weight.shape[0]
            def replace_fc(encoder, num):
                module_list = []
                for i in range(num):
                    if i == 0:
                        dim_mlp_in = encoder.fc.weight.shape[1]
                    else:
                        dim_mlp_in = dim_mlp
                    module_list.append(nn.Linear(dim_mlp_in, dim_mlp))
                    if mlp_bn:
                        module_list.append(nn.BatchNorm1d(dim_mlp))
                    module_list.append(nn.ReLU())
                if fc_in_dim == dim_mlp:
                    module_list.append(encoder.fc)
                else:
                    module_list.append(nn.Linear(dim_mlp, out_dim))
                encoder.fc = nn.Sequential(*module_list)
            if key_mlp_num is None:
                key_mlp_num = mlp_num
            assert key_mlp_num <= mlp_num
            replace_fc(encoder_q, mlp_num)
            replace_fc(encoder_k, key_mlp_num)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_im_idx", torch.zeros(K))
        self.queue_im_idx[:] = -1
        self.register_buffer('queue_im_crop', torch.zeros(4, K))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        from qd.qd_common import get_mpi_size, get_mpi_rank
        self.mpi_size = get_mpi_size()
        self.mpi_rank = get_mpi_rank()
        self._use_hvd = None

        if self.mpi_size > 1:
            from qd.qd_pytorch import replace_module
            encoder_q = replace_module(encoder_q,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d) or
                                   isinstance(m, torch.nn.BatchNorm1d),
                    lambda m: torch.nn.SyncBatchNorm(m.num_features,
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=m.affine,
                        track_running_stats=m.track_running_stats))
            encoder_k = replace_module(encoder_k,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d) or
                                   isinstance(m, torch.nn.BatchNorm1d),
                    lambda m: torch.nn.SyncBatchNorm(m.num_features,
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=m.affine,
                        track_running_stats=m.track_running_stats))

        #for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            #param_k.data.copy_(param_q.data)  # initialize

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        # the purpose is to have a key with module, so that our classification
        # module can load such model
        self.module = self.encoder_q
        self.iter = 0
        self.buffer = {}

    @property
    def use_hvd(self):
        if self._use_hvd is None:
            from qd.qd_common import is_hvd_initialized
            self._use_hvd = is_hvd_initialized()
        return self._use_hvd

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, crop=None, im_index=None, gather=True):
        if gather:
            # gather keys before updating queue
            keys = self.concat_all_gather(keys)
            if crop is not None:
                crop = self.concat_all_gather(crop)
            if im_index is not None:
                im_index = self.concat_all_gather(im_index)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        if crop is not None:
            self.queue_im_crop[:, ptr: ptr + batch_size] = crop.T
        if im_index is not None:
            self.queue_im_idx[ptr: ptr + batch_size] = im_index

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

    def forward(self, *args):
        if self.multi_crop:
            raise NotImplementedError
            return self.forward_multi(*args)
        else:
            return self.forward_two(*args)

    def forward_key(self, im_k):
        k = self.encoder_k(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)

        return k

    def forward_multi(self, *args):
        # one is query, one is key
        assert len(args[0]) == 2
        im_q = args[0][0]['image']
        im_k = args[0][1]['image']
        verbose = (self.iter % 100) == 0

        k = self.forward_key(im_k)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        result = []
        result.extend(self.forward_moco(q, k, verbose=verbose))

        for i, other_crops in enumerate(args[1:]):
            other_image_qs = torch.cat([other_crop['image'] for other_crop in other_crops],
                      dim=0)
            other_qs = self.encoder_q(other_image_qs)
            other_qs = nn.functional.normalize(other_qs, dim=1)
            expand_k = k[None, ...].expand(len(other_crops), *k.shape)
            expand_k = expand_k.reshape(-1, expand_k.shape[-1])
            result.extend(self.forward_moco(other_qs, expand_k, verbose=verbose))

        self._dequeue_and_enqueue(k)
        self.iter += 1

        return result

    def forward_two(self, in_im_q, in_im_k):
        # if the input is a dictionary, it means it has other informations,
        # e.g. image index, crop coordinates.
        verbose = (self.iter % 10) == 0
        info = []
        im_q = in_im_q['image']
        im_k = in_im_k['image']

        #__debug_info['in_im_q'] = in_im_q
        #__debug_info['in_im_k'] = in_im_k

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        if verbose:
            from qd.torch_common import describe_tensor
            info.append('before norm q = {}'.format(describe_tensor(q)))
        q = nn.functional.normalize(q, dim=1)

        k = self.encoder_k(im_k)  # keys: NxC
        if verbose:
            info.append('before norm k = {}'.format(describe_tensor(k)))
        k = nn.functional.normalize(k, dim=1)

        if verbose:
            with torch.no_grad():
                if 'old_k' in self.buffer:
                    diff = (self.buffer['old_k'] - k).abs().mean() / k.abs().mean()
                    info.append('key diff = {}'.format(diff))
                    sim = torch.matmul(self.buffer['old_k'], k.T)
                    sim.fill_diagonal_(0.)
                    info.append('{:.2f}/{:.2f}/{:.2f}'.format(sim.min(), sim.mean(),
                                                   sim.max()))

                self.buffer['old_k'] = k

        result = self.forward_moco(q, k, verbose=verbose)

        self._dequeue_and_enqueue(k, crop=in_im_k['crop'],
                                  im_index=in_im_k['index'])
        self.iter += 1
        if verbose:
            logging.info('; '.join(info))
        return result

    def forward_moco(self, q, k, verbose=False):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        queue = self.queue.clone().detach()
        l_neg_q = torch.einsum('nc,ck->nk', [q, queue])
        l_neg_k = torch.einsum('nc,ck->nk', [k, queue])

        logits = torch.cat([l_pos, l_neg_q, l_neg_k], dim=1)


        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        if verbose:
            from qd.torch_common import accuracy
            top1, top5 = accuracy(logits, labels, (1, 5))
            logging.info('acc1 = {:.2f}; acc5 = {:.2f}; '
                         'pos={:.2f}/{:.2f}/{:.2f}; '
                         'neg_q={:.2f}/{:.2f}/{:.2f}; neg_k={:.2f}/{:.2f}/{:.2f}'.format(
                             float(top1), float(top5),
                             l_pos.min(), l_pos.mean(), l_pos.max(),
                             l_neg_q.min(), l_neg_q.mean(), l_neg_q.max(),
                             l_neg_k.min(), l_neg_k.mean(), l_neg_k.max(),
                         ))
        logits /= self.T

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

class SlowContrastPipeline(MaskClassificationPipeline):
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
            'aug_plus': None,
            'cos': False,
            'log_step': 100,
            'ignore_predict': True,
            'mlp_num': 1,
            'aug_plus_iou': 0.5,
            'dict_trainer': True,
            'dim_mlp': None,
            'slow_x': 1e-2,
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
            with_sim_clr=self.with_sim_clr,
            multi_crop=self.aug_plus=='multi_crop',
            dim_mlp=self.dim_mlp,
        )

        criterion = self._get_criterion()
        # we need wrap model output and criterion into one model, to re-use
        # maskrcnn trainer
        from qd.layers.loss import UnsupervisedLoss
        model = UnsupervisedLoss(model, criterion)
        return model

    def get_optimizer(self, model):
        parameters = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.base_lr
            if 'encoder_k' in key:
                lr *= self.slow_x
                continue
            else:
                assert 'encoder_q' in key
            weight_decay = self.weight_decay
            if "bias" in key:
                weight_decay = 0.
            logging.info('{}: lr = {}; weight_decay = {}'.format(
                key, lr, weight_decay
            ))
            parameters += [{"params": [value],
                            "lr": lr,
                            "weight_decay": weight_decay,
                            'param_names': [key]}]

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
        return optimizer

    def get_test_model(self):
        # used for feasture extraction
        model = models.__dict__[self.net](num_classes=self.moco_dim)
        if self.mlp:
            model.fc = nn.Sequential(
                nn.Linear(
                    model.fc.weight.shape[1],
                    model.fc.weight.shape[1]),
                nn.ReLU(),
                model.fc)
        if self.dict_trainer:
            from qd.qd_pytorch import InputAsDict
            model = InputAsDict(model)
        return model

    def get_train_transform(self, start_iter=0):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        from qd.qd_pytorch import BGR2RGB
        if self.aug_plus == 'plus_fix':
            aug1 = [
                BGR2RGB(),
                transforms.ToPILImage(),
            ]
            aug1 = transforms.Compose(aug1)
            aug1 = ImageToImageDictTransform(aug1)

            from qd.data_layer.transform import RandomResizedCropDict
            crop = RandomResizedCropDict(224, scale=(0.2, 1.))
            aug2 = [
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
            aug2 = transforms.Compose(aug2)
            aug2 = ImageToImageDictTransform(aug2)

            final = transforms.Compose([aug1, crop, aug2])
            fix = self.get_test_transform()
            return TwoCropsTransform(final, fix)
        elif self.aug_plus == 'plus_dict':
            aug1 = [
                BGR2RGB(),
                transforms.ToPILImage(),
            ]
            aug1 = transforms.Compose(aug1)
            aug1 = ImageToImageDictTransform(aug1)

            from qd.data_layer.transform import RandomResizedCropDict
            crop = RandomResizedCropDict(224, scale=(0.2, 1.))
            aug2 = [
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
            aug2 = transforms.Compose(aug2)
            aug2 = ImageToImageDictTransform(aug2)

            final = transforms.Compose([aug1, crop, aug2])
            return TwoCropsTransform(final, final)
        elif self.aug_plus == 'multi_crop':
            from qd.data_layer.transform import MultiCropsTransform
            def get_by_size(size):
                augmentation = [
                    BGR2RGB(),
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(size, scale=(0.2, 1.)),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]
                aug = transforms.Compose(augmentation)
                aug = ImageToImageDictTransform(aug)
                return aug
            all_info = [
                {
                    'repeat': 2,
                    'transform': get_by_size(self.train_crop_size),
                },
                {
                    'repeat': self.num_small_crop,
                    'transform': get_by_size(self.small_crop_size),
                },
            ]
            aug = MultiCropsTransform(all_info)
            return aug
        elif self.aug_plus == 'PlusIoU':
            logging.info('method does not help')
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            aug1 = [
                BGR2RGB(),
                transforms.ToPILImage(),
            ]
            aug1 = transforms.Compose(aug1)
            from qd.qd_pytorch import IoURandomResizedCrop
            aug_crop = IoURandomResizedCrop(self.aug_plus_iou, 224, scale=(0.2, 1.))
            aug2 = [
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
            aug2 = transforms.Compose(aug2)
            from qd.qd_pytorch import TwoCropsTransformX
            return TwoCropsTransformX(aug1, aug_crop, aug2)

    def ensure_evaluate(self):
        pass
