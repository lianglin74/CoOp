from qd.qd_pytorch import TwoCropsTransform
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import os.path as op
import logging

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
from qd.torch_common import replace_module
from qd.qd_common import get_mpi_size, get_mpi_rank


class MultiMLP(nn.ModuleList):
    def __init__(self, in_dim, out_dims):
        all_module = [nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim),
        ) for out_dim in out_dims]
        super().__init__(all_module)
    def forward(self, x):
        return [l(x) for l in self]

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07,
                 mlp=False, mlp_bn=False, mlp_num=1,
                 key_mlp_num=None,
                 with_sim_clr=None, shuffle_bn=True,
                 multi_crop=False, dim_mlp=None,
                 multi_mlp=None,
                 moco_cluster=False,
                 moco_cluster_size=3000,
                 mlp_as_conv=False,
                 sinkhorn_func=None,
                 sinkhorn_eps=0.05,
                 cluster_skip_to_use_queue=200,
                 cluster_sinkhorn_iter=3,
                 soft_lambda=1.,
                 soft_mu=0.,
                 ):
        # multi_mlp: {'four_rotate_idx': 4}, {'is_color': 2}, the value is
        # dimention
        # mlp should always be true for better accuracy
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        print_frame_info()

        if moco_cluster:
            if sinkhorn_func == 'dist':
                # for this option, the queue is also distributed. Thus, each worker
                # can only keep a portion of it.
                assert (K % get_mpi_size()) == 0
                K = K // get_mpi_size()

        self.K = K
        self.m = m
        self.T = T
        self.with_sim_clr = with_sim_clr
        self.multi_crop = multi_crop
        self.multi_mlp = multi_mlp
        self.moco_cluster = moco_cluster
        self.moco_cluster_size = moco_cluster_size

        self.cluster_sinkhorn_eps = sinkhorn_eps
        self.cluster_skip_to_use_queue = cluster_skip_to_use_queue
        self.cluster_sinkhorn_iter = cluster_sinkhorn_iter

        # create the encoders
        # num_classes is the output fc dimension
        encoder_q = base_encoder(num_classes=dim)
        encoder_k = base_encoder(num_classes=dim)

        if mlp:
            if multi_mlp:
                def replace_fc_with_multi_mlp(encoder, extra_dims):
                    dim_mlp_in = encoder.fc.weight.shape[1]
                    out_dims = [encoder.fc.weight.shape[0]]
                    out_dims.extend(extra_dims)
                    encoder.fc = MultiMLP(dim_mlp_in, out_dims)
                extra_dims = [v for k, v in multi_mlp.items()]
                replace_fc_with_multi_mlp(encoder_q, extra_dims) # rotation and color/gray
                replace_fc_with_multi_mlp(encoder_k, [])
                assert not multi_crop
            elif mlp_as_conv:
                def remove_module(model, t):
                    model = replace_module(model,
                            lambda m: isinstance(m, t),
                            lambda m: nn.Identity(),
                            )
                    return model
                encoder_q = remove_module(encoder_q, nn.AdaptiveAvgPool2d)
                encoder_k = remove_module(encoder_k, nn.AdaptiveAvgPool2d)

                encoder_q = remove_module(encoder_q, nn.Flatten)
                encoder_k = remove_module(encoder_k, nn.Flatten)

                fc_in_dim = encoder_q.fc.weight.shape[1]
                if dim_mlp is None:
                    dim_mlp = fc_in_dim
                out_dim = encoder_q.fc.weight.shape[0]
                def replace_fc_as_conv(encoder, num):
                    module_list = []
                    for i in range(num):
                        if i == 0:
                            dim_mlp_in = encoder.fc.weight.shape[1]
                        else:
                            dim_mlp_in = dim_mlp
                        module_list.append(nn.Conv2d(dim_mlp_in, dim_mlp,
                                                     kernel_size=1))
                        if mlp_bn:
                            module_list.append(nn.BatchNorm2d(dim_mlp))
                        module_list.append(nn.ReLU())
                    module_list.append(nn.Conv2d(dim_mlp, out_dim, kernel_size=1))
                    encoder.fc = nn.Sequential(*module_list)
                if key_mlp_num is None:
                    key_mlp_num = mlp_num
                assert key_mlp_num <= mlp_num
                replace_fc_as_conv(encoder_q, mlp_num)
                replace_fc_as_conv(encoder_k, key_mlp_num)
            else:
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
        else:
            assert not mlp and not multi_mlp

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_im_idx", torch.zeros(K))
        self.queue_im_idx[:] = -1
        self.register_buffer('queue_im_crop', torch.zeros(4, K))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.mpi_size = get_mpi_size()
        self.mpi_rank = get_mpi_rank()
        self._use_hvd = None
        self.shuffle_bn = shuffle_bn

        if not shuffle_bn:
            logging.info('converting bn to sync-bn due to shuffle-bn is not '
                         'enabled')
            if self.mpi_size > 1:
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

        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if self.moco_cluster:
            cluster_center = torch.randn(
                self.moco_cluster_size,
                dim)
            cluster_center = nn.functional.normalize(cluster_center, dim=1)
            self.cluster_center = nn.Parameter(cluster_center)

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        # the purpose is to have a key with module, so that our classification
        # module can load such model
        self.module = self.encoder_q
        self.iter = 0

        self.queue_gather = True
        self.sinkhorn_func = sinkhorn_func
        if sinkhorn_func is None:
            from qd.torch_common import sinkhorn
            self.sinkhorn_func = sinkhorn
        elif sinkhorn_func == 'correct_sum':
            from qd.torch_common import sinkhorn_correct_sum
            self.sinkhorn_func = sinkhorn_correct_sum
        elif sinkhorn_func == 'dist':
            from qd.torch_common import distributed_sinkhorn
            self.sinkhorn_func = distributed_sinkhorn
            self.queue_gather = False
        else:
            raise NotImplementedError

        if sinkhorn_func == 'soft':
            self.register_buffer("prior", torch.zeros(moco_cluster_size))
            self.prior[:] = 1. / moco_cluster_size


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

    def forward(self, *args):
        if self.multi_crop:
            return self.forward_multi(*args)
        else:
            return self.forward_two(*args)

    def forward_key(self, im_k):
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.shuffle_bn:
                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            if self.shuffle_bn:
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
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
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # if the input is a dictionary, it means it has other informations,
        # e.g. image index, crop coordinates.
        verbose = (self.iter % 100) == 0
        assert isinstance(in_im_q, dict)
        im_q = in_im_q['image']
        im_k = in_im_k['image']

        # compute query features
        if self.with_sim_clr in ['qqk', 'qq_qk']:
            im = torch.cat([im_q, im_k], dim=0)
            q = self.encoder_q(im)
            q = nn.functional.normalize(q, dim=1)
        else:
            if not self.multi_mlp:
                q = self.encoder_q(im_q)  # queries: NxC
                q = nn.functional.normalize(q, dim=1)
            else:
                # the first output is the moco's output, others are aux losses
                qs = self.encoder_q(im_q)  # queries: NxC
                q = nn.functional.normalize(qs[0], dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.shuffle_bn:
                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            if not self.multi_mlp:
                k = self.encoder_k(im_k)  # keys: NxC
            else:
                # key encoder only outputs 1 entry
                ks = self.encoder_k(im_k)  # keys: NxC
                assert len(ks) == 1
                k = ks[0]
            k = nn.functional.normalize(k, dim=1)

            if self.shuffle_bn:
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        if self.with_sim_clr:
            result = self.forward_with_sim_clr(q, k)
        elif self.sinkhorn_func == 'soft':
            result = self.forward_soft_cluster(q, k)
        elif self.moco_cluster:
            result = self.forward_cluster(q, k)
        else:
            result = self.forward_moco(q, k, verbose=verbose)

        if self.multi_mlp:
            # result is a tuple
            result = list(result)
            # we need to add other losses, which are also based on cross
            # entropy loss
            for keyword, extra_out in zip(self.multi_mlp, qs[1:]):
                label = in_im_q[keyword]
                result.extend([extra_out, label])
            result = tuple(result)

        if self.sinkhorn_func != 'soft':
            self._dequeue_and_enqueue(
                k, crop=in_im_k['crop'],
                im_index=in_im_k['index'],
                gather=self.queue_gather,
            )
        self.iter += 1
        return result

    def forward_with_sim_clr(self, q, k):
        from qd.qd_pytorch import all_gather_grad_curr, all_gather
        if self.with_sim_clr == 'qk_queue':
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

            return sim, label
        elif self.with_sim_clr == 'qqk':
            # in this case, q is the feature from im_q and im_k by stacking.
            assert q.shape[0] == 2 * k.shape[0]
            assert not self.shuffle_bn, ('all gather does not support gradient '
                                         'propagation, so it is better to use '
                                         'sync-bn. here key encoder will also '
                                         'use sync-bn')
            q1 = q[:k.shape[0]]
            q2 = q[k.shape[0]:]
            q1_all = all_gather_grad_curr(q1)
            q2_all = all_gather_grad_curr(q2)
            qq = torch.cat([q1_all, q2_all], dim=0)

            qq_queue = torch.cat([qq, self.queue.detach().T], dim=0)

            sim = torch.matmul(qq, qq_queue.T)
            sim /= self.T
            bs = qq.shape[0] // 2
            sim.fill_diagonal_(-float('inf'))
            label1 = torch.arange(bs, bs + bs).to(sim.device)
            label2 = torch.arange(bs).to(sim.device)
            label = torch.cat([label1, label2], dim=0)
            return sim, label
        elif self.with_sim_clr == 'qq_qk':
            assert q.shape[0] == 2 * k.shape[0]
            assert not self.shuffle_bn, ('all gather does not support gradient '
                                         'propagation, so it is better to use '
                                         'sync-bn. here key encoder will also '
                                         'use sync-bn')
            q1 = q[:k.shape[0]]
            q2 = q[k.shape[0]:]
            q1_all = all_gather_grad_curr(q1)
            q2_all = all_gather_grad_curr(q2)
            qq = torch.cat([q1_all, q2_all], dim=0)
            sim = torch.matmul(qq, qq.T)
            sim /= self.T
            sim.fill_diagonal_(-float('inf'))
            bs = qq.shape[0] // 2
            label1 = torch.arange(bs, bs + bs).to(sim.device)
            label2 = torch.arange(bs).to(sim.device)
            label = torch.cat([label1, label2], dim=0)

            sim2, label2 = self.forward_moco(q[:k.shape[0]], k)
            return sim, label, sim2, label2
        else:
            raise NotImplementedError

    def forward_soft_cluster(self, q, k, verbose=False):
        #from qd.torch_common import concat_all_gather
        bs = k.shape[0]
        #k = concat_all_gather(k)

        norm_cluster_center = torch.nn.functional.normalize(
            self.cluster_center,
            dim=1)
        # Q: B * K
        with torch.no_grad():
            sim_mat = torch.matmul(k, norm_cluster_center.T)
            hat_sim_mat = sim_mat - self.prior
            curr_code = hat_sim_mat.max(dim=1)

        # calculate the loss
        pred = torch.matmul(q, norm_cluster_center.T) / self.T
        return pred, curr_code

    def forward_cluster(self, q, k, verbose=False):
        #from qd.torch_common import concat_all_gather
        bs = k.shape[0]
        #k = concat_all_gather(k)

        # samples, B
        if self.iter > self.cluster_skip_to_use_queue:
            k = torch.cat([k, self.queue.clone().T.detach()], dim=0)

        norm_cluster_center = torch.nn.functional.normalize(
            self.cluster_center,
            dim=1)
        # B * K
        # no need to divide it by temperature.
        sim_mat = torch.matmul(k, norm_cluster_center.T)
        # Q: B * K
        with torch.no_grad():
            Q = self.sinkhorn_func(
                sim_mat,
                eps=self.cluster_sinkhorn_eps,
                niters=self.cluster_sinkhorn_iter)
            # the code for current batch size
            curr_code = Q[: bs]

        # calculate the loss
        pred = torch.matmul(q, norm_cluster_center.T) / self.T
        return pred, curr_code

    def forward_moco(self, q, k, verbose=False):
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        #----------------------------
        #logging.info('debugging')
        #topk, idx = l_neg.topk(dim=1, topk=50).detach().cpu()

        #import matplotlib.pyplot as plt
        #for l in logits.detach().cpu():
            #plt.hist(l.numpy())
            #plt.show()

        # apply temperature
        logits /= self.T


        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # the accuracy will be logged in the loss calculation
        #if verbose:
            #from qd.torch_common import accuracy
            #top1, top5 = accuracy(logits, labels, (1, 5))
            #logging.info('acc1 = {:.1f}; acc5 = {:.1f}'.format(
                #float(top1), float(top5)))

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
            'aug_plus': None,
            'cos': False,
            'log_step': 100,
            'ignore_predict': True,
            'mlp_num': 1,
            'shuffle_bn': True,
            'aug_plus_iou': 0.5,
            'dict_trainer': True,
            'dim_mlp': None,
            'scale_min': 0.2,
            'scale_max': 1.,
            'small_scale_min': 0.2,
            'small_scale_max': 0.2,
            'moco_cluster': False,
            'moco_cluster_size': 3000,
            'cluster_sinkhorn_eps': 0.05,
            'cluster_skip_to_use_queue': 200,
            'cluster_sinkhorn_iter': 3,
        }

        from qd.qd_common import max_iter_mult
        curr_default['stageiter'] = [max_iter_mult(self.max_iter, x / 200.)
                for x in [120, 160]]
        self._default.update(curr_default)

    def get_train_model(self):
        encoder = self.get_encoder()
        model = MoCo(
            encoder,
            self.moco_dim,
            self.moco_k,
            self.moco_m,
            self.moco_t,
            self.mlp,
            mlp_bn=self.mlp_bn,
            mlp_num=self.mlp_num,
            with_sim_clr=self.with_sim_clr,
            shuffle_bn=self.shuffle_bn,
            multi_crop=self.aug_plus.startswith('multi_crop'),
            dim_mlp=self.dim_mlp,
            multi_mlp=self.multi_mlp,
            moco_cluster=self.moco_cluster,
            moco_cluster_size=self.moco_cluster_size,
            mlp_as_conv=self.mlp_as_conv,
            sinkhorn_func=self.sinkhorn_func,
            sinkhorn_eps=self.cluster_sinkhorn_eps,
            cluster_skip_to_use_queue=self.cluster_skip_to_use_queue,
            cluster_sinkhorn_iter=self.cluster_sinkhorn_iter
        )

        criterion = self._get_criterion()
        # we need wrap model output and criterion into one model, to re-use
        # maskrcnn trainer
        from qd.layers.loss import UnsupervisedLoss
        model = UnsupervisedLoss(model, criterion)
        return model

    def get_encoder(self):
        if self.net.startswith('efficientnet'):
            from qd.layers.efficient_det import EfficientNet
            encoder = lambda num_classes: EfficientNet.from_name(
                self.net,
                override_params={'num_classes': num_classes})
        else:
            from qd.qd_common import execute_func
            encoder = lambda num_classes: execute_func({
                'from': 'qd.pipelines.fb_moco',
                'import': self.net,
                'param': {'num_classes': num_classes},})
        return encoder

    def get_test_model(self):
        # used for feasture extraction
        encoder = self.get_encoder()
        model = encoder(self.moco_dim)
        if self.mlp:
            model.fc = nn.Sequential(
                nn.Linear(
                    model.fc.weight.shape[1],
                    model.fc.weight.shape[1]),
                nn.ReLU(),
                model.fc)
        model = self.model_surgery(model)
        if self.dict_trainer:
            from qd.qd_pytorch import InputAsDict
            model = InputAsDict(model)
        return model

    def get_train_transform(self, start_iter=0):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        from qd.qd_pytorch import BGR2RGB
        if self.aug_plus == 'Plus':
            raise ValueError('deprecating to use plus_dict')
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
            first = transforms.Compose(augmentation)
            second = first
            return TwoCropsTransform(first, second)
        elif self.aug_plus == 'plus_fix':
            # fix the data augmentation for key momentum enocoder. The accuracy
            # is similar with plus_dict.
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
            # by default data augmgentation for v2
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
        elif self.aug_plus == 'one_crop_multi_scale':
            aug1 = [
                BGR2RGB(),
                transforms.ToPILImage(),
            ]
            aug1 = transforms.Compose(aug1)
            aug1 = ImageToImageDictTransform(aug1)
            from qd.data_layer.transform import MultiScaleRandomResizedCrop
            aug_crop = MultiScaleRandomResizedCrop(self.aug_plus_iou, 224, scale=(0.2, 1.))
            aug2 = [
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                normalize
            ]
            aug2 = transforms.Compose(aug2)
            aug2 = ImageToImageDictTransform(aug2)
            from qd.data_layer.transform import TwoCropsTransform112
            return TwoCropsTransform112(aug1, aug_crop, aug2)
        elif self.aug_plus == 'aug_rotate':
            all_aug = []
            aug1 = [
                BGR2RGB(),
                transforms.ToPILImage(),
            ]
            aug1 = transforms.Compose(aug1)
            aug1 = ImageToImageDictTransform(aug1)
            all_aug.append(aug1)

            from qd.data_layer.transform import (
                RandomResizedCropDict,
                FourRotateImage,
                RandomGrayscaleDict,
            )
            crop = RandomResizedCropDict(224, scale=(0.2, 1.))
            all_aug.append(crop)

            aug21 = [
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
            ]
            aug21 = transforms.Compose(aug21)
            aug21 =  ImageToImageDictTransform(aug21)
            all_aug.append(aug21)

            all_aug.append(RandomGrayscaleDict(p=0.2))

            aug22 = [
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
            ]
            aug22 = transforms.Compose(aug22)
            aug22 =  ImageToImageDictTransform(aug22)
            all_aug.append(aug22)

            rotate = FourRotateImage(0.5)
            all_aug.append(rotate)

            aug3 = [
                transforms.ToTensor(),
                normalize
            ]
            aug3 = transforms.Compose(aug3)
            aug3 = ImageToImageDictTransform(aug3)
            all_aug.append(aug3)

            final = transforms.Compose(all_aug)
            return TwoCropsTransform(final, final)
        elif self.aug_plus == 'multi_crop':
            # with 200 epochs, teh accuracy is 70.1. with 800 epochs, it is
            # 69.39. not well studied for this choice
            from qd.data_layer.transform import MultiCropsTransform
            def get_by_size(size, scale_min=0.2, scale_max=1.):
                augmentation = [
                    BGR2RGB(),
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(size, scale=(
                        scale_min,
                        scale_max)),
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
                    'transform': get_by_size(
                        self.train_crop_size,
                        scale_min=self.scale_min,
                        scale_max=self.scale_max),
                },
                {
                    'repeat': self.num_small_crop,
                    'transform': get_by_size(
                        self.small_crop_size,
                        scale_min=self.small_scale_min,
                        scale_max=self.small_scale_max),
                },
            ]
            aug = MultiCropsTransform(all_info)
            return aug
        elif self.aug_plus == 'multi_crop_scale':
            # the multi-crop parameters here are the same with the paper used
            # in SwAV.
            from qd.data_layer.transform import MultiCropsTransform
            def get_by_size(size, scale_min=0.2, scale_max=1.):
                augmentation = [
                    BGR2RGB(),
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(size, scale=(
                        scale_min,
                        scale_max)),
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
                    'transform': get_by_size(
                        self.train_crop_size,
                        scale_min=0.14,
                        scale_max=1.),
                },
                {
                    'repeat': self.num_small_crop,
                    'transform': get_by_size(
                        self.small_crop_size,
                        scale_min=0.05,
                        scale_max=0.14),
                },
            ]
            aug = MultiCropsTransform(all_info)
            return aug
        elif self.aug_plus == 'PlusIoU':
            # the method does not help
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
        else:
            # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
            assert self.aug_plus is None
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
            first = transforms.Compose(augmentation)
            second = first

            return TwoCropsTransform(first, second)

    def ensure_evaluate(self):
        pass

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
