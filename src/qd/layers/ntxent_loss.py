import torch
import numpy as np
import logging
import math
from qd.layers.tensor_queue import TensorQueue
from qd.layers.loss import ExclusiveCrossEntropyLoss


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature, correct_loss=False):
        # correct_loss should always be True. Here we keep it as False by
        # default to make it back-compatible. After some time we verified the
        # correctness, we should always set it true
        super(NTXentLoss, self).__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()

        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        from qd.qd_common import get_mpi_rank, get_mpi_size
        self.rank = get_mpi_rank()
        self.world_size = get_mpi_size()

        self.correct_loss = correct_loss

        ## another way is to use register_buffer, which requires the dimension
        #self.cache = {'neg_sampler': None}

    def _get_neg_sampler(self, bs):
        diag = np.eye(2 * bs)
        l1 = np.eye(2 * bs, k=-bs)
        l2 = np.eye(2 * bs, k=bs)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        mask.requires_grad = False
        return mask

    def get_other_rank(self, x):
        if self.world_size == 1:
            return x
        with torch.no_grad():
            all_x = [torch.zeros_like(x) for _ in range(self.world_size)]
            # note, all_rep should be treated as constent, which means no grad
            # will be propagated back through all_rep
            torch.distributed.all_gather(all_x, x)
        all_x[self.rank] = x
        return torch.cat(all_x, dim=0)

    def forward(self, zis, zjs):
        device = zis.device
        zis = self.get_other_rank(zis)
        zjs = self.get_other_rank(zjs)
        rep = torch.cat([zis, zjs], dim=0)

        bs = zis.shape[0]

        similarity_matrix = torch.matmul(rep, rep.T)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, bs)
        r_pos = torch.diag(similarity_matrix, -bs)
        positives = torch.cat([l_pos, r_pos]).view(2 * bs, 1)

        neg_sampler = self._get_neg_sampler(bs).to(device)

        negatives = similarity_matrix[neg_sampler].view(2 * bs, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * bs).to(device).long()
        loss = self.criterion(logits, labels)
        # it is good to divide it by 2, but the official implementation does
        # not have that. Here, we align it to the official implementation as
        # possible.
        if self.correct_loss:
            return loss * self.world_size / (2*bs)
        else:
            return loss / bs

class DenominatorCrossEntropyLoss(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, logits, target):
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss -= (1. - self.factor) * logits.exp().sum(dim=1).log().mean()
        return loss

class SimpleQueueLoss(torch.nn.Module):
    def __init__(self, temperature, K, dim, alpha=1., alpha_policy=None,
                 max_iter=None, alpha_max=1.,
                 criterion_type=None,
                 denominator_ce_factor=None,
                 ):
        super().__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()
        self.temperature = temperature
        if criterion_type is None:
            self.criterion = torch.nn.CrossEntropyLoss()
        elif criterion_type == 'DeCE':
            self.criterion = DenominatorCrossEntropyLoss(denominator_ce_factor)
        elif criterion_type == 'ExCE':
            self.criterion = ExclusiveCrossEntropyLoss(denominator_ce_factor)
        else:
            raise NotImplementedError(criterion_type)

        from qd.qd_common import get_mpi_rank, get_mpi_size
        self.rank = get_mpi_rank()
        self.world_size = get_mpi_size()

        self.negative_queue = TensorQueue(K, dim)
        self.alpha = alpha
        self.ln_alpha = torch.log(torch.tensor(alpha))
        self.alpha_max = alpha_max
        self.ln_alpha_max = torch.log(torch.tensor(alpha_max))
        self.register_buffer('iter', torch.zeros(1))
        self.max_iter = max_iter
        self.alpha_policy = alpha_policy

    def get_ln_alpha(self):
        curr_iter = int(self.iter)
        if self.alpha_policy == 'Linear':
            curr_alpha = (self.alpha_max - self.alpha) * curr_iter / self.max_iter + \
                self.alpha
            ln_alpha = torch.log(torch.tensor(curr_alpha))
        elif self.alpha_policy == 'LogLinear':
            ln_alpha = (self.ln_alpha_max - self.ln_alpha) * curr_iter / self.max_iter + \
                self.ln_alpha
        elif self.alpha_policy == 'Sin':
            curr_alpha = (self.alpha_max - self.alpha) * (
                math.sin(1. * curr_iter / self.max_iter * math.pi - math.pi / 2) +
                1) / 2 + self.alpha
            ln_alpha = torch.log(torch.tensor(curr_alpha))
        elif self.alpha_policy is None:
            return self.ln_alpha
        else:
            raise NotImplementedError
        return ln_alpha

    def forward(self, zis, zjs):
        debug = (self.iter%10) == 0
        debug_info = []
        self.iter += 1

        device = zis.device
        curr_bs = len(zis)
        effective_batch_size = curr_bs * self.world_size

        # every time, we insert 2 times of ebs into the queue
        order = self.negative_queue.get_order(2 * effective_batch_size).float()
        ln_alpha = self.get_ln_alpha()
        order *= ln_alpha
        order = order.view(-1, 1).expand([len(order), 2 * effective_batch_size])
        order = order.reshape(-1)
        order = order.to(device)

        pos = (zis * zjs).sum(dim=1)[:, None]
        queue = self.negative_queue.queue.clone().detach()
        negi = torch.matmul(zis, queue.T)
        negj = torch.matmul(zjs, queue.T)

        if debug:
            debug_info.append('ln_alpha = {:.4f}'.format(ln_alpha))
            debug_info.append('pos sim avg/max/min = {:.3f}/{:.3f}/{:.3f}'.format(
                pos.mean(), pos.max(), pos.min()))
            debug_info.append('negi sim avg/max/min = {:.3f}/{:.3f}/{:.3f}'.format(
                negi.mean(), negi.max(), negi.min()))
            debug_info.append('negj sim avg/max/min = {:.3f}/{:.3f}/{:.3f}'.format(
                negj.mean(), negj.max(), negj.min()))

        negi /= self.temperature
        negj /= self.temperature
        pos /= self.temperature

        negi += order
        negj += order

        sim = torch.cat([pos, negi, negj], dim=1)
        label = torch.zeros(len(sim), dtype=torch.long).to(device)

        loss = self.criterion(sim, label)
        if debug:
            from qd.torch_common import accuracy
            acc1, acc5 = accuracy(sim, label, topk=(1, 5))
            debug_info.append('acc1 = {:.1f}'.format(float(acc1)))
            debug_info.append('acc5 = {:.1f}'.format(float(acc5)))

        from qd.torch_common import concat_all_gather
        self.negative_queue.en_de_queue(concat_all_gather(zis))
        self.negative_queue.en_de_queue(concat_all_gather(zjs))

        if debug:
            logging.info('; '.join(debug_info))

        return loss

class SwAVQueueLoss(torch.nn.Module):
    def __init__(self, temperature, cluster_size, queue_size, dim,
                 involve_queue_after=0
                 ):
        super().__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()
        self.temperature = temperature
        from qd.layers.loss import KLCrossEntropyLoss
        self.criterion = KLCrossEntropyLoss()

        from qd.qd_common import get_mpi_rank, get_mpi_size
        self.mpi_rank = get_mpi_rank()
        self.world_size = get_mpi_size()

        if queue_size > 0:
            self.negative_queue = TensorQueue(queue_size, dim)
        self.queue_size = queue_size

        cluster_center = torch.randn(cluster_size, dim)
        cluster_center = torch.nn.functional.normalize(cluster_center, dim=1)
        self.cluster_center = torch.nn.Parameter(cluster_center)
        self.register_buffer('iter', torch.zeros(1))
        self.involve_queue_after = involve_queue_after

    def forward(self, zis, zjs):
        from qd.torch_common import concat_all_gather
        bs = len(zis)

        # calculate the code
        with torch.no_grad():
            all_zis = concat_all_gather(zis)
            all_zjs = concat_all_gather(zjs)
            if self.queue_size > 0 and int(self.iter) > self.involve_queue_after:
                queue = self.negative_queue.queue.clone()
                all_zis = torch.cat([all_zis, queue], dim=0)
                all_zjs = torch.cat([all_zjs, queue], dim=0)
            all_scores_t = torch.matmul(all_zis, self.cluster_center.T)
            all_scores_s = torch.matmul(all_zjs, self.cluster_center.T)
            from qd.torch_common import sinkhorn
            all_t_code = sinkhorn(all_scores_t, eps=0.05, niters=3)
            all_s_code = sinkhorn(all_scores_s, eps=0.05, niters=3)
            start = self.mpi_rank * bs
            end = start + bs
            q_t = all_t_code[start:end]
            q_s = all_s_code[start:end]
        # re-calculate to make the gradient propogated back
        scores_t = torch.matmul(zis, self.cluster_center.T)
        scores_s = torch.matmul(zjs, self.cluster_center.T)
        loss1 = 0.5 * self.criterion(scores_t / self.temperature, q_s)
        loss2 = 0.5 * self.criterion(scores_s / self.temperature, q_t)
        loss = {'loss1': loss1, 'loss2': loss2}

        if self.queue_size > 0:
            self.negative_queue.en_de_queue(zis)
        self.iter += 1
        return loss

class NTXentQueueLoss(torch.nn.Module):
    def __init__(self, temperature, K, dim, alpha=1., alpha_policy=None,
                 max_iter=None, alpha_max=1.,
                 criterion_type=None,
                 denominator_ce_factor=None,
                 ):
        super().__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()
        self.temperature = temperature
        if criterion_type is None:
            self.criterion = torch.nn.CrossEntropyLoss()
        elif criterion_type == 'DeCE':
            self.criterion = DenominatorCrossEntropyLoss(denominator_ce_factor)
        elif criterion_type == 'ExCE':
            self.criterion = ExclusiveCrossEntropyLoss(denominator_ce_factor)
        else:
            raise NotImplementedError(criterion_type)

        from qd.qd_common import get_mpi_rank, get_mpi_size
        self.rank = get_mpi_rank()
        self.world_size = get_mpi_size()

        self.negative_queue = TensorQueue(K, dim)
        self.alpha = alpha
        self.ln_alpha = torch.log(torch.tensor(alpha))
        self.alpha_max = alpha_max
        self.ln_alpha_max = torch.log(torch.tensor(alpha_max))
        self.register_buffer('iter', torch.zeros(1))
        self.max_iter = max_iter
        self.alpha_policy = alpha_policy

    def get_ln_alpha(self):
        curr_iter = int(self.iter)
        if self.alpha_policy == 'Linear':
            curr_alpha = (self.alpha_max - self.alpha) * curr_iter / self.max_iter + \
                self.alpha
            ln_alpha = torch.log(torch.tensor(curr_alpha))
        elif self.alpha_policy == 'LogLinear':
            ln_alpha = (self.ln_alpha_max - self.ln_alpha) * curr_iter / self.max_iter + \
                self.ln_alpha
        elif self.alpha_policy == 'Sin':
            curr_alpha = (self.alpha_max - self.alpha) * (
                math.sin(1. * curr_iter / self.max_iter * math.pi - math.pi / 2) +
                1) / 2 + self.alpha
            ln_alpha = torch.log(torch.tensor(curr_alpha))
        elif self.alpha_policy is None:
            return self.ln_alpha
        else:
            raise NotImplementedError
        return ln_alpha

    def forward(self, zis, zjs):
        debug = (self.iter%10) == 0
        debug_info = []
        self.iter += 1

        device = zis.device
        from qd.torch_common import all_gather_grad_curr
        zis = all_gather_grad_curr(zis)
        zjs = all_gather_grad_curr(zjs)
        rep = torch.cat([zis, zjs], dim=0)
        rep_queue = torch.cat([rep, self.negative_queue.queue], dim=0)

        bs = zis.shape[0]

        sim = torch.matmul(rep, rep_queue.T)
        if debug:
            x1 = sim[list(range(bs)), list(range(bs, 2*bs))].mean()
            debug_info.append('sim = {:.3f}'.format(x1))
            debug_info.append('dissim-mean = {:.3f}'.format(sim[:, 2*bs:].mean()))
            debug_info.append('dissim-max = {:.3f}'.format(sim[:, 2*bs:].max()))
            debug_info.append('dissim-min= {:.3f}'.format(sim[:, 2*bs:].min()))

        order = self.negative_queue.get_order(rep.shape[0])
        order = torch.cat([torch.tensor([0]), order]).float()
        ln_alpha = self.get_ln_alpha()
        order *= ln_alpha
        order = order.view(-1, 1).expand([len(order), rep.shape[0]])
        order = order.reshape(-1)
        order = order.to(device)

        if debug:
            x1 = sim[list(range(bs)), list(range(bs, 2*bs))].mean()
            debug_info.append('ln_alpha = {}'.format(ln_alpha))

        sim += order

        sim /= self.temperature
        sim.fill_diagonal_(-float('inf'))

        label1 = torch.arange(bs, bs * 2).to(device).long()
        label2 = torch.arange(bs).to(device).long()
        label = torch.cat([label1, label2])

        if debug:
            from qd.torch_common import accuracy
            acc1, acc5 = accuracy(sim, label, topk=(1, 5))
            debug_info.append('acc1 = {:.1f}'.format(float(acc1)))
            debug_info.append('acc5 = {:.1f}'.format(float(acc5)))

        loss = self.criterion(sim, label)
        self.negative_queue.en_de_queue(rep)
        if debug:
            logging.info('; '.join(debug_info))
        return loss * self.world_size

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class NoisyDiscriminator(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()

        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, 2),
        )
        self.iter = 0

    def forward(self, zis, zjs):
        debug = (self.iter % 10) == 0
        sim_loss = -(zis * zjs).sum(dim=1).mean()

        bs, dim = zis.shape
        num_pos = 2 * bs
        num_neg = 2 * bs
        pos = torch.randn(num_pos, dim).to(zis.device)
        pos = torch.nn.functional.normalize(pos)
        zis = grad_reverse(zis)
        zjs = grad_reverse(zjs)
        x = torch.cat([zis, zjs, pos])
        y = self.discriminator(x)
        gt_y = torch.cat([torch.zeros(num_neg, dtype=torch.long), torch.ones(num_pos, dtype=torch.long)]).to(zis.device)
        dissim_loss = torch.nn.functional.cross_entropy(y, gt_y)

        if debug:
            from qd.torch_common import accuracy
            top1 = accuracy(y, gt_y)
            pos_top1 = accuracy(y[num_neg:], gt_y[num_neg:])
            neg_top1 = accuracy(y[:num_neg], gt_y[:num_neg])
            logging.info('top1-all = {}; pos-top1 = {}; neg-top1 = {}'.format(
                top1, pos_top1, neg_top1))


        self.iter += 1
        return {'sim_loss': sim_loss,
                'dissim_loss': dissim_loss}

