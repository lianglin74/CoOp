import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    # no need to use cosine since we have already normalized the features
    def __init__(self, temperature):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        from qd.qd_common import get_mpi_rank, get_mpi_size
        self.rank = get_mpi_rank()
        self.world_size = get_mpi_size()

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

        return loss / (2 * bs)

