import logging
import math
import torch
from qd.qd_common import calculate_iou_xywh
import numpy as np
import torch.nn as nn
import logging


class MultiIDHashNMSAny(object):
    def __init__(self, num, alpha, gamma):
        self.num = num
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, rects, conf):
        all_hash_rect = []
        num = min(len(rects), self.num)
        for i in range(num):
            bx, by, curr_w0, curr_h0 = rects[i]
            hr= SingleHashNMS(curr_w0, curr_h0,
                    self.alpha,
                    self.gamma,
                    bx, by, b_is_relative=False)
            all_hash_rect.append(hr)

        for hr in all_hash_rect:
            hr(rects, conf)

class MultiHashNMSAll(object):
    def __init__(self, num, w0, h0, alpha, gamma):
        self.hash_rect = MultiHashRect(num, w0, h0, alpha, gamma)
        self.debug = True

    def __call__(self, rects, conf):
        '''
        rects: N x 4: xywh
        conf: N x C
        we will set the conf as 0 if it is suppressed
        '''
        codes = [tuple(self.hash_rect(xywh)) for xywh in rects]
        from collections import defaultdict
        code_to_idxs = defaultdict(list)
        for i, code in enumerate(codes):
            code_to_idxs[code].append(i)
        for code, idxs in code_to_idxs.items():
            if len(idxs) == 1:
                continue
            curr_conf = conf[idxs]
            max_conf = curr_conf.max(axis=0)
            curr_conf[curr_conf < max_conf] = 0
            conf[idxs] = curr_conf

class MultiHashNMSAnyKPt(nn.ModuleList):
    def __init__(self, num, w0, h0, alpha, gamma,
            rerank=False, rerank_iou=0.5):
        all_hash_rect = []
        is_random = False
        for i in range(num):
            if is_random:
                import random
                curr_w0 = math.exp(random.random() * (-math.log(alpha)) + math.log(w0))
                curr_h0 = math.exp(random.random() * (-math.log(alpha)) + math.log(h0))
                bx = random.random()
                by = random.random()
            else:
                import random
                curr_w0 = math.exp(1. * i / num * (-math.log(alpha)) + math.log(w0))
                curr_h0 = math.exp(1. * i / num * (-math.log(alpha)) + math.log(h0))
                bx = 1. * i / num
                by = 1. * i / num
            #hr = SingleHashNMSKPt(curr_w0, curr_h0, alpha, gamma,
                    #bx, by, rerank=rerank, rerank_iou=rerank_iou)
            hr = SingleHashNMSKPtC(curr_w0, curr_h0, alpha, gamma,
                    bx, by, rerank=rerank, rerank_iou=rerank_iou)
            all_hash_rect.append(hr)
        super(MultiHashNMSAnyKPt, self).__init__(all_hash_rect)

    def forward(self, rects, conf):
        for i, hr in enumerate(self):
            if i == 0:
                curr_keep = hr(rects, conf)
                keep = curr_keep
            else:
                curr_keep = hr(rects[keep], conf[keep])
                keep = keep[curr_keep]
        return keep

class MultiHashNMSAnyK(object):
    def __init__(self, num, w0, h0, alpha, gamma,
            rerank=False, rerank_iou=0.5):
        self.all_hash_rect = []
        for i in range(num):
            import random
            curr_w0 = math.exp(random.random() * (-math.log(alpha)) + math.log(w0))
            curr_h0 = math.exp(random.random() * (-math.log(alpha)) + math.log(h0))
            bx = random.random()
            by = random.random()
            hr = SingleHashNMSK(curr_w0, curr_h0, alpha, gamma,
                    bx, by, rerank=rerank, rerank_iou=rerank_iou)
            self.all_hash_rect.append(hr)

    def __call__(self, rects, conf):
        for i, hr in enumerate(self.all_hash_rect):
            if i == 0:
                curr_keep = hr(rects, conf)
                keep = curr_keep
            else:
                curr_keep = hr(rects[keep], conf[keep])
                keep = [keep[k] for k in curr_keep]
        return keep

class MultiHashNMSAny(object):
    def __init__(self, num, w0, h0, alpha, gamma,
            rerank=False):
        self.all_hash_rect = []
        for i in range(num):
            import random
            curr_w0 = math.exp(random.random() * (-math.log(alpha)) + math.log(w0))
            curr_h0 = math.exp(random.random() * (-math.log(alpha)) + math.log(h0))
            bx = random.random()
            by = random.random()
            hr= SingleHashNMS(curr_w0, curr_h0, alpha, gamma,
                    bx, by, rerank=rerank)
            self.all_hash_rect.append(hr)

    def __call__(self, rects, conf):
        for hr in self.all_hash_rect:
            hr(rects, conf)

class HashRectsPt(nn.Module):
    def __init__(self, w0, h0, alpha, gamma, bx, by, b_is_relative=True):
        super(HashRectsPt, self).__init__()
        self.register_buffer('w0', torch.tensor(w0))
        self.register_buffer('h0', torch.tensor(h0))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('gamma', torch.tensor(gamma))
        self.register_buffer('bx', torch.tensor(bx))
        self.register_buffer('by', torch.tensor(by))
        self.b_is_relative = b_is_relative

        self.register_buffer('log_alpha', torch.log(self.alpha))
        self.register_buffer('log_w0', torch.log(self.w0))
        self.register_buffer('log_h0', torch.log(self.h0))
        self.register_buffer('gamma_ratio', (1. - self.gamma) / (1. + self.gamma))
        self.register_buffer('w0_gamma', self.w0 * self.gamma_ratio)
        self.register_buffer('h0_gamma', self.h0 * self.gamma_ratio)

    def __call__(self, xywh):
        x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        i = ((self.log_w0 - torch.log(w)) / self.log_alpha).round()
        j = ((self.log_h0 - torch.log(h)) / self.log_alpha).round()
        di = self.w0_gamma / self.alpha ** i
        dj = self.h0_gamma / self.alpha ** j
        if self.b_is_relative:
            qx = (x / di - self.bx).round()
            qy = (y / dj - self.by).round()
        else:
            qx = (x / di - self.bx / di).round()
            qy = (y / dj - self.by / dj).round()
        return torch.stack((qx, qy, i, j), dim=1).long()

class HashRect(object):
    def __init__(self, w0, h0, alpha, gamma, bx, by, b_is_relative=True):
        #from qd.qd_common import print_frame_info
        #print_frame_info()
        self.w0 = w0
        self.h0 = h0
        self.alpha = alpha
        self.gamma = gamma
        self.bx = bx
        self.by = by
        self.b_is_relative = b_is_relative

        self.gamma_ratio = (1 - self.gamma) / (1 + self.gamma)
        self.w_gamma = self.w0 * self.gamma_ratio
        self.h_gamma = self.h0 * self.gamma_ratio

    def __call__(self, xywh):
        x, y, w, h = xywh
        i = round((math.log(self.w0) - math.log(w)) / math.log(self.alpha))
        j = round((math.log(self.h0) - math.log(h)) / math.log(self.alpha))
        di = self.w_gamma / self.alpha ** i
        dj = self.h_gamma / self.alpha ** j
        if self.b_is_relative:
            qx = round(x / di - self.bx)
            qy = round(y / dj - self.by)
        else:
            qx = round(x / di - self.bx / di)
            qy = round(y / dj - self.by / dj)
        return int(qx), int(qy), int(i), int(j)

class MultiHashRect(object):
    def __init__(self, num, w0, h0, alpha, gamma):
        self.all_hash_rect = []
        for i in range(num):
            import random
            curr_w0 = math.exp(random.random() * (-math.log(alpha)) + math.log(w0))
            curr_h0 = math.exp(random.random() * (-math.log(alpha)) + math.log(h0))
            bx = random.random()
            by = random.random()
            hr= HashRect(curr_w0, curr_h0, alpha, gamma, bx, by)
            self.all_hash_rect.append(hr)

    def __call__(self, rects):
        return tuple(hr(rects) for hr in self.all_hash_rect)

class SingleHashNMSKPtC(nn.Module):
    def __init__(self, w0, h0, alpha, gamma, bx=0.5, by=0.5,
            b_is_relative=True, rerank=False, rerank_iou=0.5):
        super(SingleHashNMSKPtC, self).__init__()
        self.w0 = float(w0)
        self.h0 = float(h0)
        self.alpha = alpha
        self.gamma = gamma
        self.bx = bx
        self.by = by
        self.b_is_relative = b_is_relative
        self.rerank = rerank
        self.rerank_iou = rerank_iou

    def __call__(self, rects, conf):
        from maskrcnn_benchmark.layers import hnms
        result = hnms(rects, conf,
                self.w0, self.h0,
                self.alpha, self.gamma,
                self.bx, self.by, self.b_is_relative,
                self.rerank, self.rerank_iou)
        return result
            #result = result.cpu()
            #start = time()
            #result2 = hnms(rects.cpu(), conf.cpu(),
                    #self.w0, self.h0,
                    #self.alpha, self.gamma,
                    #self.bx, self.by, self.b_is_relative,
                    #self.rerank, self.rerank_iou)
            #self.cuda_single_hnms = SingleHashNMSKPt2(w0, h0,
                    #alpha, gamma, bx=bx, by=by, b_is_relative=b_is_relative,
                    #rerank=rerank, rerank_iou=rerank_iou)

            #result2 = self.cuda_single_hnms(rects, conf)
            #logging.info(time() - start)
            #result, _ = torch.sort(result)
            #result2, _ = torch.sort(result2)
            #assert (result - result2).abs().sum() == 0
            #import ipdb;ipdb.set_trace(context=15)
            #return result
            #rects = rects.cpu()
            #conf = conf.cpu()
            #result = hnms(rects, conf,
                    #self.w0, self.h0,
                    #self.alpha, self.gamma,
                    #self.bx, self.by, self.b_is_relative,
                    #self.rerank, self.rerank_iou)

            #return result.cuda()


class SingleHashNMSKPt2(nn.Module):
    def __init__(self, w0, h0, alpha, gamma, bx=0.5, by=0.5,
            b_is_relative=True, rerank=False, rerank_iou=0.5):
        super(SingleHashNMSKPt2, self).__init__()
        self.hash_rect = HashRectsPt(w0, h0, alpha, gamma,
                bx, by, b_is_relative)
        self.rerank = rerank
        self.rerank_iou = rerank_iou

        #self.lower_bound_check = True
        #self.lower_bound = calc_lower_iou_bound(alpha, gamma)
        #self.lower_bound_checked = 0
        factor = torch.zeros((4, 1)).long()
        factor[0] = 1
        factor[1] = 10000
        factor[2] = 100000000
        factor[3] = 1000000000000
        self.register_buffer('factor', factor)

    def __call__(self, rects, conf):
        codes = self.hash_rect(rects)
        #if codes.device.type == 'cuda':
        single_codes = codes[:, 0] + codes[:, 1] * 10000 + \
                codes[:, 2] * 100000000 + \
                codes[:, 3] * 1000000000000
        single_codes = single_codes.view((-1, 1))
        #else:
            #single_codes = torch.mm(codes, self.factor)
        unique_values = torch.unique(single_codes, sorted=False)
        assignment = single_codes == unique_values.view((1, -1))
        conf = conf.view((-1, 1)).expand(-1, len(unique_values))
        _, argidx = (conf * assignment.float()).max(dim=0)
        return argidx

class SingleHashNMSKPt(object):
    def __init__(self, w0, h0, alpha, gamma, bx=0.5, by=0.5,
            b_is_relative=True, rerank=False, rerank_iou=0.5):
        self.hash_rect = HashRectsPt(w0, h0, alpha, gamma,
                bx, by, b_is_relative)
        self.rerank = rerank
        self.rerank_iou = rerank_iou

        #self.lower_bound_check = True
        #self.lower_bound = calc_lower_iou_bound(alpha, gamma)
        #self.lower_bound_checked = 0

    def __call__(self, rects, conf):
        codes = self.hash_rect(rects)
        from collections import defaultdict
        code_to_idxs = defaultdict(list)
        for i, code in enumerate(codes):
            code = (int(code[0]), int(code[1]),
                    int(code[2]), int(code[3]))
            code_to_idxs[code].append(i)

        keep = np.full(len(rects), True, dtype=np.bool)
        for code, idxs in code_to_idxs.items():
            if len(idxs) == 1:
                continue
            idxs = np.array(idxs)
            curr_conf = conf[idxs]
            if not self.rerank:
                max_conf = curr_conf.max()
                keep[idxs[curr_conf < max_conf]] = False
                #if self.lower_bound_check:
                    #max_idx = idxs[curr_conf == max_conf][0]
                    #for i in idxs:
                        #curr_iou = calculate_iou_xywh(rects[i], rects[max_idx])
                        #assert curr_iou > self.lower_bound
                        #self.lower_bound_checked += 1
                        #if (self.lower_bound_checked % 1000) == 0:
                            #logging.info(self.lower_bound_checked)
            else:
                sorted_idx = sorted(idxs, key=lambda i: -conf[i])
                for i, i_idx in enumerate(sorted_idx):
                    if not keep[i_idx]:
                        continue
                    i_rect = rects[i_idx]
                    for j in range(i + 1, len(sorted_idx)):
                        j_idx = sorted_idx[j]
                        if not keep[j_idx]:
                            continue
                        j_rect = rects[j_idx]
                        iou = calculate_iou_xywh(i_rect, j_rect)
                        if iou > self.rerank_iou:
                            keep[j_idx] = False
        return [i for i, k in enumerate(keep) if k]

class SingleHashNMSK(object):
    def __init__(self, w0, h0, alpha, gamma, bx=0.5, by=0.5,
            b_is_relative=True, rerank=False, rerank_iou=0.5):
        self.hash_rect = HashRect(w0, h0, alpha, gamma,
                bx, by, b_is_relative)
        self.rerank = rerank
        self.rerank_iou = rerank_iou

    def __call__(self, rects, conf):
        codes = [tuple(self.hash_rect(xywh)) for xywh in rects]
        from collections import defaultdict
        code_to_idxs = defaultdict(list)
        for i, code in enumerate(codes):
            code_to_idxs[code].append(i)

        keep = np.full(len(rects), True, dtype=np.bool)
        for _, idxs in code_to_idxs.items():
            if len(idxs) == 1:
                continue
            idxs = np.array(idxs)
            curr_conf = conf[idxs]
            if not self.rerank:
                max_conf = curr_conf.max()
                assert (curr_conf == max_conf).sum() == 1
                keep[idxs[curr_conf < max_conf]] = False
            else:
                sorted_idx = sorted(idxs, key=lambda i: -conf[i])
                for i, i_idx in enumerate(sorted_idx):
                    if not keep[i_idx]:
                        continue
                    i_rect = rects[i_idx]
                    for j in range(i + 1, len(sorted_idx)):
                        j_idx = sorted_idx[j]
                        if not keep[j_idx]:
                            continue
                        j_rect = rects[j_idx]
                        iou = calculate_iou_xywh(i_rect, j_rect)
                        if iou > self.rerank_iou:
                            keep[j_idx] = False
        return [i for i, k in enumerate(keep) if k]

class SingleHashNMS(object):
    def __init__(self, w0, h0, alpha, gamma, bx=0.5, by=0.5,
            b_is_relative=True, rerank=False):
        self.hash_rect = HashRect(w0, h0, alpha, gamma,
                bx, by, b_is_relative)
        self.rerank = rerank

    def __call__(self, rects, conf):
        '''
        rects: N x 4: xywh
        conf: N x C
        we will set the conf as 0 if it is suppressed
        '''
        codes = [tuple(self.hash_rect(xywh)) for xywh in rects]
        from collections import defaultdict
        code_to_idxs = defaultdict(list)
        for i, code in enumerate(codes):
            code_to_idxs[code].append(i)
        for code, idxs in code_to_idxs.items():
            if len(idxs) == 1:
                continue
            if not self.rerank:
                curr_conf = conf[idxs]
                max_conf = curr_conf.max()
                curr_conf[curr_conf < max_conf] = 0
                conf[idxs] = curr_conf
            else:
                sorted_idx = sorted(idxs, key=lambda i: -conf[i])
                for i, i_idx in enumerate(sorted_idx):
                    if conf[i_idx] == 0:
                        continue
                    i_rect = rects[i_idx]
                    for j in range(i + 1, len(sorted_idx)):
                        j_idx = sorted_idx[j]
                        if conf[j_idx] == 0:
                            continue
                        j_rect = rects[j_idx]
                        iou = calculate_iou_xywh(i_rect, j_rect)
                        if iou > 0.5:
                            conf[j_idx] = 0

def calc_lower_iou_bound(alpha, gamma):
    if gamma <= (1 - alpha ** 0.5) / (1 + alpha ** 0.5):
        return 0
    W0 = 1.
    H0 = 1.
    i = 0
    j = 0

    min_w = W0 / alpha ** (i - 0.5)
    max_w = W0 / alpha ** (i + 0.5)
    min_h = H0 / alpha ** (j - 0.5)
    max_h = H0 / alpha ** (j + 0.5)
    assert min_w < max_w and min_h < max_h
    x1 = 0
    y1 = 0
    Wi = W0 / alpha ** i
    Hj = H0 / alpha ** j
    di = (1 - gamma) / (1 + gamma) * Wi
    dj = (1 - gamma) / (1 + gamma) * Hj
    min_x2 = x1 - di
    max_x2 = x1 + di
    min_y2 = y1 - dj
    max_y2 = y1 + dj

    def calc_iou_center_size(x1, y1, w1, h1, x2, y2, w2, h2):
        left = max(x1 - w1/2, x2 - w2/2)
        right = min(x1 + w1/2, x2 + w2/2)
        top = max(y1 - h1 / 2, y2 - h2 / 2)
        bottom = min(y1 + h1 / 2, y2 + h2 / 2)
        assert right >= left and bottom >= top
        inter = (right - left) * (bottom - top)
        return inter / (w1 * h1 + w2 * h2 - inter)

    all_iou = [calc_iou_center_size(x1, y1, w1, h1, x2, y2, w2, h2)
            for w1 in [min_w, max_w]
            for h1 in [min_h, max_h]
            for x2 in [min_x2, max_x2]
            for y2 in [min_y2, max_y2]
            for w2 in [min_w, max_w]
            for h2 in [min_h, max_h]]
    iou = min(all_iou)
    return iou


