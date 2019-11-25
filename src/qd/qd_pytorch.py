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
from collections import OrderedDict
from qd.process_tsv import load_key_rects
from qd.process_tsv import hash_sha1
from qd.tsv_io import tsv_writer, tsv_reader
from qd.tsv_io import TSVFile
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
import torch.nn.functional as F


def load_scheduler_state(scheduler, state):
    for k, v in state.items():
        if k in scheduler.__dict__:
            curr = scheduler.__dict__[k]
            from qd.qd_common import float_tolorance_equal
            # if the parameter is about the old scheduling, we ignore it. We
            # prefer the current scheduling parameters, except the last_epoch
            # or some other states which relies on the iteration.
            if k in ['milestones', 'warmup_factor', 'warmup_iters', 'warmup_method',
                'base_lrs', 'gamma'] or float_tolorance_equal(curr, v):
                continue
            elif k in ['last_epoch', '_step_count']:
                logging.info('updating {} from {} to {}'.format(k,
                    curr, v))
                scheduler.__dict__[k] = v
            else:
                raise NotImplementedError('unknown {}'.format(k))
        else:
            scheduler.__dict__[k] = v

def synchronize():
    """
    copied from maskrcnn_benchmark.utils.comm
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    from qd.qd_common import is_hvd_initialized
    use_hvd = is_hvd_initialized()
    if not use_hvd:
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        dist.barrier()
    else:
        from qd.qd_common import get_mpi_size
        if get_mpi_size() > 1:
            import horovod.torch as hvd
            hvd.allreduce(torch.tensor(0), name='barrier')

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


class CompositeTSVFile():
    def __init__(self, list_file, seq_file, cache_policy=False):
        self.seq_file = seq_file
        self.list_file = list_file
        self.cache_policy = cache_policy
        self.initialized = False
        self.initialize()

    def __getitem__(self, index):
        idx_source, idx_row = self.seq[index]
        return self.tsvs[idx_source].seek(idx_row)

    def __len__(self):
        return len(self.seq)

    def initialize(self):
        '''
        this function has to be called in init function if cache_policy is
        enabled. Thus, let's always call it in init funciton to make it simple.
        '''
        if self.initialized:
            return
        self.seq = [(int(idx_source), int(idx_row)) for idx_source, idx_row in
                tsv_reader(self.seq_file)]
        self.tsvs = [TSVFile(f, self.cache_policy) for f in load_list_file(self.list_file)]
        self.initialized = True

class TSVSplitProperty(Dataset):
    '''
    one instance of this class mean one tsv file or one composite tsv, it could
    be label tsv, or hw tsv, or image tsv
    '''
    def __init__(self, data, split, t, version=0, cache_policy=None):
        dataset = TSVDataset(data)
        if op.isfile(dataset.get_data(split, t, version)):
            self.tsv = TSVFile(dataset.get_data(split, t, version),
                    cache_policy)
        else:
            splitX = split + 'X'
            list_file = dataset.get_data(splitX, t)
            seq_file = dataset.get_shuffle_file(split)
            self.tsv = CompositeTSVFile(list_file, seq_file, cache_policy)
            assert version in [0, None]

    def __getitem__(self, index):
        row = self.tsv[index]
        return row

    def __len__(self):
        return len(self.tsv)

class TSVSplit(Dataset):
    '''
    prefer to use TSVSplitProperty, which is more general. One example is to
    read the hw property
    '''
    def __init__(self, data, split, version=0, cache_policy=None):
        # image tsv only has version 0
        self.tsv = TSVSplitProperty(data, split, t=None, version=0,
                cache_policy=cache_policy)
        self.label_tsv = TSVSplitProperty(data, split, t='label',
                version=version, cache_policy=cache_policy)

    def __getitem__(self, index):
        _, __, str_image = self.tsv[index]
        key, str_label = self.label_tsv[index]
        return key, str_label, str_image

    def __len__(self):
        return len(self.tsv)

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
            idx = self.label_to_idx[info[0]['class']]
        label = torch.from_numpy(np.array(idx, dtype=np.int))
        return label

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
        label = self.label_to_idx[rect['class']]
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

def get_test_transform(bgr2rgb=False):
    normalize = get_data_normalize()
    all_trans = []
    if bgr2rgb:
        all_trans.append(BGR2RGB())
    all_trans.extend([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])
    return transforms.Compose(all_trans)

def get_train_transform(bgr2rgb=False):
    normalize = get_data_normalize()
    totensor = transforms.ToTensor()
    color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    all_trans = []
    if bgr2rgb:
        all_trans.append(BGR2RGB())
    all_trans.extend([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        color_jitter,
        transforms.RandomHorizontalFlip(),
        totensor,
        normalize,])
    data_augmentation = transforms.Compose(all_trans)
    return data_augmentation

def get_data_normalize():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return normalize

class MultiHotCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiHotCrossEntropyLoss, self).__init__()

    def forward(self, feature, target):
        return multi_hot_cross_entropy(feature, target)

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

class BCEWithLogitsNegLoss(nn.Module):
    def __init__(self):
        super(BCEWithLogitsNegLoss, self).__init__()

    def forward(self, feature, target):
        return bce_with_logits_neg_loss(feature, target)

def bce_with_logits_neg_loss(feature, target):
    target = target.float()
    weight = torch.ones_like(target)
    weight[target == -1] = 0
    weight_sum = torch.sum(weight)
    if weight_sum == 0:
        return 0
    else:
        criterion = nn.BCEWithLogitsLoss(weight, reduction='sum')
        loss = criterion(feature, target)
        return torch.sum(loss) / weight_sum

def multi_hot_cross_entropy(pred, soft_targets):
    assert ((soft_targets != 0) & (soft_targets != 1)).sum() == 0
    logsoftmax = nn.LogSoftmax(dim=1)
    target_sum = torch.sum(soft_targets)
    if target_sum == 0:
        return 0
    else:
        return torch.sum(-soft_targets * logsoftmax(pred)) / target_sum

def load_latest_parameters(folder):
    yaml_file = get_latest_parameter_file(folder)
    logging.info('using {}'.format(yaml_file))
    param = load_from_yaml_file(yaml_file)
    return param

def get_latest_parameter_file(folder):
    yaml_pattern = op.join(folder,
            'parameters_*.yaml')
    yaml_files = glob.glob(yaml_pattern)
    assert len(yaml_files) > 0
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

def torch_save(t, f):
    ensure_directory(op.dirname(f))
    torch.save(t, f)

def torch_load(filename):
    return torch.load(filename, map_location=lambda storage, loc: storage)

def get_aml_mpi_host_names():
    return os.environ['AZ_BATCH_HOST_LIST'].split(',')

def get_master_node_ip():
    if 'AZ_BATCH_HOST_LIST' in os.environ:
        return get_aml_mpi_host_names()[0]
    elif 'AZ_BATCHAI_JOB_MASTER_NODE_IP' in os.environ:
        return os.environ['AZ_BATCHAI_JOB_MASTER_NODE_IP']
    else:
        return get_philly_mpi_hosts()[0]

def get_philly_mpi_hosts():
    return load_list_file(op.expanduser('~/mpi-hosts'))

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

        self._default = {'dist_backend': 'nccl',
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
                'weight_decay': 0.0005,
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
                }

        assert 'batch_size' not in kwargs, 'use effective_batch_size'

        self.data = kwargs.get('data', 'Unknown')
        self.net = kwargs.get('net', 'Unknown')
        self.expid = kwargs.get('expid', 'Unknown')

        self.full_expid = kwargs.get('full_expid',
                '_'.join([self.data, self.net, self.expid]))
        self.output_folder = op.join('output', self.full_expid)
        self.test_data = kwargs.get('test_data', self.data)
        self.test_batch_size = kwargs.get('test_batch_size',
                self.effective_batch_size)
        # if self.max_epoch is None and \
        #         type(self.max_iter) is str and \
        #         self.max_iter.endswith('e'):
        #     # we will not use max_epoch gradually
        #     self.max_epoch = int(self.max_iter[: -1])

        # deprecate max_epoch, use max_iter
        assert self.max_epoch is None and self.max_iter is not None
        self.mpi_rank = get_mpi_rank()
        self.mpi_size= get_mpi_size()
        self.mpi_local_rank = get_mpi_local_rank()
        self.mpi_local_size = get_mpi_local_size()

        # the following two environements are used in init_dist_process if the
        # method is by env. we make sure the world is the same here
        if 'WORLD_SIZE' in os.environ:
            assert int(os.environ['WORLD_SIZE']) == self.mpi_size
        if 'RANK' in os.environ:
            assert int(os.environ['RANK']) == self.mpi_rank

        if self.mpi_size > 1:
            self.distributed = True
        else:
            self.distributed = False
            self.is_master = True
        # adapt the batch size based on the mpi_size
        self.is_master = self.mpi_rank == 0

        assert (self.test_batch_size % self.mpi_size) == 0, self.test_batch_size
        self.test_batch_size = self.test_batch_size // self.mpi_size
        self.train_dataset = TSVDataset(self.data)

        self.initialized = False

    def demo(self, path):
        logging.info('not implemented')

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

    def _ensure_initialized(self):
        if self._initialized:
            return

        if self.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        self._setup_logging()
        # in torch 0.4, torch.randperm only supports cpu. if we set it as
        # cuda.Float by default, it will crash there
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        #torch.set_default_tensor_type('torch.FloatTensor')
        if self.use_hvd:
            # work with world size == 1, also
            import horovod.torch as hvd
            hvd.init()
            torch.cuda.set_device(hvd.local_rank())
            assert hvd.local_rank() == self.mpi_local_rank
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
                torch.cuda.set_device(self.mpi_local_rank)
                logging.info('init param: \n{}'.format(pformat(init_param)))
                if not dist.is_initialized():
                    dist.init_process_group(**init_param)
                # we need to synchronise before exit here so that all workers can
                # finish init_process_group(). If not, worker A might exit the
                # whole program first, but worker B still needs to talk with A. In
                # that case, worker B will never return and will hang there
                synchronize()
        init_random_seed(self.random_seed)
        self._initialized = True

    def _get_dataset(self, data, split, stage, labelmap):
        if not self.bgr2rgb:
            logging.warn('normally bgr2rgb should be true.')
        if stage == 'train':
            transform = get_train_transform(self.bgr2rgb)
        else:
            assert stage == 'test'
            transform = get_test_transform(self.bgr2rgb)

        if self.dataset_type == 'single':
            return TSVSplitImage(data, split,
                    version=0,
                    transform=transform,
                    labelmap=labelmap,
                    cache_policy=self.cache_policy)
        if self.dataset_type == 'crop':
            return TSVSplitCropImage(data, split,
                    version=0,
                    transform=transform,
                    labelmap=labelmap,
                    cache_policy=self.cache_policy)
        elif self.dataset_type == 'multi_hot':
            return TSVSplitImageMultiLabel(data, split,
                    version=0,
                    transform=transform,
                    cache_policy=self.cache_policy,
                    labelmap=labelmap)
        elif self.dataset_type == 'multi_hot_neg':
            return TSVSplitImageMultiLabelNeg(data, split, version=0,
                    transform=transform,
                    cache_policy=self.cache_policy,
                    labelmap=labelmap)
        else:
            raise ValueError('unknown {}'.format(self.dataset_type))

    def _get_criterion(self):
        if self.dataset_type in ['single', 'crop']:
            criterion = nn.CrossEntropyLoss().cuda()
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
        return criterion

    def _get_checkpoint_file(self, epoch=None, iteration=None):
        assert epoch is None, 'not supported'
        if iteration is None:
            iteration = self.max_iter
        iteration = self.parse_iter(iteration)
        return op.join(self.output_folder, 'snapshot',
                "model_iter_{:07d}.pt".format(iteration))

    def _get_model(self, pretrained, num_class):
        model = models.__dict__[self.net](pretrained=pretrained)
        if model.fc.weight.shape[0] != num_class:
            model.fc = nn.Linear(model.fc.weight.shape[1], num_class)
            torch.nn.init.xavier_uniform_(model.fc.weight)

        return model

    def _data_parallel_wrap(self, model):
        if self.distributed:
            model.cuda()
            if self.mpi_local_size > 1:
                if not self.use_hvd:
                    model = torch.nn.parallel.DistributedDataParallel(model,
                            device_ids=[self.mpi_local_rank])
            else:
                if not self.use_hvd:
                    model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.DataParallel(model).cuda()
        return model

    def _get_train_data_loader(self, start_iter=0):
        train_dataset = self._get_dataset(self.data,
                'train',
                stage='train',
                labelmap=self.train_dataset.load_labelmap())

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
        test_dataset = self._get_dataset(test_data, test_split, stage='test',
                labelmap=labelmap)

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
        logging.info('torch version = {}'.format(torch.__version__))

        if self.mpi_rank == 0 and not self.debug_train:
            from qd.qd_common import zip_qd
            zip_qd(op.join(self.output_folder, 'source_code'))

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
        optimizer = torch.optim.SGD(model.parameters(), self.base_lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)
        return optimizer

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                step_size=self.parse_iter(self.step_lr),
                last_epoch=last_epoch)
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
        cc.append('predict')
        cc.append('tsv')
        return '.'.join(cc)

    def append_predict_param(self, cc):
        if self.test_normalize_module != 'softmax':
            cc.append('NormBy{}'.format(self.test_normalize_module))
        if self.predict_extract:
            cc.append('Extract{}'.format(self.predict_extract))

    def monitor_train(self):
        self._ensure_initialized()
        assert self.max_epoch == None, 'use iteration'
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
        return op.isfile(last_model)

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
        cc.append('report')
        return '.'.join(cc)

    def ensure_evaluate(self, predict_file=None):
        if self.mpi_rank != 0:
            logging.info('skip because the rank {} != 0'.format(self.mpi_rank))
            return

        if self.ignore_evaluate:
            logging.info('ignore evaluate as instructed')
            return

        self._ensure_initialized()
        if not predict_file:
            model_file = self._get_checkpoint_file(iteration=self.max_iter)
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
            deteval_iter(
                    dataset.iter_data(self.test_split, 'label',
                        version=self.test_version),
                    predict_file,
                    report_file=evaluate_file,
                    **self.kwargs)
        elif self.evaluate_method == 'coco_box':
            from qd.cocoeval import convert_gt_to_cocoformat
            from qd.cocoeval import convert_to_cocoformat
            from qd.cocoeval import coco_eval_json
            gt_json = dataset.get_data(self.test_split, 'label.cocoformat',
                    version=self.test_version) + '.json'
            gt_iter = dataset.iter_data(self.test_split, 'label',
                        version=self.test_version)

            if not op.isfile(gt_json) or self.force_evaluate:
                convert_gt_to_cocoformat(gt_iter, gt_json)

            predict_json = predict_file + '.cocoformat.json'
            is_empty = False
            if worth_create(predict_file, predict_json) or self.force_evaluate:
                annotations = convert_to_cocoformat(predict_file, predict_json)
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
            label_tsv_file = dataset.get_data(self.test_split, 'label',
                    self.test_version)
            top1 = evaluate_topk(predict_file, label_tsv_file)
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

def evaluate_topk(predict_file, label_tsv_file):
    correct = 0
    total = 0
    iter_label_tsv = tsv_reader(label_tsv_file)
    iter_pred_tsv = tsv_reader(predict_file)
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
        if any(g['class'] == curr_pred_best for g in curr_gt_rects):
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
    for k in init_dict:
        if k in name_to_param and same_shape(init_dict[k], name_to_param[k]):
            real_init_dict[k] = init_dict[k]
        else:
            logging.info('ignore {} in init model'.format(k))
            num_ignored = num_ignored + 1

    logging.info('number of param ignored = {}'.format(num_ignored))

    model.load_state_dict(real_init_dict, strict=False)

if __name__ == '__main__':
    init_logging()

    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)

