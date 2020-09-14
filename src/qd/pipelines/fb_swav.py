from qd.qd_pytorch import torch_load
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline

import argparse
import math
import os
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex
from apex.parallel.LARC import LARC
import os.path as op
import argparse
from logging import getLogger
import pickle
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from qd.torch_common import synchronize

import os
import logging
import time
from datetime import timedelta
import pandas as pd

import torch.distributed as dist
from logging import getLogger

import cv2

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

logger = getLogger()

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, mpi_rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if mpi_rank > 0:
            filepath = "%s-%i" % (filepath, mpi_rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time

    return logger


class PD_Stats(object):
    """
    Log stuff with pandas library
    """

    def __init__(self, path, columns):
        self.path = path

        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)

        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)
logger = getLogger()

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


#def init_distributed_mode(args):
    #"""
    #Initialize the following variables:
        #- world_size
        #- mpi_rank
    #"""

    #args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    #if args.is_slurm_job:
        #args.mpi_rank = int(os.environ["SLURM_PROCID"])
        #args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            #os.environ["SLURM_TASKS_PER_NODE"][0]
        #)
    #else:
        ## multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        ## read environment variables
        #args.mpi_rank = int(os.environ["RANK"])
        #args.world_size = int(os.environ["WORLD_SIZE"])

    ## prepare distributed
    #dist.init_process_group(
        #backend="nccl",
        #init_method=args.dist_url,
        #world_size=args.world_size,
        #mpi_rank=args.mpi_rank,
    #)

    #return

def initialize_exp(params, *args):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.output_folder, "stats" + str(params.mpi_rank) + ".pkl"), args
    )

    # create a logger
    logger = create_logger(
        os.path.join(params.model_folder, "train.log"), mpi_rank=params.mpi_rank
    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.model_folder)
    logger.info("")
    return logger, training_stats


def restart_from_checkpoint(ckp_paths, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    # look for a checkpoint in exp repository
    if isinstance(ckp_paths, list):
        for ckp_path in ckp_paths:
            if os.path.isfile(ckp_path):
                break
    else:
        ckp_path = ckp_paths

    if not os.path.isfile(ckp_path):
        return

    logger.info("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(
        ckp_path, map_location="cuda:" + str(torch.distributed.get_rank() % torch.cuda.device_count())
    )

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(msg)
            except TypeError:
                msg = value.load_state_dict(checkpoint[key])
            logger.info("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
        else:
            logger.warning(
                "=> failed to load {} from checkpoint '{}'".format(key, ckp_path)
            )

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """computes and stores the average and current value"""

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

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

class MultiCropDataset(object):
    def __init__(
        self,
        data,
        size_crops,
        nmb_crops,
        min_scale_crops,
        max_scale_crops,
        return_index=False,
    ):
        from qd.data_layer.dataset import TSVSplitProperty
        self.tsv = TSVSplitProperty(data, 'train')
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        self.return_index = return_index

        trans = []
        color_transform = transforms.Compose([get_color_distortion(), RandomGaussianBlur()])
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                color_transform,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        from qd.qd_common import pilimg_from_base64
        image = pilimg_from_base64(self.tsv[index][-1])
        image = image.convert('RGB')
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops

    def __len__(self):
        return len(self.tsv)

class RandomGaussianBlur(object):
    def __call__(self, img):
        do_it = np.random.rand() > 0.5
        if not do_it:
            return img
        sigma = np.random.rand() * 1.9 + 0.1
        return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
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
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
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
    def __init__(
            self,
            block,
            layers,
            zero_init_residual=False,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            normalize=False,
            output_dim=0,
            hidden_mlp=0,
            nmb_prototypes=0,
            eval_mode=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group * widen

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False
        )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.eval_mode:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet50w2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


def resnet50w4(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)

def distributed_sinkhorn(Q, nmb_iters, mpi_size):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q

        u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (mpi_size * Q.shape[1])

        curr_sum = torch.sum(Q, dim=1)
        dist.all_reduce(curr_sum)

        for it in range(nmb_iters):
            u = curr_sum
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


class SwAVPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        default_param = {
            'data': 'imagenet2012Full',
            'nmb_crops': [2], # [2, 6]
            'size_crops': [224], # [224, 96]
            'min_scale_crops': [0.14],
            'max_scale_crops': [1.],
            'crops_for_assign': [0, 1],
            'temperature': 0.1,
            'epsilon': 0.05,
            'sinkhorn_iterations': 3,
            'feat_dim': 128,
            'nmb_prototypes': 3000,
            'queue_length': 3840,
            'epoch_queue_starts': 15,
            'effective_batch_size': 256, # 4 gpus
            'base_lr': 0.3, #
            'final_lr': 0,
            'freeze_prototypes_niters': 313,
            'weight_decay': 1e-6,
            'warmup_epochs': 10,
            'start_warmup': 0,
            'net': 'resnet50',
            'workers': 10,
            'checkpoint_freq': 25,
            'use_fp16': True,
            'sync_bn': 'pytorch',
            'seed': 31,
        }
        if self.swav_mc:
            default_param['nmb_crops'] = [2, 6]
            default_param['size_crops'] = [224, 96]
            default_param['min_scale_crops'] = [0.14, 0.05]
            default_param['max_scale_crops'] = [1, 0.14]
            default_param['freeze_prototypes_niters'] = 5005

        self._default.update(default_param)

    def train(self):
        args = self
        fix_random_seeds(args.seed)
        logger, training_stats = initialize_exp(args, "epoch", "loss")

        # build data
        train_dataset = MultiCropDataset(
            args.data,
            args.size_crops,
            args.nmb_crops,
            args.min_scale_crops,
            args.max_scale_crops,
        )
        train_dataset[0]

        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )
        logging.info(len(train_loader))
        logger.info("Building data done with {} images loaded.".format(len(train_dataset)))

        # build model
        model = resnet50(
            normalize=True,
            hidden_mlp=2048 if args.net == "resnet50" else 2048 * int(args.net[-1]),
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
        )
        # synchronize batch norm layers
        if args.sync_bn == "pytorch":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        elif args.sync_bn == "apex":
            process_group = None
            if args.mpi_size // 8 > 0:
                process_group = apex.parallel.create_syncbn_process_group(args.mpi_size // 8)
            model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
        # copy model to GPU
        model = model.cuda()
        if args.mpi_rank == 0:
            logger.info(model)
        logger.info("Building model done.")

        # build optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
        warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
        iters = np.arange(len(train_loader) * (args.max_epoch - args.warmup_epochs))
        cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                             math.cos(math.pi * t / (len(train_loader) * (args.max_epoch - args.warmup_epochs)))) for t in iters])
        lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        logger.info("Building optimizer done.")

        # init mixed precision
        if args.use_fp16:
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
            logger.info("Initializing mixed precision done.")

        # wrap model
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.mpi_local_rank],
            find_unused_parameters=True,
        )

        # optionally resume from a checkpoint
        to_restore = {"epoch": 0}
        restart_from_checkpoint(
            os.path.join(args.model_folder, "checkpoint.pth.tar"),
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
            amp=apex.amp,
        )
        start_epoch = to_restore["epoch"]

        # build the queue
        queue = None
        queue_path = os.path.join(args.model_folder, "queue" + str(args.mpi_rank) + ".pth")
        if os.path.isfile(queue_path):
            queue = torch.load(queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        args.queue_length -= args.queue_length % (args.batch_size_per_gpu * args.mpi_size)

        cudnn.benchmark = True

        for epoch in range(start_epoch, args.max_epoch):

            # train the network for one epoch
            logger.info("============ Starting epoch %i ... ============" % epoch)

            # set sampler
            train_loader.sampler.set_epoch(epoch)

            # optionally starts a queue
            if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
                queue = torch.zeros(
                    len(args.crops_for_assign),
                    args.queue_length // args.mpi_size,
                    args.feat_dim,
                ).cuda()

            # train the network
            scores, queue = self.train_epoch(train_loader, model, optimizer, epoch, lr_schedule, queue)
            training_stats.update(scores)

            # save checkpoints
            if args.mpi_rank == 0:
                save_dict = {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if args.use_fp16:
                    save_dict["amp"] = apex.amp.state_dict()
                torch.save(
                    save_dict,
                    os.path.join(args.model_folder, "checkpoint.pth.tar"),
                )
                if epoch % args.checkpoint_freq == 0 or epoch == args.max_epoch - 1:
                    shutil.copyfile(
                        os.path.join(args.model_folder, "checkpoint.pth.tar"),
                        os.path.join(args.model_folder, "ckp-" + str(epoch) + ".pth"),
                    )
            if queue is not None:
                torch.save({"queue": queue}, queue_path)

        model_final = op.join(self.model_folder,
                              'ckp-{}.pth'.format(self.max_epoch - 1))
        last_iter = self._get_checkpoint_file(iteration=self.max_iter)
        if self.mpi_rank == 0:
            if not op.isfile(last_iter):
                shutil.copy(model_final, last_iter)

        synchronize()
        return last_iter

    def train_epoch(self, train_loader, model, optimizer, epoch, lr_schedule, queue):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        softmax = nn.Softmax(dim=1).cuda()
        model.train()
        use_the_queue = False

        end = time.time()
        for it, inputs in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # update learning rate
            iteration = epoch * len(train_loader) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[iteration]

            # normalize the prototypes
            with torch.no_grad():
                w = model.module.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.module.prototypes.weight.copy_(w)

            # ============ multi-res forward passes ... ============
            embedding, output = model(inputs)
            embedding = embedding.detach()
            bs = inputs[0].size(0)

            # ============ swav loss ... ============
            loss = 0
            for i, crop_id in enumerate(self.crops_for_assign):
                with torch.no_grad():
                    out = output[bs * crop_id: bs * (crop_id + 1)]

                    # time to use the queue
                    if queue is not None:
                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                queue[i],
                                model.module.prototypes.weight.t()
                            ), out))
                        # fill the queue
                        queue[i, bs:] = queue[i, :-bs].clone()
                        queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                    # get assignments
                    q = torch.exp(out / self.epsilon).t()
                    q = distributed_sinkhorn(q, self.sinkhorn_iterations, self.mpi_size)[-bs:]

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum(self.nmb_crops)), crop_id):
                    p = softmax(output[bs * v: bs * (v + 1)] / self.temperature)
                    subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
                loss += subloss / (np.sum(self.nmb_crops) - 1)
            loss /= len(self.crops_for_assign)

            # ============ backward and optim step ... ============
            optimizer.zero_grad()
            if self.use_fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # cancel some gradients
            if iteration < self.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            optimizer.step()

            # ============ misc ... ============
            losses.update(loss.item(), inputs[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if self.mpi_rank ==0 and it % 50 == 0:
                logger.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Lr: {lr:.4f}".format(
                        epoch,
                        it,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        lr=optimizer.optim.param_groups[0]["lr"],
                    )
                )
        return (epoch, losses.avg), queue


    def get_test_model(self):
        # used for feasture extraction
        args = self
        model = resnet50(
            normalize=True,
            hidden_mlp=2048 if args.net == "resnet50" else 2048 * int(args.net[-1]),
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
        )
        model = self.model_surgery(model)
        return model

    def predict_iter_forward(self, model, inputs):
        return model(inputs['image'])

    def ensure_evaluate(self):
        pass

