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
    def __init__(self,
                 student_encoder,
                 teacher_encoder,
                 dim=128,
                 K=65536,
                 m=0.999,
                 T=0.07,
                 teacher_T=0.2,
                 mlp=False, mlp_bn=False, mlp_num=1,
                 key_mlp_num=None,
                 multi_crop=False,
                 dim_mlp=None,
                 teacher_mlp_dim=None,
                 ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        print_frame_info()

        self.K = K
        self.T = T
        self.teacher_T = teacher_T
        self.multi_crop = multi_crop

        # create the encoders
        # num_classes is the output fc dimension
        encoder_q = student_encoder
        encoder_k = teacher_encoder

        if mlp:
            def replace_fc(encoder, dim_mlp, num):
                fc_in_dim = encoder.fc.weight.shape[1]
                if dim_mlp is None:
                    dim_mlp = fc_in_dim
                out_dim = encoder.fc.weight.shape[0]
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
            replace_fc(encoder_q, dim_mlp, mlp_num)
            replace_fc(encoder_k, teacher_mlp_dim, key_mlp_num)

        from qd.torch_common import freeze_parameters
        freeze_parameters(encoder_k)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_im_idx", torch.zeros(K))
        self.queue_im_idx[:] = -1
        self.register_buffer('queue_im_crop', torch.zeros(4, K))

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.mpi_size = get_mpi_size()
        self.mpi_rank = get_mpi_rank()

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        # the purpose is to have a key with module, so that our classification
        # module can load such model
        self.module = self.encoder_q
        self.iter = 0

        self.queue_gather = True

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
        torch.distributed.broadcast(x, src=0)

    def forward(self, data):
        if self.multi_crop:
            return self.forward_multi(*args)
        else:
            return self.forward_two(data)

    def forward_key(self, im_k):
        with torch.no_grad():  # no gradient to keys
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

    def forward_two(self, data):
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
        im_q = data['image']
        im_k = data['image']

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        k = self.forward_key(im_k)

        result = self.forward_moco(q, k, verbose=verbose)

        self._dequeue_and_enqueue(k)

        self.iter += 1
        return result

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
        #labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        label_neg = torch.einsum('nc,ck->nk', [k, self.queue.clone().detach()])
        labels = torch.cat([
            torch.ones(len(k), 1, device=self.queue.device),
            label_neg,
        ], dim=1)
        labels /= self.teacher_T
        labels = labels.softmax(dim=1)

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
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(self.mpi_size)]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output

class DistillLoss(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.module = model
        self.criterion = criterion

    def forward(self, image, origin_target=None):
        # image could be 2 or 3 or multiple views
        feature_label = self.module(image)
        loss = self.criterion(*feature_label)
        if isinstance(loss, dict):
            return loss
        else:
            return {'criterion_loss': loss}

class MocoDistillPipeline(MaskClassificationPipeline):
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
            'teacher_net': 'resnet50',
            'teacher_T': 0.2,
            'loss_type': 'kl_ce',
            'dataset_type': 'single_dict',
        }

        from qd.qd_common import max_iter_mult
        curr_default['stageiter'] = [max_iter_mult(self.max_iter, x / 200.)
                for x in [120, 160]]
        self._default.update(curr_default)

    def get_train_model(self):
        teacher_encoder = self.get_encoder(self.teacher_net)(self.moco_dim)
        student_encoder = self.get_encoder(self.net)(self.moco_dim)
        model = MoCo(
            student_encoder,
            teacher_encoder,
            dim=self.moco_dim,
            K=self.moco_k,
            T=self.moco_t,
            mlp=self.mlp,
            mlp_bn=self.mlp_bn,
            mlp_num=self.mlp_num,
            multi_crop=self.aug_plus.startswith('multi_crop'),
            dim_mlp=self.dim_mlp,
            teacher_T=self.teacher_T,
            teacher_mlp_dim=self.teacher_mlp_dim,
        )

        criterion = self._get_criterion()
        # we need wrap model output and criterion into one model, to re-use
        # maskrcnn trainer
        model = DistillLoss(model, criterion)
        return model

    def get_encoder(self, net):
        if self.net.startswith('efficientnet'):
            from qd.layers.efficient_det import EfficientNet
            encoder = lambda num_classes: EfficientNet.from_name(
                net,
                override_params={'num_classes': num_classes})
        else:
            from qd.qd_common import execute_func
            encoder = lambda num_classes: execute_func({
                'from': 'qd.pipelines.moco_distill',
                'import': net,
                'param': {'num_classes': num_classes},})
        return encoder

    def get_test_model(self):
        # used for feasture extraction
        encoder = self.get_encoder(self.net)
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
        if self.aug_plus == 'plus_dict':
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
            return final
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
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        widen=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

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
        self.base_width = width_per_group

        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            3, num_out_filters, kernel_size=7, stride=2, padding=3, bias=False)
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
        self.fc = nn.Linear(num_out_filters * block.expansion, num_classes)

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

def resnet50w2(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)

def resnet50w4(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)

def resnet50w5(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, **kwargs)

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

