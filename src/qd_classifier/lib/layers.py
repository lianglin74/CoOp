import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class ResNetFeatureExtract(nn.Module):
    def __init__(self, model):
        super(ResNetFeatureExtract, self).__init__()
        layers = ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3",
                  "layer4", "avgpool", "fc"]
        for l_name in layers:
            setattr(self, l_name, getattr(model, l_name))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        fea = x
        x = self.fc(x)

        return x, fea

class ResNetInput112(nn.Module):
    def __init__(self, arch, num_classes):
        super(ResNetInput112, self).__init__()
        orig_model = models.__dict__[arch](num_classes=num_classes)
        self.layer_names = ["conv1", "bn1", "relu", "layer1", "layer2", "layer3",
                  "layer4", "avgpool", "fc"]   # remove maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        for l_name in self.layer_names[1:]:
            setattr(self, l_name, getattr(orig_model, l_name))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        for l_name in self.layer_names:
            x = getattr(self, l_name)(x)
            if l_name == "avgpool":
                x = x.view(x.size(0), -1)
        return x


class SigmoidCrossEntropyLossWithBalancing(nn.Module):
    def __init__(self, class_wise_negative_sample_weights):
        super(SigmoidCrossEntropyLossWithBalancing, self).__init__()
        self.class_wise_negative_sample_weights = torch.from_numpy(class_wise_negative_sample_weights).float().cuda()
        self.norm = self.class_wise_negative_sample_weights.mean().item()

    def forward(self, input, target):
        # weights = torch.bernoulli(self.class_wise_keep_ratio).cuda()
        weights = torch.ones_like(target) * self.class_wise_negative_sample_weights
        weights = weights + target
        weights = torch.clamp(weights, 0.0, 1.0)
        return F.binary_cross_entropy_with_logits(input, target, weights)/self.norm

class CCSLoss(nn.Module):
    """
    Classification vector-centered Cosine Similarity (CCS) loss at https://arxiv.org/pdf/1707.05574.pdf
    """
    def __init__(self):
        super(CCSLoss, self).__init__()

    def forward(self, feature, weight, label):
        """
        Args:
        feature: tensor of size N*D, where N is the batch size, D is the dimension
        weight: tensor of size C*D, where C is the number of classes,
        label: tensor of size N, each value is int in [0, C)
        """
        # NOTE: use detach() to not calculate grad w.r.t. weight in ccs_loss
        weight_data = weight.data.clone().detach()
        label_data = label.data.clone().detach()
        loss = 0
        num_samples, fea_dim = feature.shape
        num_cls, w_dim  = weight.shape
        assert fea_dim == w_dim, "feature dim {} does not match with weight dim {}".format(fea_dim, w_dim)
        assert num_samples == label.shape[0]

        weight_per_sample = weight_data[label_data]
        dot_product = feature * weight_per_sample
        dot_product = torch.sum(dot_product, dim=1)
        fea_norm = torch.norm(feature, dim=1)
        w_norm = torch.norm(weight_per_sample, dim=1)
        loss = torch.mean(dot_product / (fea_norm * w_norm))
        return -loss

class SmoothSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.1):
        self.eps = eps
        super(SmoothSoftmaxCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        eps = self.eps
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()

        return loss


class ModelCCSLoss(nn.Module):
    # used for mask-rcnn trainer engine
    def __init__(self, model, criterion, ccs_loss_param, label_smoothing=False):
        super().__init__()
        self.module = model if ccs_loss_param==0 else ResNetFeatureExtract(model)
        self.ccs_loss_param = ccs_loss_param
        self.ccs_loss_layer = CCSLoss()
        self.label_smoothing = label_smoothing
        if label_smoothing:
            assert isinstance(criterion, nn.CrossEntropyLoss())
            self.criterion = SmoothSoftmaxCrossEntropyLoss()
            self.loss_name = 'label_smooth_loss'
        else:
            self.criterion = criterion
            self.loss_name = 'criterion_loss'

        self.acc_meter = get_accuracy_calculator()

    def forward(self, data, target):
        # compute output
        all_outputs = self.module(data)

        loss_dict = dict()
        if self.ccs_loss_param > 0:
            output, feature = all_outputs[0], all_outputs[1]
            weight = self.module.fc.weight
            ccs_loss = self.ccs_loss_layer(feature, weight, target)

            loss_dict['ccs_loss'] = self.ccs_loss_param*ccs_loss
        else:
            output = all_outputs

        loss_dict[self.loss_name] = self.criterion(output, target)

        self.acc_meter.calc(output, target)
        return loss_dict


def create_model(arch, num_classes, pretrained=False):
    # create model
    if pretrained:
        logging.info("=> using pre-trained model '{}'".format(arch))
        model_urls[arch] = model_urls[arch].replace('https://', 'http://')
        model = models.__dict__[arch](pretrained=True)
        if model.fc.weight.shape[0] != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        # for m in model.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)
        #         nn.init.constant_(m.bias, 0)
        torch.nn.init.xavier_uniform_(model.fc.weight)
    else:
        logging.info("=> creating model '{}'".format(arch))
        model = models.__dict__[arch](num_classes=num_classes)

    return model
