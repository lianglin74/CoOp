import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class KLEntropyPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        curr_default = {
            'latent_num': 1000,
            'entropy_weight': 0.5,
        }

        self._default.update(curr_default)

        self.num_class = self.latent_num

    def _get_criterion(self):
        from qd.layers.kl_entropy import KLEntropyLoss
        return KLEntropyLoss(self.entropy_weight)

    def ensure_predict(self):
        pass

    def ensure_evaluate(self):
        pass

