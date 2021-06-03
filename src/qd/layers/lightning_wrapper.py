import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

class LightningModule(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.module = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        from qd.torch_common import recursive_to_device
        x = recursive_to_device(x, self.device, non_blocking=False)
        return self.module(x)

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(batch)
        return sum(loss_dict.values())

    def configure_optimizers(self):
        return [self.optimizer], [
            {
                'scheduler': self.scheduler,
                'interval': 'step',
            }]
