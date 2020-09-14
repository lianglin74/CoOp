import torch


class ForwardImageModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model
    def forward(self, data_dict):
        return self.module(data_dict['image'])

