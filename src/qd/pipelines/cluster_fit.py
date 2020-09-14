import torch
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
from torch import nn


def get_fc_layer(model):
    targets = [m for n, m in model.named_modules() if n.endswith('.fc')]
    assert len(targets) == 1
    return targets[0]

class MultiHeadLinear(nn.Module):
    def __init__(self, num, in_features, out_features):
        super().__init__()
        self.num = num
        self.linear = nn.Linear(in_features, out_features * num)

    def forward(self, x):
        x = self.linear(x)
        num_image, num_feat = x.shape
        return x.view(-1, num_feat // self.num)

class ClusterFitClassification(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._default.update({
            'fc2mlp': False,
            'fc2mlp_num': 1,
            'heads': 1,
            })

    def model_surgery(self, model):
        if self.fc2mlp:
            module_list = []
            fc = model.fc
            dim_mlp = fc.weight.shape[1]
            for i in range(self.fc2mlp_num):
                module_list.append(nn.Linear(dim_mlp, dim_mlp))
                module_list.append(nn.ReLU())
            module_list.append(fc)
            model.fc = nn.Sequential(*module_list)
        if self.heads != 1:
            assert not self.fc2mlp
            model.fc = MultiHeadLinear(
                self.heads,
                model.fc.weight.shape[1],
                model.fc.weight.shape[0])
        return super().model_surgery(model)

