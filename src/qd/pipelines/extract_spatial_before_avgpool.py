import torch
from torch import nn
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline

def extract_input(m, i, o):
    if isinstance(i, tuple) and len(i) == 1:
        i = i[0]
    return i.detach().clone()

def extract_spatial_input(m, i, o):
    if isinstance(i, tuple) and len(i) == 1:
        i = i[0]
    i = i.detach()
    i = torch.sqrt((i * i).sum(dim=1) / i.shape[1])
    return i

class ExtractImageFeature(MaskClassificationPipeline):
    def wrap_feature_extract(self, model):
        assert len(self.predict_extract) >= 2
        funcs = [extract_input, extract_spatial_input]
        from qd.layers.feature_extract import FeatureExtract
        model = FeatureExtract(model, self.predict_extract, funcs)
        return model
