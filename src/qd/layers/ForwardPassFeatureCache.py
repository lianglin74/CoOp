import torch
import logging
from collections import OrderedDict
from maskrcnn_benchmark.structures.bounding_box import BoxList


def clone(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().clone().detach()
    elif isinstance(x, list):
        return [clone(sub) for i, sub in enumerate(x)]
    elif isinstance(x, tuple):
        return tuple(list(clone(sub) for i, sub in enumerate(x)))
    elif isinstance(x, BoxList):
        return x.to(torch.device('cpu')).copy_with_fields(x.fields())
    else:
        logging.info('unknown type and return directly {}'.format(
            type(x)))
        return x

def sumarize_data(x):
    info = []
    if isinstance(x, torch.Tensor):
        info.append('shape = {}'.format(x.shape))
        info.append('mean = {}'.format(x.mean()))
        info.append('std = {}'.format(x.std()))
    elif isinstance(x, list):
        info = [sumarize_data(sub) for sub in x]
    elif isinstance(x, tuple):
        info = [sumarize_data(sub) for sub in x]
    else:
        info = [str(x)]
    return '; '.join(info)


class ForwardPassFeatureCache(torch.nn.Module):
    def __init__(self, model):
        super(ForwardPassFeatureCache, self).__init__()

        self.model = model
        self.model_to_start_time = OrderedDict()
        self.model_to_end_time = OrderedDict()

        self.module_to_name = OrderedDict([(m, n) for n, m in self.model.named_modules()])

        self.module_to_output = OrderedDict()
        self.module_to_input = OrderedDict()

        def forward_hooker(m, i, o):
            self.module_to_output[m] = clone(o)
            self.module_to_input[m] = clone(i)

        self.model.register_forward_hook(forward_hooker)
        for _, m in self.model.named_modules():
            if m is not self.model:
                m.register_forward_hook(forward_hooker)

    def forward(self, *args, **kwargs):
        self.module_to_output.clear()
        self.module_to_input.clear()

        result = self.model(*args, **kwargs)
        return result

    def sumarize_feature(self):
        for module, out in self.module_to_output.items():
            if module not in self.module_to_name:
                logging.info('module not exist')
                continue
            name = self.module_to_name[module]
            summary = sumarize_data(out)
            logging.info('name = {}; summary = {}'.format(name, summary))

def create_forward_pass_feature_cache(m):
    return ForwardPassFeatureCache(m)

