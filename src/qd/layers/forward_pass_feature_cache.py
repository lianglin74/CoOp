import torch
import logging
from collections import OrderedDict
#from maskrcnn_benchmark.structures.bounding_box import BoxList


#def clone(x):
    #if isinstance(x, torch.Tensor):
        #return x.cpu().clone().detach()
    #elif isinstance(x, list):
        #return [clone(sub) for i, sub in enumerate(x)]
    #elif isinstance(x, tuple):
        #return tuple(list(clone(sub) for i, sub in enumerate(x)))
    #elif isinstance(x, BoxList):
        #return x.to(torch.device('cpu')).copy_with_fields(x.fields())
    #else:
        #logging.info('unknown type and return directly {}'.format(
            #type(x)))
        #return x

def sumarize_data_by_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.float().abs().mean())
    elif hasattr(x, 'bbox'):
        # this is boxlist used in maskrcnn
        x1 = float(x.bbox.float().abs().mean())
        if hasattr(x, 'extra_fields'):
            x2 = sumarize_data_by_float(x.extra_fields)
        return x1 + x2
    elif isinstance(x, dict):
        return sum([sumarize_data_by_float(x[sub]) for sub in x])
    elif isinstance(x, list):
        return sum([sumarize_data_by_float(sub) for sub in x])
    elif isinstance(x, tuple):
        return sum([sumarize_data_by_float(sub) for sub in x])
    else:
        return 0

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

        self.module = model
        self.model_to_start_time = OrderedDict()
        self.model_to_end_time = OrderedDict()

        self.module_to_name = OrderedDict([(m, n) for n, m in self.module.named_modules()])

        self.module_to_output = OrderedDict()
        self.module_to_input = OrderedDict()

        def forward_hooker(m, i, o):
            self.module_to_input[m] = sumarize_data_by_float(i)
            self.module_to_output[m] = sumarize_data_by_float(o)

        self.module.register_forward_hook(forward_hooker)
        for _, m in self.module.named_modules():
            if m is not self.module:
                m.register_forward_hook(forward_hooker)

    def forward(self, *args, **kwargs):
        self.module_to_output.clear()
        self.module_to_input.clear()

        result = self.module(*args, **kwargs)
        return result

    def sumarize_feature(self):
        for module, out in self.module_to_output.items():
            if module not in self.module_to_name:
                logging.info('module not exist')
                continue
            name = self.module_to_name[module]
            in_s = self.module_to_input[module]
            logging.info('name = {}; in = {}; out = {}'.format(name, in_s, out))

def create_forward_pass_feature_cache(m):
    return ForwardPassFeatureCache(m)

