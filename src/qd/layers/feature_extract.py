import torch
import logging
from collections import OrderedDict


class FeatureExtract(torch.nn.Module):
    # everytime when we call model(inputs), we call model.feature to get the
    # features
    def __init__(self, model, module_names, funcs=None):
        super().__init__()
        if isinstance(module_names, str):
            module_names = [module_names]

        self.model = model

        self.target_modules = []
        for module_name in module_names:
            target_modules =  [(n, m) for n, m in self.model.named_modules()
                               if n.endswith(module_name)]
            assert len(target_modules) == 1, 'confusing {}'.format(
                '; '.join(n for n, m in target_modules))
            self.target_modules.append(target_modules[0][1])

        self.module_to_name = OrderedDict(zip(self.target_modules,
                                       module_names))

        self.module_to_feature = OrderedDict((m, None) for m in
                                             self.target_modules)
        if funcs is None:
            self.module_to_func = OrderedDict(((m, None)
                                               for m in self.target_modules))
        else:
            self.module_to_func = OrderedDict(zip(self.target_modules,
                                                  funcs))

        def forward_hooker(m, i, o):
            if m in self.module_to_feature:
                func = self.module_to_func[m]
                if func is None:
                    self.module_to_feature[m] = o.detach().clone()
                else:
                    self.module_to_feature[m] = func(m, i, o)

        self.model.register_forward_hook(forward_hooker)
        for _, m in self.model.named_modules():
            if m is not self.model:
                m.register_forward_hook(forward_hooker)

    def get_features(self):
        return [f for _, f in self.module_to_feature.items()]

    def clear_cached_feature(self):
        for m in self.module_to_feature:
            self.module_to_feature[m] = None

    def forward(self, *args, **kwargs):
        self.clear_cached_feature()
        output = self.model(*args, **kwargs)
        return output

