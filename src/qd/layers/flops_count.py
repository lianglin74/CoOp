import math
import torch
import time
from qd.qd_common import list_to_dict
from qd.logger import MeanSigmaMetricLogger


class FlopsCount(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

        self.module_costs = []
        self.ignoremodules = []

        def forward_hooker(m, i, o):
            if len(list(m.named_children())) > 0:
                return
            if isinstance(m, torch.nn.Conv2d):
                assert len(i) == 1 and len(m.kernel_size) == 2
                assert i[0].shape[0] == o.shape[0]
                flops = o.numel() * i[0].shape[1] * m.kernel_size[0] * m.kernel_size[1] / m.groups
            elif isinstance(m, torch.nn.Linear):
                assert len(i) == 1
                i = i[0]
                assert len(i.shape) == len(o.shape) == 2
                assert i.shape[0] == o.shape[0]
                flops = i.shape[0] * i.shape[1] * o.shape[1]
            else:
                self.ignoremodules.append((m, 1))
                return
            self.module_costs.append((m, flops))

        self.meters = MeanSigmaMetricLogger(delimiter="\n")
        self.ignore_meters = MeanSigmaMetricLogger(delimiter="\n")

        for _, m in self.module.named_modules():
            m.register_forward_hook(forward_hooker)

        self.module_to_name = dict((m, n) for n, m in self.module.named_modules())

    def forward(self, *args, **kwargs):
        self.module_costs.clear()
        self.ignoremodules.clear()

        result = self.module(*args, **kwargs)
        module_to_costs = list_to_dict(self.module_costs, 0)
        for m, cs in module_to_costs.items():
            c = sum(cs)
            name = self.module_to_name[m]
            self.meters.update(**{name: c})

        module_to_costs = list_to_dict(self.ignoremodules, 0)
        for m, cs in module_to_costs.items():
            c = sum(cs)
            name = self.module_to_name[m]
            self.ignore_meters.update(**{name: c})
        return result

    def get_info(self):
        from qd.qd_common import try_get_cpu_info
        from qd.gpu_util import try_get_nvidia_smi_out
        info = {'meters': self.meters.get_info(),
                'cpu_info': try_get_cpu_info(),
                'gpu_info': try_get_nvidia_smi_out()}
        return info


