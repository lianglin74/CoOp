import math
import torch
import time
from qd.qd_common import list_to_dict
from qd.logger import MeanSigmaMetricLogger


class ForwardPassTimeChecker(torch.nn.Module):
    def __init__(self, module, skip=2):
        super(ForwardPassTimeChecker, self).__init__()
        self.module = module

        self.module_start_times = []
        self.module_costs = []
        from collections import defaultdict
        self.started_module2count = defaultdict(int)

        def forward_pre_hooker(m, i):
            self.module_start_times.append((m, time.time()))
            self.started_module2count[m] += 1

        def forward_hooker(m, i, o):
            end_time = time.time()
            start_m, start_time = self.module_start_times.pop()
            self.started_module2count[start_m] -= 1
            assert start_m == m
            if self.started_module2count[start_m] == 0:
                # in some cases, a module will call itself multiple times, and
                # we only count the one which is the most outside
                self.module_costs.append((m, end_time - start_time))

        self.meters = MeanSigmaMetricLogger(delimiter="\n")

        for _, m in self.module.named_modules():
            m.register_forward_pre_hook(forward_pre_hooker)
            m.register_forward_hook(forward_hooker)

        self.module_to_name = dict((m, n) for n, m in self.module.named_modules())
        self.skip = skip

    def forward(self, *args, **kwargs):
        self.started_module2count.clear()
        self.module_start_times.clear()
        self.module_costs.clear()

        result = self.module(*args, **kwargs)
        if self.skip <= 0:
            module_to_costs = list_to_dict(self.module_costs, 0)
            for m, cs in module_to_costs.items():
                c = sum(cs)
                name = self.module_to_name[m]
                self.meters.update(**{name: c})
        else:
            self.skip -= 1
        return result

    def get_time_info(self):
        from qd.qd_common import try_get_cpu_info
        from qd.gpu_util import try_get_nvidia_smi_out
        info = {'meters': self.meters.get_info(),
                'cpu_info': try_get_cpu_info(),
                'gpu_info': try_get_nvidia_smi_out()}
        return info

