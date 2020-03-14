import math
import torch
import time
from qd.qd_common import list_to_dict


class MeanSigmaMetricLogger(object):
    def __init__(self, delimiter="\t"):
        from maskrcnn_benchmark.utils.metric_logger import MetricLogger
        self.mean_meters = MetricLogger(delimiter=delimiter)
        self.sq_meters = MetricLogger(delimiter=delimiter)

    def update(self, **kwargs):
        self.mean_meters.update(**kwargs)
        self.sq_meters.update(**dict((k, v * v) for k, v in kwargs.items()))

    def get_info(self):
        key_to_sigma = {}
        for k, v in self.mean_meters.meters.items():
            mean = v.global_avg
            mean_square = self.sq_meters.meters[k].global_avg
            sigma = mean_square - mean * mean
            sigma = math.sqrt(sigma)
            key_to_sigma[k] = sigma

        result = []
        for name, mean_meter in self.mean_meters.meters.items():
            result.append({'name': name,
                'global_avg': mean_meter.global_avg,
                'median': mean_meter.median,
                'sigma': key_to_sigma[name]})
        return result

    def __str__(self):
        result = self.get_info()

        loss_str = []
        for info in result:
            loss_str.append(
                    "{}: {:.4f} ({:.4f}+-{:.4f})".format(
                        info['name'],
                        info['median'],
                        info['global_avg'],
                        info['sigma'])
            )
        return self.mean_meters.delimiter.join(loss_str)

class ForwardPassTimeChecker(torch.nn.Module):
    def __init__(self, module, skip=2):
        super(ForwardPassTimeChecker, self).__init__()
        self.module = module

        self.module_start_times = []
        self.module_costs = []

        def forward_pre_hooker(m, i):
            self.module_start_times.append((m, time.time()))

        def forward_hooker(m, i, o):
            end_time = time.time()
            start_m, start_time = self.module_start_times.pop()
            assert start_m == m
            self.module_costs.append((m, end_time - start_time))

        self.meters = MeanSigmaMetricLogger(delimiter="\n")

        for _, m in self.module.named_modules():
            m.register_forward_pre_hook(forward_pre_hooker)
            m.register_forward_hook(forward_hooker)

        self.module_to_name = dict((m, n) for n, m in self.module.named_modules())
        self.skip = skip

    def forward(self, *args, **kwargs):

        self.module_start_times.clear()
        self.module_costs.clear()

        result = self.module(*args, **kwargs)
        if self.skip <= 0:
            for m, c in self.module_costs:
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

