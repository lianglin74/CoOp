import torch
from collections import OrderedDict
import time


class ForwardPassTimeChecker(torch.nn.Module):
    def __init__(self, model):
        super(ForwardPassTimeChecker, self).__init__()
        self.model = model
        self.model_to_start_time = OrderedDict()
        self.model_to_end_time = OrderedDict()

        def forward_hooker(m, i, o):
            assert m not in self.model_to_end_time
            self.model_to_end_time[m] = time.time()

        def forward_pre_hooker(m, i):
            assert m not in self.model_to_start_time
            self.model_to_start_time[m] = time.time()

        self.model.register_forward_hook(forward_hooker)
        self.model.register_forward_pre_hook(forward_pre_hooker)

    def forward(self, *args, **kwargs):
        self.model_to_start_time.clear()
        self.model_to_end_time.clear()
        result = self.model(*args, **kwargs)
        return result

