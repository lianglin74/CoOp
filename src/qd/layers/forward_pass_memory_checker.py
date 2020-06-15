import torch

def get_all_py_obj():
    # prints currently alive Tensors and Variables
    import gc
    all_obj = []
    for obj in gc.get_objects():
        #if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        if torch.is_tensor(obj):
            ele_length = 4 if obj.dtype == torch.float32 else 4
            all_obj.append({'type': type(obj),
                           'numel': obj.numel(),
                           'memory': obj.numel() * ele_length,
                           'shape': obj.shape})
        else:
            raise NotImplementedError
    return all_obj

def get_total_memory(all_obj):
    return sum(o['memory'] for o in all_obj)

class ForwardPassMemoryChecker(torch.nn.Module):
    def __init__(self, module, skip=2):
        super(ForwardPassMemoryChecker, self).__init__()
        self.module = module

        self.module_objs= []
        self.module_costs = []

        def forward_pre_hooker(m, i):
            self.module_objs.append((m, get_all_py_obj()))

        def forward_hooker(m, i, o):
            start_m, start_objs = self.module_objs.pop()
            curr_objs = get_all_py_obj()
            self.module_costs.append((m, get_total_memory(curr_objs)-
                get_total_memory(start_objs)))

        #for _, m in self.module.named_modules():
            #m.register_forward_pre_hook(forward_pre_hooker)
            #m.register_forward_hook(forward_hooker)

        self.module_to_name = dict((m, n) for n, m in self.module.named_modules())
        self.skip = skip

    def forward(self, *args, **kwargs):

        self.module_objs.clear()
        self.module_costs.clear()
        import logging
        logging.info(get_total_memory(get_all_py_obj()))
        result = self.module(*args, **kwargs)
        logging.info(get_total_memory(get_all_py_obj()))
        import ipdb;ipdb.set_trace(context=15)
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

