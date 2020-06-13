import torch
from torch.optim.optimizer import Optimizer
from copy import deepcopy


def replace_ema_param(checkpoint, decay):
    ema_info = [info for info in checkpoint['optimizer']['ema_infos'] if info['decay'] == decay]
    assert len(ema_info) == 1, len(ema_info)
    ema_param_groups = ema_info[0]['param_groups']
    for g in ema_param_groups:
        for p, n in zip(g['params'], g['param_names']):
            checkpoint['model'][n] = p

def get_params_with_name(model):
    '''
    used for ema optimizer, where the name is needed
    '''
    name_params = list(model.named_parameters())
    name_params = list(filter(lambda x: x[1].requires_grad, name_params))
    return [{'params': [x[1] for x in name_params],
             'param_names': [x[0] for x in name_params]}]

@torch.no_grad()
def ema_update(in_param_groups, out_param_groups, decay):
    for group, ema_group in zip(in_param_groups, out_param_groups):
        for p, ema_p in zip(group['params'], ema_group['params']):
            if p.grad is None:
                continue
            ema_p.data.mul_(decay)
            ema_p.data.add_(1. - decay, p.data)

class EMAOptimizer(Optimizer):
    def __init__(self, optimizer, decays=[0.999, 0.9998, 0.9999]):
        self.optimizer = optimizer
        # names should be available to know which parameter to replace with the
        # averaged version
        for g in optimizer.param_groups:
            assert 'param_names' in g
        if not isinstance(decays, list) and \
                not isinstance(decays, tuple):
            assert decays < 1
            decays = [decays]
        self.ema_infos = [{'decay': d, 'param_groups': deepcopy(self.param_groups)}
                                 for d in decays]

    def __getstate__(self):
        result = self.optimizer.__getstate__()
        result['ema_infos'] = self.ema_infos
        return result

    def __setstate__(self, state):
        self.ema_infos = state.pop('ema_infos')
        self.optimizer.__setstate__(state)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        result = self.optimizer.state_dict()
        result['ema_infos'] = self.ema_infos
        return result

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        self.ema_infos = state_dict.pop('ema_infos')
        self.optimizer.load_state_dict(state_dict)

    def step(self, *args, **kwargs):
        self.optimizer.step(*args, **kwargs)
        self.ema_update()

    def ema_update(self):
        for ema_info in self.ema_infos:
            ema_update(self.optimizer.param_groups,
                       ema_info['param_groups'],
                       ema_info['decay'])

