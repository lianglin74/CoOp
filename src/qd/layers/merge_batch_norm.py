import copy

import torch
from torch.nn.parameter import Parameter

class MergeBatchNorm(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = merge_bn_into_conv(copy.deepcopy(module))

    def forward(self, *args, **kwargs):
        result = self.module(*args, **kwargs)
        return result

    def backward(self, *args, **kwargs):
        raise NotImplementedError("MergeBatchNorm is for inference only")


class DummyLayer(torch.nn.Module):
    def forward(self, input):
        return input


def is_mergable_batch_norm(m):
    mergable_bns = [torch.nn.BatchNorm2d]
    try:
        import maskrcnn_benchmark
        mergable_bns.append(maskrcnn_benchmark.layers.batch_norm.FrozenBatchNorm2d)
    except:
        import logging
        logging.info('seems like maskrcnn is not installed')
    return any(isinstance(m, bn) for bn in mergable_bns)

def merge_bn_into_conv(model):
    '''
    Merge conv-bn layers into one conv with bias
    NOTE: model is modified after calling this method
    '''
    leaf_module_list = [(n, m) for n, m in model.named_modules()
            if list(m.children()) == []]
    to_be_absorbed = set()
    num_leaf_modules = len(leaf_module_list)
    m_idx = 0
    while m_idx < num_leaf_modules:
        _, module = leaf_module_list[m_idx]
        if isinstance(module, torch.nn.Conv2d) and m_idx + 1 < num_leaf_modules:
            next_layer_name, next_layer = leaf_module_list[m_idx + 1]
            old_w = module.weight.data.clone().detach()
            out_channel = old_w.shape[0]

            if is_mergable_batch_norm(next_layer):
                mean = next_layer.running_mean
                var = next_layer.running_var
                eps = next_layer.eps
                if next_layer.weight is not None:
                    scale = next_layer.weight.data.clone().detach()
                    bias = next_layer.bias.data.clone().detach()
                else:
                    import mtorch
                    if m_idx + 2 < num_leaf_modules and \
                             isinstance(leaf_module_list[m_idx + 2][1],
                                     mtorch.caffetorch.Scale):
                        scale_layer_name, scale_layer = leaf_module_list[m_idx
                                + 2]
                        scale = scale_layer.weight.data.clone().detach()
                        bias = scale_layer.bias.data.clone().detach()
                        assert scale_layer_name not in to_be_absorbed
                        to_be_absorbed.add(scale_layer_name)
                        m_idx += 1
                    else:
                        scale = torch.ones([out_channel])
                        bias = torch.zeros([out_channel])
                invstd = scale / torch.Tensor.sqrt(var + eps)

                if module.bias is None:
                    m_bias = torch.zeros([out_channel], device=scale.device)
                    module.register_parameter('bias', Parameter(m_bias))
                old_b = module.bias.data.clone().detach()

                module.weight.data = old_w * invstd.view(out_channel, 1, 1, 1)
                module.bias.data = (old_b - mean) * invstd + bias

                assert next_layer_name not in to_be_absorbed
                to_be_absorbed.add(next_layer_name)
                m_idx += 1
        m_idx += 1

    output_model = convert_layers_to_dummy(model, to_be_absorbed)
    assert len(to_be_absorbed) == 0
    return output_model


def convert_layers_to_dummy(module, to_be_converted_layers, module_name=None):
    module_output = module
    if module_name in to_be_converted_layers:
        module_output = DummyLayer()
        to_be_converted_layers.remove(module_name)

    for cur_name, child in module.named_children():
        cur_module_name = '.'.join([module_name, cur_name]) if module_name else cur_name
        child = convert_layers_to_dummy(child, to_be_converted_layers, module_name=cur_module_name)
        module_output.add_module(cur_name, child)

    del module
    return module_output
