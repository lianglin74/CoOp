from collections import OrderedDict
import os
import shutil
import torch

import torchvision.models as models

def load_from_checkpoint(model_file):
    is_cpu_only = not torch.cuda.is_available()
    print("=> loading checkpoint '{}', use GPU: {}".format(model_file, not is_cpu_only))
    if is_cpu_only:
        checkpoint = torch.load(model_file, map_location='cpu')
    else:
        checkpoint = torch.load(model_file)
    arch = checkpoint['arch']
    model = models.__dict__[arch](num_classes=checkpoint['num_classes'])

    load_model_state_dict(model, checkpoint['state_dict'], skip_unmatched_layers=False)
    print("=> loaded checkpoint '{}' (epoch {}), use GPU: {}".format(model_file, checkpoint['epoch'], not is_cpu_only))

    # load labelmap
    if 'labelmap' in checkpoint:
        labelmap = checkpoint['labelmap']
    else:
        labelmap = [str(i) for i in range(checkpoint['num_classes'])]

    return model, labelmap

def save_checkpoint(state, epoch, model_dir, is_best):
    filename = os.path.join(model_dir, 'model_epoch_{:04d}.pth.tar'.format(epoch))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_dir, 'model_best.pth.tar'))
    # comment out to keep only the latest 3 models
    # if epoch >= 3:
    #     old_filename = os.path.join(output_dir, '%s-%04d.pth.tar' % (prefix, epoch - 3))
    #     if os.path.isfile(old_filename):
    #         os.remove(old_filename)

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def align_and_update_state_dicts(model_state_dict, loaded_state_dict, skip_unmatched_layers=False):
    """
    If skip_unmatched_layers is True, it will skip layers when the shape mismatch.
    Otherwise, it will raise error.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if model_state_dict[key].shape != loaded_state_dict[key_old].shape and skip_unmatched_layers:
            # if layer weights does not match in size, skip this layer
            print("SKIPPING LAYER {} because of size mis-match".format(key))
            continue
        model_state_dict[key] = loaded_state_dict[key_old]

def load_model_state_dict(model, loaded_state_dict, skip_unmatched_layers=False):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, skip_unmatched_layers)

    # use strict loading
    model.load_state_dict(model_state_dict)
