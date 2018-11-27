import os
import shutil
import torch


def save_checkpoint(state, is_best, prefix, epoch, output_dir):
    filename = os.path.join(output_dir, '%s-%04d.pth.tar' % (prefix, epoch))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(output_dir, 'model_best.pth.tar'))
