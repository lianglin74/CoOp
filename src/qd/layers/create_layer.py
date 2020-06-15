def create_frozen_batchnorm(**kwargs):
    from torch.nn import BatchNorm2d
    bn = BatchNorm2d(**kwargs)
    bn.train(False)
    return bn

