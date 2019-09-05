from yacs.config import CfgNode as CN

_C = CN()

# necessary inputs
_C.full_expid = ""
_C.arch = "resnet18"
_C.data = ""
_C.output_dir = "./output"
_C.test_data = ""

# data
_C.enlarge_bbox = 2.0
_C.data_aug = 0

_C.force_predict = False
_C.cache_policy = ''
_C.debug = False
_C.print_freq = 100
_C.random_seed = 6

# epoch, batch
_C.epochs = 120
_C.effective_batch_size = 256
_C.num_workers = 4

# resume
_C.start_epoch = 0
_C.pretrained = True
_C.init_from = ''
_C.skip_unmatched_layers = False
_C.restore_latest_snapshot = False
_C.resume = ''

# distributed
_C.dist_backend = "nccl"
_C.dist_url_tcp_port = 23456
_C.init_method_type = "tcp"

# solver
_C.lr_policy = "STEP"
_C.lr = 0.1
_C.step_size = 30
_C.gamma = 0.1
_C.milestones = (30,60,90)
_C.momentum = 0.9
_C.weight_decay = 0.001
_C.bn_no_weight_decay = True

_C.balance_class = False
_C.balance_sampler = False
_C.ccs_loss_param = 2.0
_C.label_smoothing = False

_C.finetune = False
_C.fixfeature = False
_C.fixpartialfeature = False
_C.input_size = 224

# not used, for compatiablity
_C.expid = ""
_C.neg_weight_file = ""
_C.force = True
_C.BatchNormEvalMode = False
_C.batch_size = 0      # use effective batch size


def create_config(args):
    cfg = _C.clone()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg
