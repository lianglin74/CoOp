# no dependency on maskrcnn-benchmark

# - run on 4 GPUs locally
#   mpirun -4 python src/qd/examples/efficient_det0.py
#
# - run on 1 GPU and would like to go step by step
#   1) decrease the batch size to avoid out of memory issue by setting effective_batch_size
# to a lower value, e.g. 2;
#   2) change num_gpu as 1 in env the following
#   3) run the following to start the training
#       python src/qd/examples/efficient_det0.py
#
# - train on the custom dataset
#   change the value of data in swap_params to other names.
#
# - submit it to aml with 4 GPU
#   a -c cluster_name -n 4 submit python src/qd/examples/efficient_det0.py
#   the tool is under src/qd/gpu_clusters

env = {
    'run_type': 'local',
    #'cluster': 'wev32',
    'num_gpu': 4,
}
swap_params = [
    ('data', [
        'coco2017Full',
    ]),
    ('net', [
        0,
    ]),
    ('full_expid', [
        'experiment_name_please_change_it_manually'
    ]),
    ('dict_trainer', [
        True,
    ]),
    ('anchor_scale', [
        3,
    ]),
    ('adaptive_up', [
        True,
    ]),
    ('reg_weight', [
        2,
    ]),
    ('min_size_range32', [
        (512 - 128, 512 + 128),
    ]),
    ('efficient_net_simple_padding', [
        True,
    ]),
    ('reg_loss_type', [
        'GIOU',
    ]),
    ('expid_prefix', [
        'EffDet',
    ]),
    #('scale', [2]),
    ('num_workers', [8]),
    ('init_full_expid', [
        'O365_0_EffDet_basemodel3d79a_BS128_MaxIter100e_LR0.2_IMIN384.640_cosine_WD1e-05_CR_S_GIOU_Assign1_Adapt_A3',
    ]),
    ('scheduler_type', [
        'cosine',
    ]),
    ('prior_prob', [
        0.01,
    ]),
    ('at_least_1_assgin', [
        True,
    ]),
    ('base_lr', [
        0.1,
    ]),
    ('weight_decay',[
        1e-5,
    ]),
    ('log_step', [1]),
    ('effective_batch_size', [
        64,
    ]),
    ('max_iter', [
        '100e',
    ]),
    ('pipeline_type', [
        {
            'from': 'qd.pipelines.efficient_det_pipeline',
            'import': 'EfficientDetPipeline',
        },
    ]),
    ('env', [env]),
]
from qd.pipeline import run_training_pipeline
run_training_pipeline(swap_params)

