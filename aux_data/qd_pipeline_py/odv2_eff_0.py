from qd.pipeline import run_training_pipeline

def test_fcos_pipeline():
    env = {
        #'run_type': 'local',
        #'run_type': 'aml',
        #'run_type': 'debug',
        'run_type': 'print',

        # aml config should be in aux_data/aml/sc.yaml, will be ignored if
        # run_type is not aml
        'cluster': 'sc',
        'num_gpu': 0.5,
        'availability_check': True,
    }
    swap_params = {
        'data': [
            'Tax1300V14.4_0.0_0.0_with_bb',
        ],
        'net': [
            'e2e_faster_rcnn_efficient_det_tb',
        ],

        'MODEL$BACKBONE$EFFICIENT_DET_COMPOUND': [
            0,
        ],
        'MODEL$RPN$FPN_POST_NMS_TOP_N_EACH_IMAGE_TRAIN': [
            1000,
        ],

        'affine_resize': [
            'RC',
        ],
        'train_size_mode': [
            'mm_cut512',
        ],

        'scale': [
            16,
        ],
        'test_batch_size': [
            16,
        ],

    'effective_batch_size': [
        16,
    ],
    'max_iter': [
        '60e',
    ],
    'SOLVER$WEIGHT_DECAY': [
        1e-6,
    ],
    'log_step': [100],
    'base_lr': [
        0.05,
    ],
    'expid_prefix': [
        'FCOS',
    ],
    'pipeline_type': [
        {
            'from': 'qd.pipelines.fcos',
            'import': 'FCOSPipeline',
        },
    ],
    'env': [
        env,
    ],
    }
    run_training_pipeline(swap_params)

if __name__ == '__main__':
    from qd.qd_common import init_logging
    init_logging()
    test_fcos_pipeline()
