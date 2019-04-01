from .qd_common import init_logging


def create_parameters(**kwargs):
    kwargs.update({'data': 'voc20',
                   'max_iter': 10000,
                   'effective_batch_size': 64})
    return kwargs

def create_pipeline(kwargs):
    from .qd_pytorch import YoloV2PtPipeline
    return YoloV2PtPipeline(**kwargs)

def test_model_pipeline(**kwargs):
    '''
    run the script by

    mpirun -npernode 4 \
            python script_with_this_function_called.py
    '''
    init_logging()
    kwargs.update(create_parameters(**kwargs))
    pip = create_pipeline(kwargs)

    if kwargs.get('monitor_train_only'):
        pip.monitor_train()
    else:
        pip.ensure_train()
        pip.ensure_predict()
        pip.ensure_evaluate()
