from qd.qd_pytorch import ModelPipeline
import os.path as op
from qd.tsv_io import TSVDataset
import logging
import os
from qd.qd_common import write_to_yaml_file


class YoloV2PtPipeline(ModelPipeline):
    def __init__(self, **kwargs):
        super(YoloV2PtPipeline, self).__init__(**kwargs)
        self._default.update({'stagelr': [0.0001,0.001,0.0001,0.00001],
            'effective_batch_size': 64,
            'ovthresh': [0.5],
            'display': 100,
            'lr_policy': 'multifixed',
            'momentum': 0.9,
            'is_caffemodel_for_predict': False,
            'workers': 4})
        self.num_train_images = None

    def append_predict_param(self, cc):
        super(YoloV2PtPipeline, self).append_predict_param(cc)
        if self.yolo_predict_session_param:
            test_input_size = self.yolo_predict_session_param.get('test_input_size', 416)
            if test_input_size != 416:
                cc.append('InputSize{}'.format(test_input_size))

    def _get_checkpoint_file(self, epoch=None, iteration=None):
        assert epoch is None, 'not supported'
        if iteration is None:
            iteration = self.max_iter
        iteration = self.parse_iter(iteration)
        return op.join(self.output_folder, 'snapshot',
                "model_iter_{:07d}.pt".format(iteration))

    def monitor_train(self):
        # not implemented. but we just return rather than throw exception
        # need to use maskrcnn checkpointer to load and save the intermediate
        # models.
        return

    def train(self):
        dataset = TSVDataset(self.data)
        param = {
                'is_caffemodel': True,
                'logdir': self.output_folder,
                'labelmap': dataset.get_labelmap_file(),
                }
        if not self.use_treestructure:
            param.update({'use_treestructure': False})
        else:
            param.update({'use_treestructure': True,
                          'tree': dataset.get_tree_file()})
        train_data_path = '$'.join([self.data, 'train'])

        lr_policy = self.lr_policy
        snapshot_prefix = os.path.join(self.output_folder, 'snapshot', 'model')
        max_iter = self.parse_iter(self.max_iter)

        stageiter = list(map(lambda x:int(x*max_iter/7000),
            [100,5000,6000,7000]))

        solver_param = {
                'lr_policy': lr_policy,
                'gamma': 0.1,
                'momentum': self.momentum,
                'stagelr': self.stagelr,
                'stageiter': stageiter,
                'weight_decay': self.weight_decay,
                'snapshot_prefix': op.relpath(snapshot_prefix),
                'max_iter': max_iter,
                'snapshot': 500,
                }

        solver_yaml = op.join(self.output_folder,
                'solver.yaml')

        write_to_yaml_file(solver_param, solver_yaml)

        basemodel = self.basemodel
        assert self.effective_batch_size % self.mpi_size == 0
        if 'WORLD_SIZE' in os.environ:
            assert int(os.environ['WORLD_SIZE']) == self.mpi_size
        else:
            os.environ['WORLD_SIZE'] = str(self.mpi_size)
        if 'RANK' in os.environ:
            assert int(os.environ['RANK']) == self.mpi_rank
        else:
            os.environ['RANK'] = str(self.mpi_rank)

        param.update({'solver': solver_yaml,
                      'only_backbone': False,
                      'batch_size': self.effective_batch_size // self.mpi_size,
                      'workers': self.workers,
                      'model': basemodel,
                      'restore': False,
                      'latest_snapshot': None,
                      'display': self.display,
                      'train': train_data_path})

        is_local_rank0 = self.mpi_local_rank == 0

        from mmod.file_logger import FileLogger
        log = FileLogger(self.output_folder, is_master=self.is_master,
                is_rank0=is_local_rank0)

        param.update({'distributed': self.distributed,
                      'local_rank': self.mpi_local_rank,
                      'dist_url': self.get_dist_url()})

        if 'last_fixed_param' in self.kwargs:
            param['last_fixed_param'] = self.kwargs['last_fixed_param']

        param_yaml = op.join(self.output_folder,
                'param.yaml')
        write_to_yaml_file(param, param_yaml)

        if self.yolo_train_session_param is not None:
            from qd.qd_common import dict_update_nested_dict
            dict_update_nested_dict(param, self.yolo_train_session_param)

        from mmod import yolo_train_session
        snapshot_file = yolo_train_session.main(param, log)

        if self.is_master:
            import shutil
            shutil.copyfile(snapshot_file,
                    self._get_checkpoint_file())

    def predict(self, model_file, predict_result_file):
        if self.mpi_rank != 0:
            logging.info('ignore to predict for non-master process')
            return
        from mmod.file_logger import FileLogger
        from mmod import yolo_predict_session

        dataset = TSVDataset(self.data)
        param = {
                'model': model_file,
                'labelmap': dataset.get_labelmap_file(),
                'test': '$'.join([self.test_data, self.test_split]),
                'output': predict_result_file,
                'logdir': self.output_folder,
                'batch_size': self.test_batch_size,
                'workers': self.workers,
                'is_caffemodel': self.is_caffemodel_for_predict,
                'single_class_nms': False,
                'obj_thresh': 0.01,
                'thresh': 0.01,
                'log_interval': 100,
                'test_max_iter': self.test_max_iter,
                'device': self.device,
                }
        is_tree = self.use_treestructure
        param['use_treestructure'] = is_tree
        if is_tree:
            param['tree'] = dataset.get_tree_file()

        if self.yolo_predict_session_param is not None:
            param.update(self.yolo_predict_session_param)

        is_local_rank0 = self.mpi_local_rank == 0
        log = FileLogger(self.output_folder, is_master=self.is_master,
                is_rank0=is_local_rank0)
        yolo_predict_session.main(param, log)

