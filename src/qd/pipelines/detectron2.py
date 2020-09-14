import os.path as op
from qd.tsv_io import TSVDataset
import logging
import torch
from pprint import pformat
from qd.layers.loss import ModelLoss
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
import torch.nn as nn
from qd.qd_common import json_dump
import copy
from qd.qd_common import load_from_yaml_file
from qd.qd_common import dict_update_nested_dict
from qd.qd_common import merge_dict_to_cfg
from qd.qd_common import dict_update_path_value
from qd.qd_common import dump_to_yaml_str
import datetime
import time

import torch.distributed as dist

from qd.qd_common import get_mpi_size as get_world_size
from qd.qd_common import get_mpi_rank as get_rank

from maskrcnn_benchmark.utils.metric_logger import MetricLogger


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        name_dims = [(k, v.dim()) for k, v in zip(loss_names, all_losses)]
        for k in range(1, len(name_dims)):
            if name_dims[k][1] != name_dims[0][1]:
                logging.info('{}={} not equal {}={}'.format(name_dims[k][0],
                    name_dims[k][1], name_dims[0][0], name_dims[0][1]))
                raise Exception()
        all_losses = torch.stack(all_losses, dim=0)
        from qd.qd_common import is_hvd_initialized
        if not is_hvd_initialized():
            dist.reduce(all_losses, dst=0)
        else:
            import horovod.torch as hvd
            all_losses = hvd.allreduce(all_losses, average=False)
        if get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def forward_backward(model, data,
        optimizer,
        arguments, checkpointer, use_hvd,
        meters, device):
     loss_dict = model(data)

     losses = sum(loss for loss in loss_dict.values())
     if losses != losses:
         logging.info('NaN encountered!')
         arguments['data'] = data
         checkpointer.save("NaN_context_{}".format(get_rank()), **arguments)
         raise RuntimeError('NaN encountered!')

     # reduce losses over all GPUs for logging purposes
     if not use_hvd:
         loss_dict_reduced = reduce_loss_dict(loss_dict)
         losses_reduced = sum(loss for loss in loss_dict_reduced.values())
         meters.update(loss=losses_reduced, **loss_dict_reduced)
     else:
         losses_reduced = sum(loss for loss in loss_dict.values())
         meters.update(loss=losses_reduced, **loss_dict)

     # Note: If mixed precision is not used, this ends up doing nothing
     # Otherwise apply loss scaling for mixed-precision recipe
     if device.type == 'cpu':
         losses.backward()
     else:
         #if not use_hvd:
             #from apex import amp
             #with amp.scale_loss(losses, optimizer) as scaled_losses:
                 #scaled_losses.backward()
         #else:
         losses.backward()

def partition_data(images, targets, num):
    if num == 1 or len(images.image_sizes) < num:
        return [(images, targets)]
    each = len(images.image_sizes) // num
    result = []
    from maskrcnn_benchmark.structures.image_list import ImageList
    for i in range(num):
        start = i * each
        end = start + each
        curr_tensors = images.tensors[start: end]
        curr_sizes = images.image_sizes[start: end]
        curr_imagelist = ImageList(curr_tensors, curr_sizes)
        curr_target = targets[start: end]
        result.append((curr_imagelist, curr_target))
    return result

def average_gradients(model):
    size = dist.get_world_size()
    if size == 1:
        return
    size = float(size)
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

from qd.qd_common import try_once
@try_once
def try_save_intermediate_snapshot(checkpointer, iteration, arguments):
    checkpointer.save("model_{:07d}".format(iteration), **arguments)

def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    log_step=20,
    data_partition=1,
    max_iter=1,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    log_start = time.time()
    from qd.qd_common import is_hvd_initialized
    use_hvd = is_hvd_initialized()
    data_loader_iter = iter(data_loader)
    from detectron2.utils.events import EventStorage
    with EventStorage(start_iter):
        for iteration in range(start_iter, max_iter):
            data = next(data_loader_iter)
            data_time = time.time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            optimizer.zero_grad()

            forward_backward(model, data,
                    optimizer,
                    arguments, checkpointer, use_hvd,
                    meters, device)

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            if iteration > start_iter + 5:
                # we will skip the first few iterations since the time cost
                # evaluation for those are not good
                meters.update(time=batch_time, data=data_time)

            if iteration % log_step == 0 or iteration == max_iter:
                speed = get_world_size() * log_step * len(data) / (time.time() - log_start)
                if hasattr(meters, 'time'):
                    eta_seconds = meters.time.global_avg * (max_iter - iteration)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    eta_string = 'Unknown'

                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            'speed: {speed:.1f} images/sec',
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        speed=speed,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                log_start = time.time()
            if iteration % checkpoint_period == 0:
                # with blobfuse, saving could fail with unknown reason. Instead of
                # saving and crashing, we do a best-effort manner.
                try_save_intermediate_snapshot(checkpointer, iteration, arguments)
            if iteration >= max_iter:
                checkpointer.save("model_final", **arguments)
                if get_rank() > 0:
                    old_value = checkpointer.save_to_disk
                    checkpointer.save_to_disk = True
                    checkpointer.save("model_final_{}".format(get_rank()), **arguments)
                    checkpointer.save_to_disk = old_value
            scheduler.step()


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def load_default_config(net, detectron2_root):
    folder = 'COCO-Detection'
    if net.endswith('syncbn'):
        folder = 'Misc'
    cfg_file = op.join(detectron2_root, 'configs',
            folder,
            net + '.yaml')
    # cfg_file contains the base keyword, which has to be resolved by
    # cfg.merge_from_file. If we use load_from_yaml_file, we can not
    # resolve that keyword
    from fvcore.common.config import CfgNode
    param = CfgNode.load_yaml_with_base(cfg_file)
    return param


class Detectron2Pipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super(Detectron2Pipeline, self).__init__(**kwargs)

        detectron2_root = op.join('src', 'detectron2')

        # the basic flow is to move the parameter in net.yaml to kwargs. Then,
        # adapt the parameters in kwargs to the parameter path for cfg, but the
        # conversion is still in kwargs. Finally, move the valid parameters in
        # kwargs to cfg. In this way, we can easily configure any parameters in
        # kwargs.
        if self.net is not None and self.net != 'Unknown':
            param = load_default_config(self.net,
                    detectron2_root)
        else:
            param = {}
        self.default_net_param = copy.deepcopy(param)
        dict_update_nested_dict(self.kwargs, param, overwrite=False)

        # train data
        train_args = {
                'data': self.data,
                'split': 'train',
                'version': self.train_version,
                }
        dict_update_path_value(self.kwargs, 'DATASETS$TRAIN',
                ('$' + dump_to_yaml_str(train_args),))

        # test data
        test_args = {
                'data': self.test_data,
                'split': self.test_split,
                }
        dict_update_path_value(self.kwargs, 'DATASETS$TEST',
                ('$' + dump_to_yaml_str(test_args),))

        dict_update_path_value(self.kwargs,
                               'SOLVER$IMS_PER_BATCH',
                               self.effective_batch_size)

        self.labelmap = TSVDataset(self.data).load_labelmap()
        # no need to add 1 to the number classes as it will do it in
        # detectron2/modeling/roi_heads/fast_rcnn.py:FastRCNNOutputLayers
        dict_update_path_value(self.kwargs,
                               'MODEL$ROI_HEADS$NUM_CLASSES',
                               len(self.labelmap))

        dict_update_path_value(self.kwargs,
                               'SOLVER$MAX_ITER',
                               self.parse_iter(self.max_iter))
        dict_update_path_value(self.kwargs,
                               'DATALOADER$NUM_WORKERS',
                               self.num_workers)

        if self.stageiter:
            dict_update_path_value(
                self.kwargs,
                'SOLVER$STEPS',
                tuple([self.parse_iter(i) for i in self.stageiter]))
        else:
            max_iter = self.parse_iter(self.max_iter)
            dict_update_path_value(
                self.kwargs,
                'SOLVER$STEPS',
                (6*max_iter//9, 8*max_iter//9))

        dict_update_path_value(self.kwargs,
                               'OUTPUT_DIR',
                               op.join(self.output_folder, 'snapshot'))

        from detectron2.config import get_cfg
        cfg = get_cfg()
        merge_dict_to_cfg(self.kwargs, cfg)

        logging.info('cfg = \n{}'.format(cfg))
        self.cfg = cfg

    def has_background_output(self):
        # we should always use self.kwargs rather than cfg. cfg is only used by
        # mask-rcnn lib, we only set it instead of read
        from qd.qd_common import dict_has_path
        if dict_has_path(self.kwargs,
                'MODEL$ROI_BOX_HEAD$CLASSIFICATION_LOSS'):
            p = self.kwargs['MODEL']['ROI_BOX_HEAD']['CLASSIFICATION_LOSS']
        else:
            p = 'CE'
        if p in ['BCE'] or \
                any(p.startswith(x) for x in ['IBCE']):
            return False
        elif p in ['CE', 'MCEB']:
            # 0 is background
            return True
        elif p in ['tree']:
            return True
        else:
            raise NotImplementedError()

    def append_predict_param(self, cc):
        super(Detectron2Pipeline, self).append_predict_param(cc)

    def init_apex_amp(self, model, optimizer):
        return model, optimizer

    def monitor_train(self):
        # not implemented. but we just return rather than throw exception
        # need to use maskrcnn checkpointer to load and save the intermediate
        # models.
        return

    def do_train(self, model, train_loader, optimizer, scheduler, checkpointer,
            arguments):
        device = torch.device('cuda')
        assert model.training
        do_train(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpointer=checkpointer,
            device=device,
            checkpoint_period=self.get_snapshot_steps(),
            arguments=arguments,
            log_step=self.log_step,
            max_iter=self.max_iter,
        )

    def get_optimizer(self, model):
        from detectron2.solver import build_optimizer
        optimizer = build_optimizer(self.cfg, model)
        return optimizer

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        from detectron2.solver import build_lr_scheduler
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return scheduler

    def get_train_data_loader(self, start_iter=0):
        from detectron2.data import build_detection_train_loader
        return build_detection_train_loader(self.cfg)

    def get_test_data_loader(self):
        from detectron2.data import build_detection_test_loader
        dataset_name = '${}'.format(dump_to_yaml_str({
            'data': self.test_data,
            'split': self.test_split,
            }))
        return build_detection_test_loader(self.cfg, dataset_name)

    def get_train_model(self):
        from detectron2.modeling import build_model
        model = build_model(self.cfg)
        return model

    def get_backbone_model(self):
        raise NotImplementedError

    def get_test_model(self):
        model = self.get_train_model()
        model.eval()
        return model

    def train(self):
        if self.detectron2_trainer or True:
            if self.mpi_size > 1:
                from detectron2.utils import comm
                assert comm._LOCAL_PROCESS_GROUP is None
                num_machines = self.mpi_size // self.mpi_local_size
                machine_rank = self.mpi_rank // self.mpi_local_size
                for i in range(num_machines):
                    ranks_on_i = list(range(i * self.mpi_local_size,
                                            (i + 1) * self.mpi_local_size))
                    pg = dist.new_group(ranks_on_i)
                    if i == machine_rank:
                        comm._LOCAL_PROCESS_GROUP = pg

            from src.detectron2.tools.train_net import Trainer
            trainer = Trainer(self.cfg)
            trainer.resume_or_load(resume=True)
            if self.cfg.TEST.AUG.ENABLED:
                from detectron2.engine import hooks
                trainer.register_hooks(
                    [hooks.EvalHook(0, lambda: trainer.test_with_TTA(self.cfg, trainer.model))]
                )
            trainer.train()

            model_final = op.join(self.output_folder, 'snapshot', 'model_final.pth')
            last_iter = self._get_checkpoint_file(iteration=self.max_iter)
            if self.mpi_rank == 0:
                if not op.isfile(last_iter):
                    import shutil
                    shutil.copy(model_final, last_iter)
            return last_iter
        else:
            return super().train()

    def predict_iter(self, dataloader, model, softmax_func, meters):
        from tqdm import tqdm
        start = time.time()
        for i, inputs in tqdm(enumerate(dataloader),
                total=len(dataloader)):
            if self.test_max_iter is not None and i >= self.test_max_iter:
                # this is used for speed test, where we only would like to run a
                # few images
                break
            meters.update(data=time.time() - start)
            start = time.time()
            if not isinstance(inputs, list):
                inputs = inputs.to(self.device)
            meters.update(input_to_cuda=time.time() - start)
            start = time.time()
            outputs = model(inputs)
            meters.update(model=time.time() - start)
            start = time.time()
            if softmax_func is not None:
                outputs = softmax_func(outputs)
            meters.update(softmax=time.time() - start)
            for row in self.predict_output_to_tsv_row(outputs, inputs):
                yield row
            start = time.time()

    def predict_output_to_tsv_row(self, outputs, inputs):
        for _output, _input in zip(outputs, inputs):
            key = _input['key']
            ins = _output['instances'].to('cpu')
            boxes = ins.pred_boxes.tensor.tolist()
            scores = ins.scores.tolist()
            cls_idx = ins.pred_classes.tolist()
            rects = [{
                'rect': rect, 'conf': conf, 'class': self.labelmap[idx]}
                for rect, conf, idx in zip(boxes, scores, cls_idx)]
            yield key, json_dump(rects)

    def _get_test_normalize_module(self):
        return

