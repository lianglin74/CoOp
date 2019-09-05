# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, get_rank
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
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    log_start = time.time()
    # from qd.qd_common import is_hvd_initialized
    # use_hvd = is_hvd_initialized()
    use_hvd = False
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        #if len(images.image_sizes) == 0:
        #    continue
        if targets.shape[0] == 0:
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device)
        else:
            targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        if losses != losses:
            logging.info('NaN encountered!')
            arguments['images'] = images
            arguments['targets'] = targets
            checkpointer.save("NaN_context_{}".format(get_rank()), **arguments)
            print(losses)
            raise RuntimeError('NaN encountered!')

        # reduce losses over all GPUs for logging purposes
        if not use_hvd:
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
        else:
            assert False
            losses_reduced = sum(loss for loss in loss_dict.values())
            meters.update(loss=losses_reduced, **loss_dict)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        if device.type == 'cpu':
            losses.backward()
        else:
            if not use_hvd:
                # from apex import amp
                # with amp.scale_loss(losses, optimizer) as scaled_losses:
                #     scaled_losses.backward()
                losses.backward()
            else:
                assert False
                losses.backward()
        optimizer.step()
        # in pytorch >= 1.1, do scheduler.step() after optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % log_step == 0 or iteration == max_iter:
            speed = get_world_size() * log_step * len(targets) / (time.time() - log_start)
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        'speed: {speed:.1f} images/sec',
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                        "acc: {acc}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    speed=speed,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    acc=model.module.acc_meter.result_str(),
                )
            )
            log_start = time.time()
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_iter_{:07d}".format(iteration), **arguments)
        if iteration >= max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
