from qd.torch_common import describe_tensor
from pprint import pformat
from qd.logger import SmoothedValue
import datetime
import logging
from qd.torch_common import recursive_to_device
import time

import torch
import torch.distributed as dist

from qd.qd_common import get_mpi_rank, get_mpi_size
from qd.logger import MetricLogger
from qd.torch_common import to


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_mpi_size()
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
        if get_mpi_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def forward_backward(model, images, targets,
        optimizer,
        arguments, checkpointer, use_hvd,
        meters, device, loss_scalar, no_update=False):
    start_fw = time.time()
    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values()) * loss_scalar
    end_fw = time.time()
    if losses != losses:
        logging.info('NaN encountered!')
        arguments['images'] = images
        arguments['targets'] = targets
        checkpointer.save("NaN_context_{}".format(get_mpi_rank()), **arguments)
        raise RuntimeError('NaN encountered!')

    # reduce losses over all GPUs for logging purposes
    if not use_hvd:
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
    else:
        losses_reduced = sum(loss for loss in loss_dict.values())
        meters.update(loss=losses_reduced, **loss_dict)
    meters.update(fw_time=(end_fw-start_fw))

    #from pprint import pformat
    #x = model.get_time_info()['meters']
    #logging.info(pformat([(y['name'], y['global_avg']) for y in x]))
    #import ipdb;ipdb.set_trace(context=15)

    # Note: If mixed precision is not used, this ends up doing nothing
    # Otherwise apply loss scaling for mixed-precision recipe
    start_bw = time.time()
    if not no_update:
       if device.type == 'cpu':
           losses.backward()
       else:
           #if not use_hvd:
               #from apex import amp
               #with amp.scale_loss(losses, optimizer) as scaled_losses:
                   #scaled_losses.backward()
           #else:
           losses.backward()
    meters.update(bw_time=(time.time() - start_bw))

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
def try_save_intermediate_snapshot(checkpointer, name, arguments):
    checkpointer.save(name, **arguments)

# use do_train_dict if possible.
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
    explicit_average_grad=False,
    no_update=False,
    use_amp=False,
):
    logging.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    log_start = time.time()

    visualize_input = False
    fix_input = False

    #from qd.layers.forward_pass_memory_checker import ForwardPassMemoryChecker
    #model = ForwardPassMemoryChecker(model)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    debug = False
    if debug:
        from qd.torch_common import init_random_seed
        init_random_seed(99)
        from qd.layers.forward_pass_feature_cache import ForwardPassFeatureCache
        model = ForwardPassFeatureCache(model)
    NaN_times = 0

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        if debug:
            from qd.torch_common import torch_load
            images = torch_load('/tmp/images.pt')
            targets = torch_load('/tmp/targets.pt')
            m = torch_load('/tmp/state_dict.pt')
            model.module.load_state_dict(m)

        if hasattr(images, 'image_sizes') and len(images.image_sizes) == 0:
            logging.error('this should never happen since different workers '
                    'will have different numbers of iterations.')
            continue

        if fix_input:
            logging.info('fix input')
            from qd.qd_common import run_if_not_memory_cached
            def get_x(x):
                return x
            images = run_if_not_memory_cached(get_x, images, __key='images')
            targets = run_if_not_memory_cached(get_x, targets, __key='targets')

        if visualize_input:
            from qd.qd_pytorch import visualize_maskrcnn_input
            logging.info(images.tensors.shape)
            visualize_maskrcnn_input(images, targets, show_box=True)

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        if isinstance(images, list):
            images = [x.to(device) for x in images]
        else:
            images = images.to(device)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device)
        else:
            targets = [target.to(device) for target in targets]

        if not no_update:
            optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            if losses != losses:
                if NaN_times > 100:
                    logging.info('NaN encountered!')
                    arguments['images'] = images
                    arguments['targets'] = targets
                    checkpointer.save("NaN_context_{}".format(get_mpi_rank()), **arguments)
                    raise RuntimeError('NaN encountered!')
                else:
                    NaN_times += 1
            else:
                NaN_times = 0
            scaler.scale(losses).backward()
            if not no_update:
                scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            if losses != losses:
                logging.info('NaN encountered!')
                arguments['images'] = images
                arguments['targets'] = targets
                checkpointer.save("NaN_context_{}".format(get_mpi_rank()), **arguments)
                raise RuntimeError('NaN encountered!')
            losses.backward()
            if not no_update:
                optimizer.step()
        meters.update(loss=losses, **loss_dict)

        if debug:
            model.sumarize_feature()
            import ipdb;ipdb.set_trace(context=15)

        #if explicit_average_grad:
            #average_gradients(model)

        start_opt_step = time.time()
        end_opt_step = time.time()

        if not no_update:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()

        if iteration > start_iter + 5:
            # we will skip the first few iterations since the time cost
            # evaluation for those are not good
            meters.update(time=batch_time, data=data_time)
            meters.update(opt_step_time=(end_opt_step-start_opt_step))

        if iteration % log_step == 0 or iteration == max_iter:
            speed = get_mpi_size() * log_step * len(targets) / (time.time() - log_start)
            if hasattr(meters, 'time'):
                eta_seconds = meters.time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                eta_string = 'Unknown'

            logging.info(
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
            #time_model = model
            #time_info = None
            #while True:
                #if hasattr(time_model, 'get_time_info'):
                    #time_info = time_model.get_time_info()
                    #break
                #if hasattr(time_model, 'module'):
                    #time_model = time_model.module
                    #continue
                #break
            #if time_info is not None:
                #logger.info('\n'.join((', '.join(('{} = {}'.format(k, m[k])for k in
                                       #['name', 'global_avg'])) for m in
                                      #time_info['meters'])))
            log_start = time.time()
        if iteration % checkpoint_period == 0:
            # with blobfuse, saving could fail with unknown reason. Instead of
            # saving and crashing, we do a best-effort manner.
            try_save_intermediate_snapshot(checkpointer, "model_iter_{:07d}".format(iteration), arguments)

    checkpointer.save("model_iter_{:07d}".format(arguments['iteration']), **arguments)
    checkpointer.save("model_final", **arguments)
    if get_mpi_rank() > 0:
        old_value = checkpointer.save_to_disk
        checkpointer.save_to_disk = True
        checkpointer.save("model_final_{}".format(get_mpi_rank()), **arguments)
        checkpointer.save_to_disk = old_value

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logging.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (1 if max_iter == 0 else
                                                   max_iter)
        )
    )

def get_num_image(d):
    if isinstance(d, dict):
        if 'image' in d:
            return len(d['image'])
        elif 'targets' in d:
            # mask-rcnn -> dict
            return len(d['targets'])
        elif 'input_ids' in d:
            # vl
            return len(d['input_ids'])
    elif isinstance(d, tuple) or isinstance(d, list):
        return get_num_image(d[0])
    elif isinstance(d, torch.Tensor):
        return len(d)
    else:
        raise NotImplementedError

def do_train_dict(
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
    explicit_average_grad=False,
    no_update=False,
    ema=None,
    use_amp=False,
    gradient_clip=None,
    model_sub_name_fn=None,
    async_loader=False,
    pt_swa=None,
    pt_swa_lr=None,
    pt_swa_update_step=None,
):
    from qd.qd_common import print_frame_info
    print_frame_info()
    if model_sub_name_fn is None:
        model_sub_name_fn = lambda i: "model_iter_{:07d}".format(i)
    logging.info("Start training")
    from qd.logger import SmoothedValue
    meters = MetricLogger(delimiter="  ", meter_creator=lambda:
                          SmoothedValue(log_step))
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    log_start = time.time()

    if use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if pt_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        max_iter = len(data_loader)
        anneal_epochs = int(0.05 * max_iter)
        swa_scheduler = SWALR(optimizer, swa_lr=pt_swa_lr, anneal_epochs=anneal_epochs)
        pt_swa_start = int(len(data_loader) * 0.75)
        assert pt_swa_lr is not None
        arguments['swa_model'] = swa_model.state_dict()
        if pt_swa_update_step is None:
            pt_swa_update_step = int(max(1, max_iter * 0.001))
        logging.info(f'pt_swa_update_step = {pt_swa_update_step}, anneal_epochs={anneal_epochs}')

    debug = False
    if debug:
        from qd.torch_common import init_random_seed
        torch.set_deterministic(False)
        init_random_seed(99)
        from qd.layers.forward_pass_time_checker import ForwardPassTimeChecker
        model = ForwardPassTimeChecker(model)

    #try_save_intermediate_snapshot(
        #checkpointer,
        #model_sub_name_fn(start_iter),
        #arguments)
    if async_loader:
        logging.info('wrapping with AsynchronousLoader')
        from qd.data_layer.loader import AsynchronousLoader
        data_loader = AsynchronousLoader(data_loader, device=device)

    for iteration, dict_data in enumerate(data_loader, start_iter):
        if iteration == start_iter:
            print('{}\n'.format(logging.getLogger().handlers))
            for k, v in dict_data.items():
                if isinstance(v, torch.Tensor):
                    # this is only used for debugging as no log is printed out
                    # print dtype to understand what parameter is fp16 or fp32
                    logging.info('{}: {}, {}'.format(
                        k, describe_tensor(v), v.dtype))
            for n, p in model.named_parameters():
                logging.info('{}: {}'.format(n, p.dtype))
        iteration = iteration + 1
        arguments["iteration"] = iteration

        before_to_device = time.time()
        data_time = before_to_device - end
        if not async_loader:
            dict_data = to(dict_data, device)
        before_gpu = time.time()
        time_to_device = before_gpu - before_to_device

        if not no_update:
            optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(dict_data)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            if gradient_clip:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                meters.update(total_norm=total_norm)
            if not no_update:
                scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(dict_data)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            if gradient_clip:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                meters.update(total_norm=total_norm)
            if not no_update:
                optimizer.step()

        if debug and iteration > 10:
            info = model.get_time_info()
            from qd.qd_common import write_to_yaml_file, create_vis_net_file
            write_to_yaml_file(info, '/tmp/a.yaml')
            create_vis_net_file('/tmp/a.yaml',
                            '/tmp/a.vis.txt')
            #model.sumarize_feature()
            import ipdb;ipdb.set_trace(context=15)

        if not use_amp and losses != losses:
            logging.info('NaN encountered!')
            checkpointer.save("NaN_context_{}".format(get_mpi_rank()), **arguments)
            raise RuntimeError('NaN encountered!')

        meters.update(loss=losses, **loss_dict)

        if not no_update:
            if pt_swa and iteration > pt_swa_start:
                if (iteration - pt_swa_start) % pt_swa_update_step == 0:
                    # update every 100 iteratins
                    swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
        time_gpu = time.time() - before_gpu

        if ema is not None:
            ema.update(model)

        if (iteration % log_step) == 0 or iteration == max_iter:
            speed = get_mpi_size() * log_step * get_num_image(dict_data) / (time.time() - log_start)
            if hasattr(meters, 'time'):
                eta_seconds = meters.time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                eta_string = 'Unknown'

            logging.info(
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
            before_save = time.time()
            try_save_intermediate_snapshot(
                checkpointer,
                model_sub_name_fn(iteration),
                arguments)
            meters.update(save_time=(time.time() - before_save))

        batch_time = time.time() - end
        end = time.time()

        if iteration > start_iter + 5:
            # we will skip the first few iterations since the time cost
            # evaluation for those are not good
            meters.update(time=batch_time, data=data_time)
            meters.update(to_device=time_to_device)
            meters.update(time_gpu=time_gpu)


    checkpointer.save("model_final", **arguments)
    if get_mpi_rank() > 0:
        old_value = checkpointer.save_to_disk
        checkpointer.save_to_disk = True
        checkpointer.save("model_final_{}".format(get_mpi_rank()), **arguments)
        checkpointer.save_to_disk = old_value

    checkpointer.save(model_sub_name_fn(arguments['iteration']), **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logging.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (1 if max_iter == 0 else
                                                   max_iter)
        )
    )

def recover_stdout_error():
    import sys
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

def do_train_by_deepspeed(
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
    explicit_average_grad=False,
    no_update=False,
    ema=None,
    use_amp=False,
    gradient_clip=None,
    model_sub_name_fn=None,
    zero_opt_stage=None,
    use_fp16=None,
    async_loader=False,
    no_flops_profiler=False,
    model_engine=None,
):
    recover_stdout_error()

    start_step = arguments['iteration']

    #client_sd = {}
    #_, client_sd = model_engine.load_checkpoint(
        #checkpointer.save_dir, args.ckpt_id)
    #start_step = client_sd['step'] + 1
    log_start = time.time()
    meters = MetricLogger(delimiter="  ", meter_creator=lambda:
                          SmoothedValue(log_step))
    max_iter = len(data_loader)
    end = time.time()

    debug = False
    #debug = True
    if debug:
        from qd.torch_common import init_random_seed
        torch.set_deterministic(False)
        init_random_seed(99)
        #from qd.layers.forward_pass_feature_cache import ForwardPassFeatureCache
        from qd.layers.forward_pass_time_checker import ForwardPassTimeChecker
        model = ForwardPassTimeChecker(model)

    effective_batch_size = get_mpi_size() * data_loader.batch_sampler.batch_size
    need_to_device = False
    if async_loader:
        logging.info('wrapping with AsynchronousLoader')
        from qd.data_layer.loader import AsynchronousLoader
        data_loader = AsynchronousLoader(data_loader, device=device)
    else:
        need_to_device = True
    recover_stdout_error()
    for h in logging.getLogger().handlers:
        h.setLevel(logging.INFO)
    print('logger handler: {}\n'.format(logging.getLogger().handlers))
    print('logger level = {}\n'.format(logging.getLogger().getEffectiveLevel()))
    for step, dict_data in enumerate(data_loader, start_step):
        #forward() method
        if step == start_step:
            recover_stdout_error()
            print('logger handler: {}\n'.format(logging.getLogger().handlers))
            for k, v in dict_data.items():
                if isinstance(v, torch.Tensor):
                    # this is only used for debugging as no log is printed out
                    logging.info('{}: {}'.format(k, describe_tensor(v)))
            for n, p in model.named_parameters():
                logging.info('{}: {}'.format(n, p.dtype))
        arguments['iteration'] = step + 1
        if need_to_device:
            dict_data = recursive_to_device(dict_data, device, non_blocking=True)
        data_time = time.time() - end
        loss_dict = model_engine(dict_data)
        loss = sum(loss_dict.values())
        meters.update(**loss_dict)

        #runs backpropagation
        model_engine.backward(loss)

        #weight update
        model_engine.step()

        if debug and step > 10:
            info = model.get_time_info()
            from qd.qd_common import write_to_yaml_file, create_vis_net_file
            write_to_yaml_file(info, '/tmp/a.yaml')
            create_vis_net_file('/tmp/a.yaml',
                            '/tmp/a.vis.txt')
            import ipdb;ipdb.set_trace(context=15)
        #if (step % checkpoint_period) == 0:
            #client_sd['step'] = step
            #model_engine.save_checkpoint(
                #checkpointer.save_dir, step, client_state=client_sd)
        if (step % log_step) == 0:
            speed =  log_step * effective_batch_size / (time.time() - log_start)
            if hasattr(meters, 'time'):
                eta_seconds = meters.time.global_avg * (max_iter - step)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                eta_string = 'Unknown'

            logging.info(
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
                    iter=step,
                    speed=speed,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            log_start = time.time()

        if (step + 1) % checkpoint_period == 0:
            # with blobfuse, saving could fail with unknown reason. Instead of
            # saving and crashing, we do a best-effort manner.
            before_save = time.time()
            try_save_intermediate_snapshot(
                checkpointer,
                model_sub_name_fn(step + 1),
                arguments)
            meters.update(save_time=(time.time() - before_save))

        batch_time = time.time() - end
        end = time.time()
        if step > start_step + 5:
            # we will skip the first few iterations since the time cost
            # evaluation for those are not good
            meters.update(time=batch_time, data=data_time)
        step += 1

    checkpointer.save(model_sub_name_fn(arguments['iteration']), **arguments)

