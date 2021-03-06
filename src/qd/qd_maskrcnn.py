import logging
import time
from torch import Tensor
import json
import copy
import shutil
from pprint import pformat
import torch
from qd.mask.config import cfg
from qd.mask.data import make_data_loader
from qd.mask.utils.checkpoint import DetectronCheckpointer
from qd.mask.utils.comm import synchronize, get_rank
from qd.mask.utils.comm import is_main_process
from qd.mask.structures.bounding_box import BoxList
from qd.mask.layers.batch_norm import FrozenBatchNorm2d
from qd.qd_pytorch import ModelPipeline
from qd.qd_common import ensure_directory
from qd.qd_common import try_delete
from qd.qd_common import set_if_not_exist
from qd.qd_common import load_from_yaml_file
from qd.qd_common import get_mpi_rank, get_mpi_size
from qd.qd_common import pass_key_value_if_has
from qd.qd_common import dump_to_yaml_str
from qd.qd_common import dict_has_path
from qd.qd_common import dict_get_path_value, dict_update_path_value
from qd.qd_common import write_to_yaml_file
from qd.tsv_io import TSVDataset
from qd.tsv_io import tsv_reader, tsv_writer
from qd.layers import ForwardPassTimeChecker
from qd.qd_pytorch import replace_module
import os.path as op
from tqdm import tqdm
from qd.qd_common import merge_dict_to_cfg
from qd.layers import ensure_shape_bn_layer
from torch.nn import BatchNorm2d
from qd.torch_common import update_bn_momentum
from qd.torch_common import boxlist_to_list_dict


def convert_to_sync_bn(module, norm=torch.nn.SyncBatchNorm, exclude_gn=False,
        is_pre_linear=False):
    module_output = module
    info = {'num_convert_bn': 0, 'num_convert_gn': 0}
    import maskrcnn_benchmark
    if isinstance(module,
            maskrcnn_benchmark.layers.batch_norm.FrozenBatchNorm2d) or \
        isinstance(module, torch.nn.BatchNorm2d):
        module_output = ensure_shape_bn_layer(norm, module.bias.size(),
                is_pre_linear)
        module_output.weight.data = module.weight.data.clone().detach()
        module_output.bias.data = module.bias.data.clone().detach()
        assert module.eps == 1e-5
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        info['num_convert_bn'] += 1
    elif isinstance(module, torch.nn.GroupNorm) and not exclude_gn:
        # gn has no running mean and running var
        module_output = ensure_shape_bn_layer(norm, module.bias.size(),
                is_pre_linear)
        module_output.weight.data = module.weight.data.clone().detach()
        module_output.bias.data = module.bias.data.clone().detach()
        info['num_convert_gn'] += 1
    # if the previous is a linear layer, we will not convert it to BN
    is_pre_linear = False
    for name, child in module.named_children():
        child, child_info = convert_to_sync_bn(child, norm, exclude_gn, is_pre_linear)
        module_output.add_module(name, child)
        for k, v in child_info.items():
            info[k] += v
        is_pre_linear = isinstance(child, torch.nn.Linear)
    del module
    return module_output, info

def lock_except_classifier(model):
    ignore = 0
    for name, param in model.named_parameters():
        if name not in ['roi_heads.box.predictor.cls_score.weight',
                'roi_heads.box.predictor.cls_score.bias']:
            param.requires_grad = False
        else:
            ignore += 1
    assert ignore == 2

def sync_model(model):
    module_states = list(model.state_dict().values())
    if len(module_states) > 0:
        # copied from https://github.com/pytorch/pytorch/blob/168c0797c45c1a26b0612c8496fe4ad112aeabfd/torch/nn/parallel/distributed.py
        import torch.distributed as dist
        from torch.distributed.distributed_c10d import _get_default_group
        process_group = _get_default_group()
        dist._dist_broadcast_coalesced(process_group,
                module_states, 250 * 1024 * 1024, False)

def lock_model_param_up_to(model, lock_up_to):
    ignore = 0
    found = False
    for name, param in model.named_parameters():
        if not found:
            logging.info('lock {}'.format(name))
            param.requires_grad = False
            ignore += 1
        else:
            logging.info('not lock {}'.format(name))
        if name == lock_up_to:
            found = True
    assert found
    assert ignore > 0, 'some bug?'
    logging.info('fix {} params'.format(ignore))

def lock_batch_norm(model):
    num = 0
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train(False)
            num += 1
    logging.info('lock bn = {}'.format(num))

def train(cfg, model, local_rank, distributed, log_step=20, sync_bn=False,
        exclude_convert_gn=False,
        opt_cls_only=False, bn_momentum=0.1, init_model_only=True,
        use_apex_ddp=False,
        data_partition=1,
        use_ddp=True,
        lock_up_to=None,
        zero_num_tracked=None,
        lock_bn=False,
        precise_bn2=False
        ):
    logging.info('start to train')
    if precise_bn2:
        from qd.layers.precise_bn import create_precise_bn2
        model = replace_module(model,
                lambda m: isinstance(m, torch.nn.BatchNorm2d),
                lambda m: create_precise_bn2(m),
                )
    if bn_momentum != 0.1:
        update_bn_momentum(model, bn_momentum)
    if opt_cls_only:
        lock_except_classifier(model)
    if lock_up_to:
        lock_model_param_up_to(model, lock_up_to)
    if lock_bn:
        lock_batch_norm(model)
    logging.info(model)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    from maskrcnn_benchmark.solver import make_optimizer
    optimizer = make_optimizer(cfg, model)
    from qd.qd_common import is_hvd_initialized
    use_hvd = is_hvd_initialized()
    if use_hvd:
        import horovod.torch as hvd
        optimizer = hvd.DistributedOptimizer(optimizer,
                model.named_parameters())
    from maskrcnn_benchmark.solver import make_lr_scheduler
    scheduler = make_lr_scheduler(cfg, optimizer)

    if cfg.MODEL.DEVICE == 'cuda':
        if not use_hvd:
            from apex import amp
            # Initialize mixed-precision training
            use_mixed_precision = cfg.DTYPE == "float16"
            amp_opt_level = 'O1' if use_mixed_precision else 'O0'
            logging.info('start to amp init')
            model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
            logging.info('end amp init')
    else:
        assert cfg.MODEL.DEVICE == 'cpu'

    if distributed:
        if not use_hvd:
            if use_apex_ddp:
                from apex.parallel import DistributedDataParallel as DDP
                model = DDP(model)
            else:
                if use_ddp:
                    logging.info('using ddp for parallel')
                    from torch.nn.parallel import DistributedDataParallel as DDP
                    model = DDP(
                        model, device_ids=[local_rank], output_device=local_rank,
                        # this should be removed if we update BatchNorm stats
                        broadcast_buffers=False,
                        find_unused_parameters=True,
                    )
                else:
                    logging.info('start to sync model')
                    # sync parameters among gpus
                    sync_model(model)
                    logging.info('finished to sync model')

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    # model_only should be True by default. That means, we only initialize the
    # weight and ignore teh parameters in optimizer and lr_scheduler. The
    # scenario is to finetune the model we trained before. If there is a
    # checkpoint in the current training, the model_only will be set back to
    # False. That means, it is compatible for continous training
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT,
            model_only=init_model_only)

    if zero_num_tracked:
        num = 0
        for n, p in model.named_buffers():
            if 'num_batches_tracked' in n:
                p.zero_()
                num += 1
        logging.info('zeroed num batch tracked = {}'.format(num))

    if use_hvd:
        import horovod.torch as hvd
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    from qd.opt.trainer import do_train

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        log_step=log_step,
        data_partition=data_partition,
        explicit_average_grad=not use_ddp,
        # only buffers are learned
        no_update=cfg.SOLVER.BASE_LR==0,
    )

    return model

class GeneralizedRCNNExtractModel(torch.nn.Module):
    def __init__(self, model, feature_name):
        super(GeneralizedRCNNExtractModel, self).__init__()
        self.feature_name = feature_name
        self.model = model
        self.score_thresh = 0.05

    def forward(self, x):
        if self.feature_name == 'RoiOutput':
            return self.forward_extract_roioutput(x)

    def forward_extract_roioutput(self, images):
        from maskrcnn_benchmark.structures.image_list import to_image_list
        images = to_image_list(images)
        features = self.model.backbone(images.tensors)
        proposals, _ = self.model.rpn(images, features, None)

        roi_head = self.model.roi_heads.box

        roi_extractor = roi_head.feature_extractor
        x = roi_extractor.pooler(features, proposals)
        x = x.view(x.size(0), -1)
        from torch.nn import functional as F
        x = F.relu(roi_extractor.fc6(x))
        roi_outputs = x
        x = F.relu(roi_extractor.fc7(x))

        # final classifier that converts the features into predictions
        class_logits, box_regression = roi_head.predictor(x)

        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        boxes = proposals
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        proposals = roi_head.post_processor.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        roi_outputs = roi_outputs.split(boxes_per_image, dim=0)

        results = []
        for prob, boxes_per_img, image_shape, roi_output in zip(
            class_prob, proposals, image_shapes, roi_outputs
        ):
            boxlist = roi_head.post_processor.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist.add_field('roi_output', roi_output)
            boxlist = self.per_class_extend(boxlist, num_classes)
            results.append(boxlist)
        return results

    def per_class_extend(self, boxlist, num_classes):
        '''
        boxlist: one box with one score vector. Now we need to expand it as
        multiple boxes, each of which is for one category
        '''
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        roi_output = boxlist.get_field('roi_output')

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class.add_field('roi_output', roi_output[inds, :])
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
        result = cat_boxlist(result)
        return result

def make_extract_model(model, feature_name):
    import maskrcnn_benchmark
    if type(model) == maskrcnn_benchmark.modeling.detector.generalized_rcnn.GeneralizedRCNN:
        model = GeneralizedRCNNExtractModel(model, feature_name)
    else:
        raise ValueError('unknown model = {}'.format(type(model)))
    return model

def extract_model_feature_iter(model, data_loader, device, feature_name):
    model.eval()
    cpu_device = torch.device("cpu")
    model = make_extract_model(model, feature_name)
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        for img_id, result in zip(image_ids, output):
            yield img_id, result

def compute_on_dataset_iter(model, data_loader, device, test_max_iter=None):
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        if test_max_iter is not None and i >= test_max_iter:
            # this is used for speed test, where we only would like to run a
            # few images
            break
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        for img_id, result in zip(image_ids, output):
            yield img_id, result

def list_dict_to_boxlist(rects, hw, label_to_id):
    extra_fields = set()
    for r in rects:
        extra_fields.update(r.keys())
    if 'rect' in extra_fields:
        # if len(rects) == 0, there is no such field here
        extra_fields.remove('rect')

    boxes = [r['rect'] for r in rects]
    extra_values = [[r.get(e) for r in rects] for e in extra_fields]
    boxes = Tensor(boxes)
    if len(boxes) == 0:
        boxes = torch.full((0, 4), 0)
    result = BoxList(boxes, image_size=(hw[1], hw[0]), mode='xyxy')
    for i, k in enumerate(extra_fields):
        v = extra_values[i]
        if k == 'class':
            v = list(map(lambda x: label_to_id[x], v))
            k = 'labels'
        elif k == 'conf':
            k = 'scores'
        result.add_field(k, Tensor(v))

    return result

def boxlist_to_rects(box_list, func_label_id_to_label):
    box_list = box_list.convert("xyxy")
    if len(box_list) == 0:
        return []
    box_key_to_dict_key = dict(((k, k) for k in box_list.extra_fields))
    if 'scores' in box_key_to_dict_key:
        box_key_to_dict_key['scores'] = 'conf'
    if 'labels' in box_key_to_dict_key:
        del box_key_to_dict_key['labels']
    boxes = box_list.bbox
    rects = []
    for i, box in enumerate(boxes):
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

        r = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
        rect = {'rect': r}
        if box_list.has_field('labels'):
            label_idx = box_list.get_field('labels')[i]
            rect['class'] = func_label_id_to_label(label_idx)
        for box_key, dict_key in box_key_to_dict_key.items():
            box_value = box_list.get_field(box_key)[i]
            if isinstance(box_value, Tensor):
                box_value = box_value.squeeze()
                if len(box_value.shape) == 1:
                    # just to make it a list of float
                    rect[dict_key] = list(map(float, box_value))
                elif len(box_value.shape) == 0:
                    rect[dict_key] = float(box_value)
                else:
                    raise ValueError('unknown Tensor {}'.format(
                        ','.join(map(str, box_value.shape))))
            else:
                raise ValueError('unknown {}'.format(type(box_value)))
        rects.append(rect)
    return rects

def only_inference_to_tsv(
        model,
        data_loader,
        cache_file,
        label_id_to_label,
        device="cuda",
        **kwargs
):
    device = torch.device(device)
    predict_threshold = kwargs.get('predict_threshold')

    if op.isfile(cache_file) and not kwargs.get('force_predict'):
        logging.info('ignore to run predict')
        return
    else:
        if kwargs.get('test_mergebn'):
            from qd.layers import MergeBatchNorm
            model = MergeBatchNorm(model)
        model = ForwardPassTimeChecker(model)
        if kwargs.get('predict_extract'):
            predictions = extract_model_feature_iter(model, data_loader, device,
                    kwargs['predict_extract'])
        else:
            predictions = compute_on_dataset_iter(model, data_loader, device,
                    test_max_iter=kwargs.get('test_max_iter'))
        ds = data_loader.dataset
        from maskrcnn_benchmark.utils.metric_logger import MetricLogger
        meters = MetricLogger(delimiter="  ")
        def gen_rows():
            start = time.time()
            for i, (idx, box_list) in enumerate(predictions):
                if i > 1:
                    meters.update(predict_time=time.time() - start)
                    start = time.time()
                key = ds.id_to_img_map[idx]
                wh_info = ds.get_img_info(idx)
                box_list = box_list.resize((wh_info['width'], wh_info['height']))
                if i > 1:
                    meters.update(finalize_boxlist=time.time() - start)
                    start = time.time()
                rects = boxlist_to_list_dict(box_list, label_id_to_label)
                if i > 1:
                    meters.update(convert_to_list_dict=time.time() - start)
                    start = time.time()
                if predict_threshold is not None:
                    rects = [r for r in rects if r['conf'] >= predict_threshold]
                str_rects = json.dumps(rects)
                if i > 1:
                    meters.update(json_encode=time.time() - start)
                    start = time.time()
                yield key, str_rects
                if i > 1:
                    meters.update(write=time.time() - start)
                start = time.time()
        start = time.time()
        tsv_writer(gen_rows(), cache_file)
        logging.info('total_time = {}'.format(time.time() - start))
        loss_str = []
        for name, meter in meters.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.total, meter.median)
            )
        logging.info(';'.join(loss_str))
        logging.info(str(model.meters))
        # note, we will not delete this cache file. since it is small, it
        # should be ok
        speed_yaml = cache_file + '.speed.yaml'
        write_to_yaml_file(model.get_time_info(), speed_yaml)
        from qd.qd_common import create_vis_net_file
        create_vis_net_file(speed_yaml,
                op.splitext(speed_yaml)[0] + '.vis.txt')

def test(cfg, model, distributed, predict_files, label_id_to_label,
        **kwargs):
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    assert len(data_loaders_val) == len(predict_files)
    for dataset_name, data_loader_val, predict_file in zip(dataset_names, data_loaders_val, predict_files):
        ds = data_loader_val.dataset
        from maskrcnn_benchmark.data.datasets.coco import COCODataset
        from maskrcnn_benchmark.data.datasets import MaskTSVDataset
        if type(ds) == COCODataset:
            con_to_json_id = ds.contiguous_category_id_to_json_id
            logging.info('overwrite the label id to label')
            label_id_to_label = {i: ds.coco.cats[con_to_json_id[i]]['name']
                for i in con_to_json_id}
        else:
            assert type(ds) == MaskTSVDataset

        mpi_size = get_mpi_size()
        if mpi_size == 1:
            cache_file = predict_file
        else:
            cache_file = '{}_{}_{}.tsv'.format(predict_file,
                    get_mpi_rank(),
                    mpi_size)

        only_inference_to_tsv(
            model,
            data_loader_val,
            cache_file,
            label_id_to_label,
            **kwargs
        )
        synchronize()

        if is_main_process() and mpi_size > 1:
            from qd.process_tsv import concat_tsv_files
            cache_files = ['{}_{}_{}.tsv'.format(predict_file, i, mpi_size)
                for i in range(mpi_size)]
            concat_tsv_files(cache_files, predict_file)
            from qd.process_tsv import delete_tsv_files
            delete_tsv_files(cache_files)
            # during prediction, we also computed the time cost. Here we
            # merge the time cost
            speed_cache_files = [c + '.speed.yaml' for c in cache_files]
            speed_yaml = predict_file + '.speed.yaml'
            from qd.qd_common import merge_speed_info
            merge_speed_info(speed_cache_files, speed_yaml)
            for x in speed_cache_files:
                try_delete(x)
            vis_files = [op.splitext(c)[0] + '.vis.txt' for c in speed_cache_files]
            from qd.qd_common import merge_speed_vis
            merge_speed_vis(vis_files,
                    op.splitext(speed_yaml)[0] + '.vis.txt')
            for x in vis_files:
                try_delete(x)
        if is_main_process() and 'test_max_iter' not in kwargs:
            # in distributed testing, some images might be predicted by
            # more than one worker since the distributed sampler only
            # garrantee each image will be processed at least once, not
            # exactly once. Thus, we have to remove the duplicate
            # predictions.
            ordered_keys = data_loader_val.dataset.get_keys()
            from qd.tsv_io import reorder_tsv_keys
            reorder_tsv_keys(predict_file, ordered_keys, predict_file)

        synchronize()

def upgrade_maskrcnn_param(kwargs):
    old_key = 'MODEL$BACKBONE$OUT_CHANNELS'
    if dict_has_path(kwargs, old_key):
        origin = dict_get_path_value(kwargs, old_key)
        new_key = 'MODEL$RESNETS$BACKBONE_OUT_CHANNELS'
        if not dict_has_path(kwargs, new_key):
            dict_update_path_value(kwargs, new_key,
                    origin)
        else:
            assert dict_get_path_value(kwargs, new_key)== origin
        from qd.qd_common import dict_remove_path
        dict_remove_path(kwargs, old_key)

class MaskRCNNPipeline(ModelPipeline):
    def __init__(self, **kwargs):
        super(MaskRCNNPipeline, self).__init__(**kwargs)

        from qd.mask.config import _default_cfg
        # the cfg could be modified by other places and we have to do this
        # trick to restore to its default. Gradually, we can remove the
        # dependence on this global variable of cfg.
        cfg.merge_from_other_cfg(_default_cfg)


        self._default.update({
            'num_workers': 4,
            'log_step': 20,
            'apply_nms_gt': True,
            'apply_nms_det': True,
            'expand_label_det': True,
            'bn_momentum': 0.1,
            # for the inital model, whether we want to load the lr scheduler and
            # optimer
            'init_model_only': True,
            'sync_bn': False,
            'exclude_convert_gn': False,
            'data_partition': 1,
            'use_ddp': True,
            'min_size_range32': (800, 800),
            'dap_max_grid': 3,
            })

        #maskrcnn_root = op.join('src', 'maskrcnn-benchmark')
        maskrcnn_root = op.join('aux_data')
        if self.net is not None and self.net != 'Unknown':
            if self.net.startswith('retinanet'):
                cfg_file = op.join(maskrcnn_root, 'FCOS_configs', 'retinanet',
                        self.net + '.yaml')
            elif 'dconv_' in self.net:
                cfg_file = op.join(maskrcnn_root, 'FCOS_configs', 'dcn',
                        self.net + '.yaml')
            elif self.net.endswith('_gn'):
                cfg_file = op.join(maskrcnn_root, 'FCOS_configs', 'gn_baselines',
                        self.net + '.yaml')
            else:
                cfg_file = op.join(maskrcnn_root, 'FCOS_configs', self.net + '.yaml')
            param = load_from_yaml_file(cfg_file)
        else:
            param = {}
        self.default_net_param = copy.deepcopy(param)
        from qd.qd_common import dict_update_nested_dict
        dict_update_nested_dict(self.kwargs, param, overwrite=False)

        set_if_not_exist(self.kwargs, 'SOLVER', {})
        set_if_not_exist(self.kwargs, 'DATASETS', {})
        set_if_not_exist(self.kwargs, 'TEST', {})
        set_if_not_exist(self.kwargs, 'DATALOADER', {})
        set_if_not_exist(self.kwargs, 'MODEL', {})
        if 'DEVICE' in self.kwargs['MODEL'] and 'device' not in self.kwargs:
            self.kwargs['device'] = self.kwargs['MODEL']['DEVICE']
        self.kwargs['MODEL']['DEVICE'] = self.device
        set_if_not_exist(self.kwargs['MODEL'], 'ROI_BOX_HEAD', {})

        self.kwargs['SOLVER']['IMS_PER_BATCH'] = int(self.effective_batch_size)
        self.kwargs['SOLVER']['MAX_ITER'] = self.parse_iter(self.max_iter)
        train_arg = {'data': self.data,
                'split': 'train',
                'bgr2rgb': self.bgr2rgb}
        logging.info('bgr2rgb = {}; Should be true unless on purpose'.format(self.bgr2rgb))
        if self.MaskTSVDataset is not None:
            train_arg.update(self.MaskTSVDataset)
            assert 'bgr2rgb' not in self.MaskTSVDataset or \
                    self.MaskTSVDataset['bgr2rgb'] == self.bgr2rgb
        self.kwargs['DATASETS']['TRAIN'] = ('${}'.format(
            dump_to_yaml_str(train_arg)),)
        test_arg = {'data': self.test_data,
                    'split': self.test_split,
                    'version': self.test_version,
                    'remove_images_without_annotations': False,
                    'bgr2rgb': self.bgr2rgb,
                    }
        self.kwargs['DATASETS']['TEST'] = ('${}'.format(
            dump_to_yaml_str(test_arg)),)
        self.kwargs['OUTPUT_DIR'] = op.join('output', self.full_expid, 'snapshot')
        # the test_batch_size is the size for each rank. We call the mask-rcnn
        # API, which should be the batch size for all rank
        self.kwargs['TEST']['IMS_PER_BATCH'] = self.test_batch_size * self.mpi_size
        self.kwargs['DATALOADER']['NUM_WORKERS'] = self.num_workers
        self.labelmap = TSVDataset(self.data).load_labelmap()
        if not self.has_background_output():
            self.kwargs['MODEL']['ROI_BOX_HEAD']['NUM_CLASSES'] = len(self.labelmap)
            assert train_arg.get('multi_hot_label', True)
        else:
            self.kwargs['MODEL']['ROI_BOX_HEAD']['NUM_CLASSES'] = len(self.labelmap) + 1

        set_if_not_exist(self.kwargs['MODEL'], 'RETINANET', {})
        self.kwargs['MODEL']['RETINANET']['NUM_CLASSES'] = len(self.labelmap) + 1

        if self.stageiter:
            self.kwargs['SOLVER']['STEPS'] = tuple([self.parse_iter(i) for i in self.stageiter])
        else:
            self.kwargs['SOLVER']['STEPS'] = (6*self.kwargs['SOLVER']['MAX_ITER']//9,
                    8*self.kwargs['SOLVER']['MAX_ITER']//9)
        pass_key_value_if_has(self.kwargs, 'base_lr',
                self.kwargs['SOLVER'], 'BASE_LR')
        pass_key_value_if_has(self.kwargs, 'basemodel',
                self.kwargs['MODEL'], 'WEIGHT')

        min_size_train = tuple(range(self.min_size_range32[0], self.min_size_range32[1] + 32, 32))
        if self.affine_resize:
            if self.affine_resize == 'AF':
                # AF: affine
                info = {'from': 'qd.qd_pytorch',
                        'import': 'DictTransformAffineResize',
                        'param': {'out_sizes': min_size_train}}
            elif self.affine_resize == 'RC':
                # RC: resize and crop
                info = {'from': 'qd.qd_pytorch',
                        'import': 'DictTransformResizeCrop',
                        'param': {'all_crop_size': min_size_train}}
            elif self.affine_resize == 'DAP':
                info = {'from': 'qd.qd_pytorch',
                        'import': 'DictDAPlacing',
                        'param': {'all_crop_size': min_size_train,
                                  'max_grid': self.dap_max_grid}}
            else:
                raise NotImplementedError(self.affine_resize)
            dict_update_path_value(self.kwargs, 'INPUT$TRAIN_RESIZER',
                    dump_to_yaml_str(info))
        else:
            dict_update_path_value(self.kwargs, 'INPUT$MIN_SIZE_TRAIN',
                    min_size_train)

        if self.first_feature_anchor_size:
            strides = dict_get_path_value(self.kwargs, 'MODEL$RPN$ANCHOR_STRIDE')
            if isinstance(strides, str):
                strides = eval(strides)
            assert isinstance(strides, tuple)
            num = len(strides)
            dict_update_path_value(self.kwargs, 'MODEL$RPN$ANCHOR_SIZES',
                    tuple(self.first_feature_anchor_size * 2 ** i
                        for i in range(num)))

        if self.with_dcn:
            self.kwargs['MODEL']['RESNETS']['STAGE_WITH_DCN'] = (False, True,
                    True, True)

        if self.all_color_jitter:
            dict_update_path_value(self.kwargs, 'INPUT$BRIGHTNESS',
                    self.all_color_jitter)
            dict_update_path_value(self.kwargs, 'INPUT$CONTRAST',
                    self.all_color_jitter)
            dict_update_path_value(self.kwargs, 'INPUT$SATURATION',
                    self.all_color_jitter)
            dict_update_path_value(self.kwargs, 'INPUT$HUE',
                    self.all_color_jitter / 2)

        if self.ssfpn_weight_scaled is not None or \
                self.ssfpn_weight_ss is not None or \
                self.ssfpn_weight_original is not None or\
                self.ssfpn_scale_factor is not None or \
                self.ssfpn_detach_larger is not None:
            arch = dict_get_path_value(self.kwargs, 'MODEL$META_ARCHITECTURE')
            from qd.qd_common import load_from_yaml_str
            arch = load_from_yaml_str(arch)
            if self.ssfpn_weight_scaled is not None:
                arch['param']['weight_scaled'] = self.ssfpn_weight_scaled
            if self.ssfpn_weight_ss is not None:
                arch['param']['weight_ss'] = self.ssfpn_weight_ss
            if self.ssfpn_weight_original is not None:
                arch['param']['weight_original'] = self.ssfpn_weight_original
            if self.ssfpn_scale_factor is not None:
                arch['param']['scale_factor'] = self.ssfpn_scale_factor
            if self.ssfpn_detach_larger is not None:
                arch['param']['detach_larger'] = self.ssfpn_detach_larger
            if self.ssfpn_extra_conv:
                # if it is not 256, it will crash. Then, we should fix it.
                # Here, let's just use the defualt 256 which works in most of
                # the cases
                arch['param']['extra_conv_cfg'] = {'num': 4,
                        'in_channels': 256, 'out_channels': 256}
            dict_update_path_value(self.kwargs, 'MODEL$META_ARCHITECTURE',
                dump_to_yaml_str(arch))

        # use self.kwargs instead  of kwargs because we might load parameters
        # from local disk not from the input argument
        upgrade_maskrcnn_param(self.kwargs)
        merge_dict_to_cfg(self.kwargs, cfg)

        # train -> iter
        # next time, we dont need to parse it again
        self.kwargs['max_iter'] = cfg.SOLVER.MAX_ITER

        # evaluation
        self._default['ovthresh'] = [-1, 0.3, 0.5]

        logging.info('cfg = \n{}'.format(cfg))

    def train(self):
        ensure_directory(cfg.OUTPUT_DIR)

        self.train_by_maskrcnn()

        model_final = op.join(self.output_folder, 'snapshot', 'model_final.pth')
        last_iter = self._get_checkpoint_file(iteration=self.max_iter)
        if self.mpi_rank == 0:
            if not op.isfile(last_iter):
                shutil.copy(model_final, last_iter)
        synchronize()
        return last_iter

    def train_by_maskrcnn(self):
        import os
        # mask_rcnn's downloading function has race issue: if the model folder
        # does not exist, all workers will mkdir(), which will crash. We do not
        # want to make the code change in maskrcnnn for the ease of code merge.
        torch_home = os.path.expanduser(os.getenv("TORCH_HOME", "~/.torch"))
        model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch_home, "models"))
        ensure_directory(model_dir)
        if self.data_partition > 1:
            if self.use_ddp:
                logging.info('set use_ddp=False sine data_partition > 1')
            self.use_ddp = False
        model = self.build_detection_model(training=True)
        model = self.model_surgery(model)
        model = ForwardPassTimeChecker(model)
        train(cfg, model, self.mpi_local_rank, self.distributed, self.log_step,
                sync_bn=self.sync_bn,
                exclude_convert_gn=self.exclude_convert_gn,
                opt_cls_only=self.opt_cls_only,
                bn_momentum=self.bn_momentum,
                init_model_only=self.init_model_only,
                use_apex_ddp=self.use_apex_ddp,
                data_partition=self.data_partition,
                use_ddp=self.use_ddp,
                lock_up_to=self.lock_up_to,
                zero_num_tracked=self.zero_num_tracked,
                lock_bn=self.lock_bn,
                precise_bn2=self.precise_bn2,
                )

    def append_predict_param(self, cc):
        super().append_predict_param(cc)
        default_post_nms_top_n_test = (dict_get_path_value(self.default_net_param,
            'MODEL$RPN$POST_NMS_TOP_N_TEST') if dict_has_path(self.default_net_param,
                'MODEL$RPN$POST_NMS_TOP_N_TEST') else 1000)
        if cfg.MODEL.RPN.POST_NMS_TOP_N_TEST != default_post_nms_top_n_test:
            cc.append('rpPost{}'.format(cfg.MODEL.RPN.POST_NMS_TOP_N_TEST))
        default_fpn_post_nms_top_n_test = (dict_get_path_value(self.default_net_param,
            'MODEL$RPN$FPN_POST_NMS_TOP_N_TEST') if dict_has_path(self.default_net_param,
                'MODEL$RPN$FPN_POST_NMS_TOP_N_TEST') else 2000)
        if cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST != default_fpn_post_nms_top_n_test:
            cc.append('FPNPostNMSTop{}'.format(cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST))
        default_pre_nms_top_n_test = (dict_get_path_value(self.default_net_param,
            'MODEL$RPN$PRE_NMS_TOP_N_TEST') if dict_has_path(self.default_net_param,
                'MODEL$RPN$PRE_NMS_TOP_N_TEST') else 6000)
        if cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST != default_pre_nms_top_n_test:
            cc.append('rpPre{}'.format(cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST))
        if cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_ACTIVATE != 'softmax':
            cc.append('{}'.format(cfg.MODEL.ROI_BOX_HEAD.CLASSIFICATION_ACTIVATE))
        if self.predict_threshold is not None:
            cc.append('th{}'.format(self.predict_threshold))
        if cfg.INPUT.MIN_SIZE_TEST != 800:
            cc.append('testInputSize{}'.format(cfg.INPUT.MIN_SIZE_TEST))
        if cfg.INPUT.MAX_SIZE_TEST != 1333:
            cc.append('MaxIn{}'.format(cfg.INPUT.MAX_SIZE_TEST))
        if cfg.MODEL.ROI_HEADS.NMS != 0.5:
            cc.append('roidNMS{}'.format(cfg.MODEL.ROI_HEADS.NMS))
        if cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG != 100:
            cc.append('roidDets{}'.format(cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG))
        if cfg.TEST.DETECTIONS_PER_IMG != 100:
            cc.append('upto{}'.format(cfg.TEST.DETECTIONS_PER_IMG))
        if cfg.MODEL.ROI_HEADS.SCORE_THRESH != 0.05:
            cc.append('roidth{}'.format(cfg.MODEL.ROI_HEADS.SCORE_THRESH))

        if cfg.MODEL.RETINANET.INFERENCE_TH != 0.05:
            cc.append('retinath{}'.format(cfg.MODEL.RETINANET.INFERENCE_TH))
        if cfg.MODEL.RETINANET.PRE_NMS_TOP_N != 1000:
            cc.append('retprenms{}'.format(cfg.MODEL.RETINANET.PRE_NMS_TOP_N))

        if cfg.MODEL.RPN.NMS_POLICY.TYPE == 'nms':
            if cfg.MODEL.RPN.NMS_THRESH != cfg.MODEL.RPN.NMS_POLICY.THRESH:
                cfg.MODEL.RPN.NMS_POLICY.THRESH = cfg.MODEL.RPN.NMS_THRESH
        if cfg.MODEL.RPN.NMS_POLICY.TYPE != 'nms':
            cc.append(cfg.MODEL.RPN.NMS_POLICY.TYPE)
        if cfg.MODEL.RPN.NMS_POLICY.THRESH != 0.7:
            cc.append('rpnnmspolicy{}'.format(cfg.MODEL.RPN.NMS_POLICY.THRESH))
        if cfg.MODEL.RPN.NMS_POLICY.ALPHA != 0.5:
            cc.append('rpA{}'.format(cfg.MODEL.RPN.NMS_POLICY.ALPHA))
        if cfg.MODEL.RPN.NMS_POLICY.GAMMA != 0.5:
            cc.append('rpG{}'.format(cfg.MODEL.RPN.NMS_POLICY.GAMMA))
        if cfg.MODEL.RPN.NMS_POLICY.NUM != 2:
            cc.append('rpN{}'.format(cfg.MODEL.RPN.NMS_POLICY.NUM))

        if cfg.MODEL.RPN.NMS_POLICY.ALPHA2 != 0.1:
            cc.append('rpA2{}'.format(cfg.MODEL.RPN.NMS_POLICY.ALPHA2))
        if cfg.MODEL.RPN.NMS_POLICY.GAMMA2 != 0.1:
            cc.append('rpG2{}'.format(cfg.MODEL.RPN.NMS_POLICY.GAMMA2))
        if cfg.MODEL.RPN.NMS_POLICY.NUM2 != 1:
            cc.append('rpN2{}'.format(cfg.MODEL.RPN.NMS_POLICY.NUM2))
        if cfg.MODEL.RPN.NMS_POLICY.COMPOSE_FINAL_RERANK:
            cc.append('rpnnmsRerank')

        if cfg.MODEL.ROI_HEADS.NMS_POLICY.TYPE != 'nms':
            cc.append(cfg.MODEL.ROI_HEADS.NMS_POLICY.TYPE)
        if cfg.MODEL.ROI_HEADS.NMS_POLICY.THRESH != 0.5:
            cc.append('roinmspolicy{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.THRESH))

        if cfg.MODEL.ROI_HEADS.NMS_POLICY.ALPHA != 0.5:
            cc.append('roinmsAlpha{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.ALPHA))
        if cfg.MODEL.ROI_HEADS.NMS_POLICY.GAMMA != 0.5:
            cc.append('roinmsGamma{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.GAMMA))
        if cfg.MODEL.ROI_HEADS.NMS_POLICY.NUM != 2:
            cc.append('roinmsNum{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.NUM))

        if cfg.MODEL.ROI_HEADS.NMS_POLICY.ALPHA2 != 0.1:
            cc.append('roinmsAlpha2{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.ALPHA2))
        if cfg.MODEL.ROI_HEADS.NMS_POLICY.GAMMA2 != 0.1:
            cc.append('roinmsGamma2{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.GAMMA2))
        if cfg.MODEL.ROI_HEADS.NMS_POLICY.NUM2 != 1:
            cc.append('roinmsNum2{}'.format(cfg.MODEL.ROI_HEADS.NMS_POLICY.NUM2))
        if cfg.MODEL.ROI_HEADS.NMS_POLICY.COMPOSE_FINAL_RERANK:
            cc.append('roinmsRerank')

        if cfg.MODEL.RETINANET.NMS_POLICY.TYPE != 'nms':
            cc.append(cfg.MODEL.RETINANET.NMS_POLICY.TYPE)
        if cfg.MODEL.RETINANET.NMS_POLICY.THRESH != 0.4:
            cc.append('rnpolicy{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.THRESH))

        if cfg.MODEL.RETINANET.NMS_POLICY.ALPHA != 0.4:
            cc.append('rnAlpha{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.ALPHA))
        if cfg.MODEL.RETINANET.NMS_POLICY.GAMMA != 0.4:
            cc.append('rnGamma{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.GAMMA))
        if cfg.MODEL.RETINANET.NMS_POLICY.NUM != 1:
            cc.append('rnNum{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.NUM))

        if cfg.MODEL.RETINANET.NMS_POLICY.ALPHA2 != 0.1:
            cc.append('rnAlpha2{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.ALPHA2))
        if cfg.MODEL.RETINANET.NMS_POLICY.GAMMA2 != 0.1:
            cc.append('rnGamma2{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.GAMMA2))
        if cfg.MODEL.RETINANET.NMS_POLICY.NUM2 != 0:
            cc.append('rnNum2{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.NUM2))
        if cfg.MODEL.RETINANET.NMS_POLICY.THRESH2 != 0.4:
            cc.append('rnTh2{}'.format(cfg.MODEL.RETINANET.NMS_POLICY.THRESH2))
        if cfg.MODEL.RETINANET.NMS_POLICY.COMPOSE_FINAL_RERANK:
            cc.append('rnRerank')
            if cfg.MODEL.RETINANET.NMS_POLICY.COMPOSE_FINAL_RERANK_TYPE != 'nms':
                cc.append(cfg.MODEL.RETINANET.NMS_POLICY.COMPOSE_FINAL_RERANK_TYPE)

        if cfg.MODEL.ROI_HEADS.NMS_ON_MAX_CONF_AGNOSTIC:
            cc.append('nmsMax')

        if self.bn_train_mode_test:
            cc.append('trainmode')

    def get_old_check_point_file(self, curr):
        return op.join(self.output_folder, 'snapshot',
                    'model_{7d}.pth'.format(curr))

    def get_snapshot_steps(self):
        return cfg.SOLVER.CHECKPOINT_PERIOD

    def _get_old_check_point_file(self, i):
        return op.join(self.output_folder, 'snapshot',
                'model_{:07d}.pth'.format(i))

    def has_background_output(self):
        # we should always use self.kwargs rather than cfg. cfg is only used by
        # mask-rcnn lib, we only set it instead of read
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

    def build_detection_model(self, training=True):
        from maskrcnn_benchmark.modeling.detector import build_detection_model
        model = build_detection_model(cfg)
        if training:
            # we might use eval model for BN layers after this
            model.train()
        if self.sync_bn:
            if get_mpi_size() == 1 or not training:
                # in prediction, we may not wrap the model with
                # DistributedDataParallel
                norm = torch.nn.BatchNorm2d
            else:
                # SyncBatchNorm only works in distributed env. We have not tested
                # the same code path if size == 1,e.g. initialize parallel group
                # also when size = 1
                if self.use_apex_ddp:
                    import apex
                    norm = apex.parallel.optimized_sync_batchnorm.SyncBatchNorm
                else:
                    norm = torch.nn.SyncBatchNorm
                # need to convert to BN
            model, convert_info = convert_to_sync_bn(model,
                    norm, self.exclude_convert_gn)
            logging.info(pformat(convert_info))
        if self.convert_bn_to_nsbn:
            from detectron2.layers.batch_norm import NaiveSyncBatchNorm
            norm = NaiveSyncBatchNorm
            replace_module(model,
                    lambda m: type(m) == FrozenBatchNorm2d,
                    lambda m: NaiveSyncBatchNorm(num_features=m.num_features,
                        eps=m.eps))
            replace_module(model,
                    lambda m: type(m) == BatchNorm2d,
                    lambda m: NaiveSyncBatchNorm(num_features=m.num_features,
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=m.affine,
                        track_running_stats=m.track_running_stats))
        if self.convert_gn:
            if self.convert_gn == 'GBN':
                from qd.layers.group_batch_norm import GroupBatchNorm, get_normalize_groups
                model = replace_module(model,
                        lambda m: isinstance(m, torch.nn.GroupNorm),
                        lambda m: GroupBatchNorm(get_normalize_groups(m.num_channels,
                            self.normalization_group,
                            self.normalization_group_size), m.num_channels))
            elif self.convert_gn == 'SGBN':
                from qd.layers.group_batch_norm import SyncGroupBatchNorm, get_normalize_groups
                model = replace_module(model,
                        lambda m: isinstance(m, torch.nn.GroupNorm),
                        lambda m: SyncGroupBatchNorm(get_normalize_groups(m.num_channels,
                            self.normalization_group,
                            self.normalization_group_size), m.num_channels))
            else:
                raise NotImplementedError
        if self.convert_fbn_to_bn:
            model = replace_module(model,
                    lambda m: type(m) == FrozenBatchNorm2d,
                    lambda m: BatchNorm2d(num_features=m.num_features,
                        eps=m.eps))
        return model

    def predict(self, model_file, predict_result_file):
        model = self.build_detection_model(training=False)
        #model = self._data_parallel_wrap(model)
        model.eval()
        model.to(cfg.MODEL.DEVICE)
        checkpointer = DetectronCheckpointer(cfg, model,
                save_dir=self.output_folder,
                )
        checkpointer.load(model_file)

        if self.bn_train_mode_test:
            for _, m in model.named_modules():
                if isinstance(m, torch.nn.SyncBatchNorm):
                    m.train()

        dataset = TSVDataset(self.data)
        labelmap = dataset.load_labelmap()
        extra = 1 if self.has_background_output() else 0
        label_id_to_label = {i + extra: l for i, l in enumerate(labelmap)}
        old_value = cfg.DATALOADER.ASPECT_RATIO_GROUPING
        cfg.DATALOADER.ASPECT_RATIO_GROUPING = False
        test(cfg, model, self.distributed, [predict_result_file],
                label_id_to_label, **self.kwargs)
        cfg.DATALOADER.ASPECT_RATIO_GROUPING = old_value


    def _get_load_model(self):
        if self.model is None:
            from maskrcnn_benchmark.modeling.detector import build_detection_model
            model = build_detection_model(cfg)
            if self.sync_bn:
                # need to convert to BN
                model, convert_info = convert_to_sync_bn(model,
                        torch.nn.BatchNorm2d, self.exclude_convert_gn)
                logging.info(pformat(convert_info))
            model.to(cfg.MODEL.DEVICE)
            checkpointer = DetectronCheckpointer(cfg, model,
                    save_dir=self.output_folder)
            model_file = self._get_checkpoint_file()
            checkpointer.load(model_file)
            self.model = model
        return self.model

    def demo(self, image_path):
        from qd.process_image import load_image
        cv_im = load_image(image_path)
        rects = self.predict_one(cv_im)
        from qd.process_image import draw_rects, show_image
        draw_rects(rects, cv_im, add_label=True)
        show_image(cv_im)

    def predict_one(self, cv_im):
        model = self._get_load_model()

        dataset = TSVDataset(self.data)
        labelmap = dataset.load_labelmap()
        extra = 1 if self.has_background_output() else 0
        label_id_to_label = {i + extra: l for i, l in enumerate(labelmap)}

        height, width = cv_im.shape[:2]
        if self.bgr2rgb:
            import cv2
            im = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB)
        else:
            im = cv_im

        import torchvision.transforms as transforms
        im = transforms.ToPILImage()(im)
        from maskrcnn_benchmark.data.transforms import build_transforms
        curr_transform = build_transforms(cfg, is_train=False)

        target = create_empty_boxlist(width, height, self.device)
        transform_result = curr_transform({'image': im, 'rects': target,
            'iteration': 0})
        im, target = transform_result['image'], transform_result['rects']

        from maskrcnn_benchmark.structures.image_list import to_image_list
        image_list = to_image_list(im, cfg.DATALOADER.SIZE_DIVISIBILITY)

        image_list = image_list.to(self.device)
        cpu_device = torch.device("cpu")
        model.eval()
        with torch.no_grad():
            output = model(image_list)
            output = [o.to(cpu_device) for o in output]
        box_list = output[0]
        box_list = box_list.resize((width, height))
        rects = boxlist_to_list_dict(box_list, label_id_to_label)

        threshold = self._get_predict_one_threshold()
        rects = [r for r in rects if
                r['conf'] >= threshold.get(r['class'], 0.2)]

        logging.info('result = \n{}'.format(pformat(rects)))
        return rects

    def _get_predict_one_threshold(self):
        if self.predict_one_threshold is None:
            threshold_file = op.join(self.output_folder, 'deploy', 'threshold.tsv')
            if op.isfile(threshold_file):
                result = {c: float(th) for c, th in tsv_reader(threshold_file)}
            else:
                result = {}
            self.predict_one_threshold = result
        return self.predict_one_threshold

def create_empty_boxlist(w, h, device):
    boxlist_empty = BoxList(torch.zeros((0,4)).to(device),
            (w, h),
            mode='xyxy')
    return boxlist_empty


