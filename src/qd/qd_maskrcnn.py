import logging
import time
import datetime
import pickle as pkl
from torch import Tensor
import json
import copy
import shutil
import numpy as np

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
import maskrcnn_benchmark
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import compute_on_dataset
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.comm import is_main_process
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.engine.trainer import do_train
from qd.qd_pytorch import ModelPipeline
from qd.qd_pytorch import torch_load, torch_save
from qd.qd_common import ensure_directory
from qd.qd_common import json_dump
from qd.qd_common import init_logging
from qd.qd_common import list_to_dict
from qd.qd_common import calculate_image_ap
from qd.qd_common import set_if_not_exist
from qd.qd_common import load_from_yaml_file
from qd.qd_common import calculate_iou
from qd.qd_common import get_mpi_rank, get_mpi_size
from qd.tsv_io import TSVDataset
from qd.tsv_io import tsv_reader, tsv_writer
from qd.process_tsv import TSVFile, convert_one_label
from qd.qd_pytorch import ModelPipeline
from yacs.config import CfgNode
import os.path as op
from tqdm import tqdm
from apex import amp

def merge_dict_to_cfg(dict_param, cfg):
    """merge the key, value pair in dict_param into cfg

    :dict_param: TODO
    :cfg: TODO
    :returns: TODO

    """
    def trim_dict(d, c):
        """remove all the keys in the dictionary of d based on the existance of
        cfg
        """
        to_remove = [k for k in d if k not in c]
        for k in to_remove:
            del d[k]
        to_check = [(k, d[k]) for k in d if d[k] is dict]
        for k, t in to_check:
            trim_dict(t, getattr(c, k))
    trimed_param = copy.deepcopy(dict_param)
    trim_dict(trimed_param, cfg)
    cfg.merge_from_other_cfg(CfgNode(trimed_param))

def train(cfg, local_rank, distributed):
    logging.info('start to train')
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    logging.info('start to amp init')
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    logging.info('end amp init')

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)

    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
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
            from maskrcnn_benchmark.structures.bounding_box import BoxList
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

def compute_on_dataset_iter(model, data_loader, device):
    model.eval()
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        for img_id, result in zip(image_ids, output):
            yield img_id, result

def boxlist_to_list_dict(box_list, label_id_to_label):
    box_list = box_list.convert("xyxy")
    if len(box_list) == 0:
        return []
    scores = box_list.get_field("scores").tolist()
    labels = box_list.get_field("labels").tolist()
    extra_key_values = [(k, v) for k, v in box_list.extra_fields.items()
            if k not in ['scores', 'labels']]
    boxes = box_list.bbox
    rects = []
    for i, (box, score, label_id) in enumerate(zip(boxes, scores, labels)):
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()

        r = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
        rect = {'class': label_id_to_label[label_id], 'conf': score, 'rect': r}

        for k, v in extra_key_values:
            f = v[i]
            if isinstance(f, Tensor):
                f = f.squeeze()
                if len(f.shape) == 1:
                    # just to make it a list of float
                    rect[k] = list(map(float, f))
                elif len(f.shape) == 0:
                    rect[k] = float(f)
                else:
                    raise ValueError('unknown Tensor {}'.format(
                        ','.join(map(str, f.shape))))
            else:
                raise ValueError('unknown {}'.format(type(f)))
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

    if op.isfile(cache_file) and not kwargs.get('force_predict'):
        logging.info('ignore to run predict')
        return
    else:
        if kwargs.get('predict_extract'):
            predictions = extract_model_feature_iter(model, data_loader, device,
                    kwargs['predict_extract'])
        else:
            predictions = compute_on_dataset_iter(model, data_loader, device)
        ds = data_loader.dataset
        def gen_rows():
            for idx, box_list in predictions:
                key = ds.id_to_img_map[idx]
                wh_info = ds.get_img_info(idx)
                box_list = box_list.resize((wh_info['width'],
                    wh_info['height']))
                rects = boxlist_to_list_dict(box_list, label_id_to_label)
                str_rects = json.dumps(rects)
                yield key, str_rects
        tsv_writer(gen_rows(), cache_file)

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
        #idx_to_boxlist = only_inference(
            #model,
            #data_loader_val,
            #predict_file,
            #device=cfg.MODEL.DEVICE,
            #**kwargs,
        #)
        #synchronize()

        #if is_main_process():
            #write_key_to_boxlist_to_tsv(idx_to_boxlist,
                    #predict_file, ds=data_loader_val.dataset,
                    #label_id_to_label=label_id_to_label)

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
            device=cfg.MODEL.DEVICE,
            **kwargs
        )
        synchronize()

        if is_main_process():
            if mpi_size > 1:
                from qd.process_tsv import concat_tsv_files
                cache_files = ['{}_{}_{}.tsv'.format(predict_file, i, mpi_size)
                    for i in range(mpi_size)]
                concat_tsv_files(cache_files, predict_file)
                from qd.process_tsv import delete_tsv_files
                delete_tsv_files(cache_files)
                # in distributed testing, some images might be predicted by
                # more than one worker since the distributed sampler only
                # garrantee each image will be processed at least once, not
                # exactly once. Thus, we have to remove the duplicate
                # predictions.
                ordered_keys = data_loader_val.dataset.get_keys()
                from qd.tsv_io import reorder_tsv_keys
                reorder_tsv_keys(predict_file, ordered_keys, predict_file)
        synchronize()


class MaskRCNNPipeline(ModelPipeline):
    def __init__(self, **kwargs):
        super(MaskRCNNPipeline, self).__init__(**kwargs)

        self._default.update({'workers': 4})

        maskrcnn_root = op.join('src', 'maskrcnn-benchmark')
        if self.net.startswith('retinanet'):
            cfg_file = op.join(maskrcnn_root, 'configs', 'retinanet',
                    self.net + '.yaml')
        else:
            cfg_file = op.join(maskrcnn_root, 'configs', self.net + '.yaml')
        param = load_from_yaml_file(cfg_file)
        self.kwargs.update(param)

        set_if_not_exist(self.kwargs, 'SOLVER', {})
        set_if_not_exist(self.kwargs, 'DATASETS', {})
        set_if_not_exist(self.kwargs, 'TEST', {})
        set_if_not_exist(self.kwargs, 'DATALOADER', {})
        set_if_not_exist(self.kwargs, 'MODEL', {})
        set_if_not_exist(self.kwargs['MODEL'], 'ROI_BOX_HEAD', {})

        self.kwargs['SOLVER']['IMS_PER_BATCH'] = int(self.effective_batch_size)
        self.kwargs['SOLVER']['MAX_ITER'] = self.parse_iter(self.max_iter)
        self.kwargs['DATASETS']['TRAIN'] = ('{}$train'.format(self.data),)
        self.kwargs['DATASETS']['TEST'] = ('{}${}'.format(self.test_data, self.test_split),)
        self.kwargs['OUTPUT_DIR'] = op.join('output', self.full_expid, 'snapshot')
        # the test_batch_size is the size for each rank. We call the mask-rcnn
        # API, which should be the batch size for all rank
        self.kwargs['TEST']['IMS_PER_BATCH'] = self.test_batch_size * self.mpi_size
        self.kwargs['DATALOADER']['NUM_WORKERS'] = self.workers
        self.kwargs['MODEL']['ROI_BOX_HEAD']['NUM_CLASSES'] = len(TSVDataset(self.data).load_labelmap()) + 1

        from qd.qd_common import pass_key_value_if_has

        pass_key_value_if_has(self.kwargs, 'stageiter',
                self.kwargs['SOLVER'], 'STEPS')
        pass_key_value_if_has(self.kwargs, 'base_lr',
                self.kwargs['SOLVER'], 'BASE_LR')


        # use self.kwargs instead  of kwargs because we might load parameters
        # from local disk not from the input argument
        merge_dict_to_cfg(self.kwargs, cfg)

        # train -> iter
        assert 'max_iter' not in self.kwargs or \
                self.kwargs['max_iter'] == cfg.SOLVER.MAX_ITER
        self.kwargs['max_iter'] = cfg.SOLVER.MAX_ITER

        # evaluation
        self._default['ovthresh'] = [-1, 0.3, 0.5]

        cfg.freeze()

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

        train(cfg, self.mpi_local_rank, self.distributed)

    def append_predict_param(self, cc):
        super(MaskRCNNPipeline, self).append_predict_param(cc)
        if cfg.MODEL.RPN.POST_NMS_TOP_N_TEST != 1000:
            cc.append('RPN.PostNMSTop{}'.format(cfg.MODEL.RPN.POST_NMS_TOP_N_TEST))

    def predict(self, model_file, predict_result_file):
        model = build_detection_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        checkpointer = DetectronCheckpointer(cfg, model,
                save_dir=self.output_folder)
        checkpointer.load(model_file)

        dataset = TSVDataset(self.data)
        labelmap = dataset.load_labelmap()
        # 0 is background
        label_id_to_label = {i + 1: l for i, l in enumerate(labelmap)}
        test(cfg, model, self.distributed, [predict_result_file],
                label_id_to_label, **self.kwargs)


