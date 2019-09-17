import copy
import torch
import os.path as op
import logging
import shutil
from qd.qd_pytorch import ModelPipeline, synchronize
from qd.qd_common import ensure_directory
from qd.qd_common import read_to_buffer
from qd.qd_common import json_dump
from qd.tsv_io import tsv_writer
from qd.tsv_io import TSVDataset
from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets import get_dataset
from mmdet.apis import train_detector
from mmdet.datasets.coco import CocoDataset
from mmcv.runner import load_checkpoint, get_dist_info
from mmdet.datasets import build_dataloader
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
import mmcv


def single_gpu_test(model, data_loader, labelmap, pred_file,
        show=False):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    def gen_rows():
        for i, data in enumerate(data_loader):
            key = data['img_meta'][0].data[0][0]['key']
            with torch.no_grad():
                result = model(return_loss=False, rescale=not show, **data)
            rects = convert_mmresult_to_rects(result, labelmap)
            yield key, json_dump(rects)

            if show:
                model.module.show_result(data, result, dataset.img_norm_cfg)

            batch_size = data['img'][0].size(0)
            for _ in range(batch_size):
                prog_bar.update()
    tsv_writer(gen_rows(), pred_file)

def multi_gpu_test(model, data_loader, labelmap, pred_file):
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    def gen_rows():
        for i, data in enumerate(data_loader):
            key = data['img_meta'][0].data[0][0]['key']
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            rects = convert_mmresult_to_rects(result, labelmap)
            yield key, json_dump(rects)

            if rank == 0:
                batch_size = data['img'][0].size(0)
                for _ in range(batch_size * world_size):
                    prog_bar.update()
    pred_file_rank = pred_file + '_{}_{}.tsv'.format(rank, world_size)
    tsv_writer(gen_rows(), pred_file_rank)
    synchronize()
    if rank == 0:
        if world_size > 1:
            from qd.process_tsv import concat_tsv_files
            cache_files = ['{}_{}_{}.tsv'.format(pred_file, i, world_size)
                for i in range(world_size)]
            concat_tsv_files(cache_files, pred_file)
            from qd.process_tsv import delete_tsv_files
            delete_tsv_files(cache_files)
            # in distributed testing, some images might be predicted by
            # more than one worker since the distributed sampler only
            # garrantee each image will be processed at least once, not
            # exactly once. Thus, we have to remove the duplicate
            # predictions.
            ordered_keys = data_loader.dataset.get_keys()
            from qd.tsv_io import reorder_tsv_keys
            reorder_tsv_keys(pred_file, ordered_keys, pred_file)
    synchronize()

def set_dict_or_list_of_dict(d, k, v):
    if isinstance(d, dict):
        from qd.qd_common import dict_update_path_value
        dict_update_path_value(d, k, v)
    elif isinstance(d, list):
        for x in d:
            set_dict_or_list_of_dict(x, k, v)
    else:
        raise NotImplementedError()

class MMDetPipeline(ModelPipeline):
    def __init__(self, **kwargs):
        super(MMDetPipeline, self).__init__(**kwargs)

        self._default.update({
            'workers': 4,
            'ignore_filter_img': False,
            'multi_hot_label': False,
            'classification_loss_type': 'CE',
            })

        mmdet_root = op.join('src', 'mmdetection')

        config_file = op.join(mmdet_root, 'configs', self.net + '.py')
        cfg = Config.fromfile(config_file)

        cfg.work_dir = op.join(self.output_folder, 'snapshot')

        del cfg._cfg_dict['data']['val']
        for split in ['train', 'test']:
            if split not in cfg._cfg_dict['data']:
                continue
            for key in ['ann_file', 'img_prefix']:
                if key in cfg._cfg_dict['data'][split]:
                    del cfg._cfg_dict['data'][split][key]

        cfg._cfg_dict['data']['train']['type'] = 'MMTSVDataset'
        cfg._cfg_dict['data']['train']['data'] = self.data
        cfg._cfg_dict['data']['train']['split'] = 'train'
        cfg._cfg_dict['data']['train']['version'] = self.train_version
        cfg._cfg_dict['data']['train']['ignore_filter_img'] = self.ignore_filter_img
        cfg._cfg_dict['data']['train']['multi_hot_label'] = self.multi_hot_label

        cfg._cfg_dict['data']['test']['type'] = 'MMTSVDataset'
        cfg._cfg_dict['data']['test']['data'] = self.test_data
        cfg._cfg_dict['data']['test']['split'] = self.test_split
        cfg._cfg_dict['data']['test']['version'] = self.test_version

        cfg._cfg_dict['data']['workers_per_gpu'] = self.workers
        cfg._cfg_dict['data']['imgs_per_gpu'] = self.batch_size
        cfg._cfg_dict['optimizer']['lr'] = self.base_lr

        cfg._cfg_dict['max_iter'] = self.parse_iter(self.max_iter)

        self.labelmap = TSVDataset(self.data).load_labelmap()
        # in the cascade scenario, the field of bbox_head is a list of dict;
        # while for the normal one, it is a dictionary
        if self.has_background_output():
            set_dict_or_list_of_dict(cfg._cfg_dict['model']['bbox_head'],
                    'num_classes', len(self.labelmap) + 1)
        else:
            set_dict_or_list_of_dict(cfg._cfg_dict['model']['bbox_head'],
                    'num_classes', len(self.labelmap))
        set_dict_or_list_of_dict(cfg._cfg_dict['model']['bbox_head'],
                'classification_loss_type', self.classification_loss_type)

        self.cfg = cfg

    def has_background_output(self):
        # we should always use self.kwargs rather than cfg. cfg is only used by
        # mask-rcnn lib, we only set it instead of read
        p = self.kwargs.get('classification_loss_type', 'CE')
        if p in ['BCE'] or \
                any(p.startswith(x) for x in ['IBCE']):
            return False
        elif p in ['CE', 'MCEB']:
            # 0 is background
            return True
        else:
            raise NotImplementedError()
    def train(self):
        ensure_directory(self.output_folder)
        ensure_directory(op.join(self.output_folder, 'snapshot'))

        self.train_by_mmdet()

        model_final = op.join(self.output_folder, 'snapshot',
                'epoch_{}.pth'.format(self.cfg._cfg_dict['total_epochs']))
        last_iter = self._get_checkpoint_file()
        if self.mpi_rank == 0:
            if not op.isfile(last_iter):
                shutil.copy(model_final, last_iter)
        synchronize()
        return last_iter

    def train_by_mmdet(self):
        cfg = self.cfg

        if type(self.max_iter) is str:
            assert self.max_iter.endswith('e')
            cfg._cfg_dict['total_epochs'] = int(self.max_iter[:-1])
        else:
            if self.num_train_images is None:
                self.num_train_images = TSVDataset(self.data).num_rows('train')
            cfg._cfg_dict['total_epochs'] = (cfg._cfg_dict['max_iter'] *
                self.effective_batch_size // self.num_train_images + 1)

        latest_file_ptr = op.join(self.output_folder, 'snapshot', 'latest.pth.tsv')
        if op.isfile(latest_file_ptr):
            cfg.resume_from = read_to_buffer(latest_file_ptr).decode()
        else:
            cfg.resume_from = None
        cfg.gpus = 1

        distributed = self.distributed

        model = build_detector(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        train_dataset = get_dataset(cfg.data.train)
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=torch.__version__,
                config=cfg.text,
                classes=train_dataset.CLASSES)
        # add an attribute for visualization convenience
        model.CLASSES = train_dataset.CLASSES
        train_detector(
            model,
            train_dataset,
            cfg,
            distributed=distributed,
            validate=None,
            logger=logging.getLogger())

    def predict(self, model_file, predict_result_file):
        cfg = copy.deepcopy(self.cfg)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True

        dataset = get_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset,
                                       imgs_per_gpu=1,
                                       workers_per_gpu=cfg.data.workers_per_gpu,
                                       dist=self.distributed,
                                       shuffle=False)

        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        dataset = TSVDataset(self.data)
        labelmap = dataset.load_labelmap()

        load_checkpoint(model, model_file, map_location='cpu')

        if not self.distributed:
            model = MMDataParallel(model, device_ids=[0])
            single_gpu_test(model, data_loader, labelmap,
                    predict_result_file)
        else:
            model = MMDistributedDataParallel(model.cuda())
            multi_gpu_test(model, data_loader,
                    labelmap,
                    predict_result_file)

def convert_mmresult_to_rects(result, labelmap):
    rects = []
    for label in range(len(result)):
        bboxes = result[label]
        for i in range(bboxes.shape[0]):
            rect = {}
            rect['rect'] = [float(x) for x in bboxes[i]]
            rect['conf'] = float(bboxes[i][4])
            rect['class'] = labelmap[label]
            rects.append(rect)
    return rects

