import json
import logging
import os.path as op
import torch
from torch import nn
from qd.qd_pytorch import ModelPipeline, torch_load
from qd.layers.loss import ModelLoss
from qd.qd_common import get_mpi_rank
from qd.tsv_io import tsv_writer
from qd.opt.checkpoint import DetectronCheckpointer
from qd.qd_pytorch import synchronize
import shutil
from qd.qd_common import DummyCfg
import time
from qd.qd_pytorch import replace_module
from qd.layers.group_batch_norm import GroupBatchNorm, get_normalize_groups
from qd.qd_common import json_dump


class MaskClassificationPipeline(ModelPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'normalization_group': 32,
            'normalization_group_size': None,
            'freeze_last_bn_stats': 0,
            'predict_ema_decay': None,
            })
        self.step_lr = self.parse_iter(self.step_lr)
        self.max_iter = self.parse_iter(self.max_iter)
        self.num_class = len(self.get_labelmap())

    def get_train_model(self):
        # prepare the model
        model = self._get_model(self.pretrained, self.num_class)
        model.train()

        model = self.model_surgery(model)

        criterion = self._get_criterion()
        # we need wrap model output and criterion into one model, to re-use
        # maskrcnn trainer
        model = self.combine_model_criterion(model, criterion)
        return model

    def combine_model_criterion(self, model, criterion):
        model = ModelLoss(model, criterion)
        return model

    def init_apex_amp(self, model, optimizer):
        from apex import amp
        # Initialize mixed-precision training
        use_mixed_precision = False
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        logging.info('start to amp init')
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
        logging.info('end amp init')
        return model, optimizer

    def freeze_parameters(self, model):
        if self.last_fixed_param:
            from qd.qd_pytorch import freeze_parameters_by_last_name
            freeze_parameters_by_last_name(model, self.last_fixed_param)
        if self.freeze_all_except:
            from qd.qd_pytorch import freeze_all_parameters_except
            freeze_all_parameters_except(model, self.freeze_all_except)
        if self.freeze_last_bn_stats:
            from qd.qd_pytorch import freeze_last_bn_stats
            freeze_last_bn_stats(model, self.freeze_last_bn_stats)

    def create_checkpointer(self, model, optimizer, scheduler):
        save_to_disk = get_mpi_rank() == 0
        if self.cfg is not None:
            cfg = self.cfg
        else:
            cfg = DummyCfg()
        checkpointer = DetectronCheckpointer(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=op.join(self.output_folder, 'snapshot'),
            save_to_disk=save_to_disk,
        )
        return checkpointer

    def train(self):
        device = torch.device('cuda')
        model = self.get_train_model()
        self.freeze_parameters(model)
        model.to(device)

        optimizer = self.get_optimizer(model)
        logging.info(optimizer)

        model, optimizer = self.init_apex_amp(model, optimizer)

        model = self._data_parallel_wrap(model)

        scheduler = self.get_lr_scheduler(optimizer)
        logging.info(scheduler)

        logging.info(model)

        checkpointer = self.create_checkpointer(model, optimizer, scheduler)

        extra_checkpoint_data = checkpointer.load(self.basemodel,
                model_only=True)

        arguments = {}
        arguments['iteration'] = 0
        arguments.update(extra_checkpoint_data)

        # use the maskrcnn trainer engine
        train_loader = self.get_train_data_loader(
                start_iter=arguments['iteration'])

        self.do_train(model, train_loader, optimizer, scheduler, checkpointer,
            arguments)

        model_final = op.join(self.output_folder, 'snapshot', 'model_final.pth')
        last_iter = self._get_checkpoint_file(iteration=self.max_iter)
        if self.mpi_rank == 0:
            if not op.isfile(last_iter):
                shutil.copy(model_final, last_iter)

        synchronize()
        return last_iter

    def do_train(self, model, train_loader, optimizer, scheduler, checkpointer,
            arguments):
        device = torch.device('cuda')
        if self.dict_trainer:
            from qd.opt.trainer import do_train_dict
            do_train_dict(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpointer=checkpointer,
                device=device,
                checkpoint_period=self.get_snapshot_steps(),
                arguments=arguments,
                log_step=self.log_step,
            )
        else:
            from qd.opt.trainer import do_train
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
            )

    def _get_old_check_point_file(self, i):
        return op.join(self.output_folder, 'snapshot', 'model_{:07d}.pth'.format(i))

    def get_train_data_loader(self, start_iter=0):
        return self._get_data_loader(data=self.data,
                split='train', stage='train',
                start_iter=start_iter)

    def get_sampler(self, dataset, stage, shuffle):
        if self.distributed:
            if stage == 'train' and self.infinite_sampler:
                from qd.opt.sampler import InfiniteSampler
                sampler = InfiniteSampler(len(dataset))
            else:
                from maskrcnn_benchmark.data import samplers
                sampler = samplers.DistributedSampler(dataset,
                        shuffle=shuffle,
                        length_divisible=self.batch_size)
        else:
            if stage == 'train' and self.infinite_sampler:
                from qd.opt.sampler import InfiniteSampler
                sampler = InfiniteSampler(len(dataset), shuffle_at_init=shuffle)
            else:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(
                            dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        return sampler

    def get_train_batch_sampler(self, sampler, stage, start_iter):
        if stage == 'train':
            if not self.infinite_sampler:
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    sampler, self.batch_size, drop_last=False
                )
                from maskrcnn_benchmark.data import samplers
                batch_sampler = samplers.IterationBasedBatchSampler(
                    batch_sampler, self.max_iter, start_iter
                )
            else:
                from qd.opt.sampler import InfiniteBatchSampler, MaxIterBatchSampler
                batch_sampler = InfiniteBatchSampler(sampler, self.batch_size)
                batch_sampler = MaxIterBatchSampler(
                    batch_sampler,
                    self.max_iter,
                    start_iter)
        return batch_sampler

    def _get_data_loader(self, data, split, stage, shuffle=True, start_iter=0):
        if stage == 'train':
            dataset_type = self.dataset_type
        elif self.test_dataset_type is None:
            dataset_type = self.dataset_type
        else:
            dataset_type = self.test_dataset_type
        dataset = self._get_dataset(data, split, stage=stage,
                labelmap=self.get_labelmap(),
                dataset_type=dataset_type)

        sampler = self.get_sampler(dataset, stage, shuffle)

        if stage == 'train':
            batch_sampler = self.get_train_batch_sampler(sampler, stage, start_iter)
            logging.info('batch sampler = {}'.format(batch_sampler))
            loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=self.num_workers,
                pin_memory=True,
                batch_sampler=batch_sampler,
                collate_fn=self.train_collate_fn,
                )
        else:
            loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.test_collate_fn,
                )
        return loader

    def get_test_data_loader(self):
        return self._get_data_loader(
                data=self.test_data,
                split=self.test_split,
                stage='test',
                shuffle=False)

    def _get_test_data_loader(self, test_data, test_split, labelmap):
        test_dataset = self._get_dataset(test_data, test_split, stage='test',
                labelmap=labelmap)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.test_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True)

        return test_dataset, test_loader

    def get_rank_specific_tsv(self, f, rank):
        return '{}_{}_{}.tsv'.format(f, rank, self.mpi_size)

    def model_surgery(self, model):
        if self.convert_bn == 'L1':
            raise NotImplementedError
        elif self.convert_bn == 'L2':
            raise NotImplementedError
        elif self.convert_bn == 'GN':
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: torch.nn.GroupNorm(32, m.num_features),
                    )
        elif self.convert_bn == 'LNG': # layer norm by group norm
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: torch.nn.GroupNorm(1, m.num_features))
        elif self.convert_bn == 'ING': # Instance Norm by group norm
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: torch.nn.GroupNorm(m.num_features, m.num_features))
        elif self.convert_bn == 'GBN':
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: GroupBatchNorm(get_normalize_groups(m.num_features, self.normalization_group,
                        self.normalization_group_size), m.num_features))
        elif self.convert_bn == 'SBN':
            if self.distributed:
                model = replace_module(model,
                        lambda m: isinstance(m, torch.nn.BatchNorm2d) or
                                       isinstance(m, torch.nn.BatchNorm1d),
                        lambda m: torch.nn.SyncBatchNorm(m.num_features,
                            eps=m.eps,
                            momentum=m.momentum,
                            affine=m.affine,
                            track_running_stats=m.track_running_stats))
        #elif self.convert_bn == 'NSBN':
            #if self.distributed:
                #from qd.layers.batch_norm import NaiveSyncBatchNorm
                #model = replace_module(model,
                        #lambda m: isinstance(m, torch.nn.BatchNorm2d),
                        #lambda m: NaiveSyncBatchNorm(m.num_features,
                            #eps=m.eps,
                            #momentum=m.momentum,
                            #affine=m.affine,
                            #track_running_stats=m.track_running_stats))
        elif self.convert_bn == 'CBN':
            from qd.layers.batch_norm import ConvergingBatchNorm
            model = replace_module(model,
                    lambda m: isinstance(m, torch.nn.BatchNorm2d),
                    lambda m: ConvergingBatchNorm(
                        policy=self.cbn_policy,
                        max_iter=self.max_iter,
                        gamma=self.cbn_gamma,
                        num_features=m.num_features,
                        eps=m.eps,
                        momentum=m.momentum,
                        affine=True,
                        track_running_stats=m.track_running_stats,
                        ))
        else:
            assert self.convert_bn is None, self.convert_bn
        if self.hswish2relu6:
            from qd.layers.mitorch_models.modules.activation import HardSwish
            model = replace_module(model,
                    lambda m: isinstance(m,
                                         HardSwish),
                    lambda m: torch.nn.ReLU6(inplace=True))
        # assign a name to each module so that we can use it in each module to
        # print debug information
        for n, m in model.named_modules():
            m.name_from_root = n
        return model

    def demo(self, image_path):
        from qd.process_image import load_image
        cv_im = load_image(image_path)
        self.predict_one(cv_im)

    def _get_load_model_demo(self):
        if self.model is None:
            model = self.get_test_model()
            model_file = self._get_checkpoint_file()
            self.load_test_model(model, model_file)
            model = model.to(self.device)
            self.model = model
            model.eval()
        return self.model

    def predict_one(self, cv_im):
        model = self._get_load_model_demo()
        softmax_func = self._get_test_normalize_module()
        from qd.layers import ForwardPassTimeChecker
        model = ForwardPassTimeChecker(model)
        transform = self.get_transform('test')
        im = transform(cv_im)
        im = im[None, :]
        im = im.to(self.device)
        output = model(im)
        if softmax_func is not None:
            output = softmax_func(output)
        all_tops, all_top_indexes = output.topk(5, dim=1,
                largest=True, sorted=False)

        tops, top_indexes = all_tops[0], all_top_indexes[0]
        labelmap = self.get_labelmap()
        all_tag = [{'class': labelmap[i], 'conf': float(t)} for t, i in
                zip(tops, top_indexes)]
        return all_tag

    def get_test_model(self):
        model = self._get_model(pretrained=False,
                num_class=len(self.get_labelmap()))

        model = self.model_surgery(model)

        return model

    def load_test_model(self, model, model_file):
        if self.predict_ema_decay:
            out_model_file = op.splitext(model_file)[0] + '.ema{}.pt'.format(self.predict_ema_decay)
            if self.mpi_rank == 0 and not op.isfile(out_model_file):
                param = torch_load(model_file)
                from qd.opt.ema_optimizer import replace_ema_param
                replace_ema_param(param, decay=self.predict_ema_decay)
                from qd.qd_pytorch import torch_save
                torch_save(param, out_model_file)
            synchronize()
            model_file = out_model_file

        checkpointer = DetectronCheckpointer(cfg=DummyCfg(),
                model=model,
                save_dir=self.output_folder)
        checkpointer.load(model_file)

    def wrap_feature_extract(self, model):
        from qd.layers.feature_extract import FeatureExtract
        model = FeatureExtract(model, self.predict_extract)
        return model

    def predict_iter(self, dataloader, model, softmax_func, meters):
        from tqdm import tqdm
        start = time.time()
        if self.predict_extract:
            model = self.wrap_feature_extract(model)
        for i, data_from_loader in tqdm(enumerate(dataloader),
                total=len(dataloader)):
            is_dict_data = isinstance(data_from_loader, dict)
            if not is_dict_data:
                # this is the deprecated cases. Dictionary should be better for
                # abstract
                inputs, _, keys = data_from_loader
            else:
                inputs = data_from_loader
                keys = data_from_loader

            #logging.info('debugging')
            #inputs = torch_load('/tmp/i')

            if self.test_max_iter is not None and i >= self.test_max_iter:
                # this is used for speed test, where we only would like to run a
                # few images
                break
            meters.update(data=time.time() - start)
            start = time.time()
            if is_dict_data:
                for k in inputs:
                    if hasattr(inputs[k], 'to'):
                        inputs[k] = inputs[k].to(self.device)
            elif hasattr(inputs, 'to'):
                inputs = inputs.to(self.device)
            meters.update(input_to_cuda=time.time() - start)
            start = time.time()
            output = model(inputs)
            meters.update(model=time.time() - start)
            start = time.time()
            if self.predict_extract:
                for row in self.feature_to_tsv_row(
                    model.get_features(), self.predict_extract, keys):
                    yield row
            else:
                if softmax_func is not None:
                    output = softmax_func(output)
                for row in self.predict_output_to_tsv_row(output, keys,
                                                          dataloader=dataloader):
                    yield row
            meters.update(write=time.time() - start)
            start = time.time()

    def feature_to_tsv_row(self, features, feature_names, keys):
        for i, key in enumerate(keys):
            info = []
            for f, f_name in zip(features, feature_names):
                info.append({'feature': f[i].tolist(), 'name': f_name})
            yield key, json_dump(info)

    def predict(self, model_file, predict_result_file):
        if self.mpi_size > 1:
            sub_predict_file = self.get_rank_specific_tsv(predict_result_file,
                    self.mpi_rank)
        else:
            sub_predict_file = predict_result_file

        model = self.get_test_model()
        self.load_test_model(model, model_file)

        model = model.to(self.device)
        dataloader = self.get_test_data_loader()

        softmax_func = self._get_test_normalize_module()

        model.eval()
        from maskrcnn_benchmark.utils.metric_logger import MetricLogger
        meters = MetricLogger(delimiter="  ")
        if self.test_mergebn:
            from qd.layers import MergeBatchNorm
            model = MergeBatchNorm(model)
        from qd.layers import ForwardPassTimeChecker
        model = ForwardPassTimeChecker(model)
        tsv_writer(self.predict_iter(dataloader, model, softmax_func, meters),
                sub_predict_file)

        speed_yaml = sub_predict_file + '.speed.yaml'
        from qd.qd_common import write_to_yaml_file
        write_to_yaml_file(model.get_time_info(), speed_yaml)
        from qd.qd_common import create_vis_net_file
        create_vis_net_file(speed_yaml,
                op.splitext(speed_yaml)[0] + '.vis.txt')
        logging.info(str(meters))

        from maskrcnn_benchmark.utils.comm import is_main_process
        # we need to sync before merging all to make sure each rank finish its
        # own task
        synchronize()
        if self.mpi_size > 1 and is_main_process():
            from qd.process_tsv import concat_tsv_files
            cache_files = [self.get_rank_specific_tsv(predict_result_file, i)
                for i in range(self.mpi_size)]
            concat_tsv_files(cache_files, predict_result_file)
            from qd.process_tsv import delete_tsv_files
            delete_tsv_files(cache_files)
        if is_main_process():
            # in distributed testing, some images might be predicted by
            # more than one worker since the distributed sampler only
            # garrantee each image will be processed at least once, not
            # exactly once. Thus, we have to remove the duplicate
            # predictions.
            ordered_keys = dataloader.dataset.get_keys()
            from qd.tsv_io import reorder_tsv_keys
            reorder_tsv_keys(predict_result_file, ordered_keys, predict_result_file)

            # during prediction, we also computed the time cost. Here we
            # merge the time cost
            speed_cache_files = [c + '.speed.yaml' for c in cache_files]
            speed_yaml = predict_result_file + '.speed.yaml'
            from qd.qd_common import merge_speed_info
            merge_speed_info(speed_cache_files, speed_yaml)
            from qd.qd_common import try_delete
            for x in speed_cache_files:
                try_delete(x)
            vis_files = [op.splitext(c)[0] + '.vis.txt' for c in speed_cache_files]
            from qd.qd_common import merge_speed_vis
            merge_speed_vis(vis_files,
                    op.splitext(speed_yaml)[0] + '.vis.txt')
            for x in vis_files:
                try_delete(x)

        synchronize()
        return predict_result_file

    def predict_output_to_tsv_row(self, output, keys, **kwargs):
        topk = 50
        if output.shape[1] < topk:
            topk = output.shape[1]
        labelmap = self.get_labelmap()
        all_tops, all_top_indexes = output.topk(topk, dim=1,
                largest=True, sorted=False)

        for key, tops, top_indexes in zip(keys, all_tops, all_top_indexes):
            all_tag = [{'class': labelmap[i], 'conf': float(t)} for t, i in
                    zip(tops, top_indexes)]
            yield key, json.dumps(all_tag)

