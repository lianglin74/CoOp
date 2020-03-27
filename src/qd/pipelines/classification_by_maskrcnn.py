import json
import logging
import os.path as op
import torch
from torch import nn
from qd.qd_pytorch import ModelPipeline, torch_load
from qd.layers.loss import ModelLoss
from qd.qd_common import get_mpi_rank
from qd.tsv_io import tsv_writer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize
import shutil
from qd.qd_common import DummyCfg
import time
from qd.qd_pytorch import replace_module
from qd.layers.group_batch_norm import GroupBatchNorm, get_normalize_groups


class MaskClassificationPipeline(ModelPipeline):
    def __init__(self, **kwargs):
        super(MaskClassificationPipeline, self).__init__(**kwargs)
        self._default.update({
            'normalization_group': 32,
            'normalization_group_size': None
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

    def train(self):
        device = torch.device('cuda')
        model = self.get_train_model()
        model.to(device)

        optimizer = self.get_optimizer(model)

        model, optimizer = self.init_apex_amp(model, optimizer)

        model = self._data_parallel_wrap(model)

        scheduler = self.get_lr_scheduler(optimizer)

        save_to_disk = get_mpi_rank() == 0
        checkpointer = DetectronCheckpointer(
            cfg=DummyCfg(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=op.join(self.output_folder, 'snapshot'),
            save_to_disk=save_to_disk,
        )

        extra_checkpoint_data = checkpointer.load(self.basemodel,
                model_only=True)

        logging.info(model)

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
        from maskrcnn_benchmark.engine.trainer import do_train
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
            set_model_to_train=False,
        )

    def get_train_data_loader(self, start_iter=0):
        return self._get_data_loader(data=self.data,
                split='train', stage='train',
                start_iter=start_iter)

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

        if self.distributed:
            from maskrcnn_benchmark.data import samplers
            sampler = samplers.DistributedSampler(dataset,
                    shuffle=shuffle,
                    length_divisible=self.batch_size)
        else:
            if shuffle:
                sampler = torch.utils.data.sampler.RandomSampler(
                        dataset)
            else:
                sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        if stage == 'train':
            batch_sampler = torch.utils.data.sampler.BatchSampler(
                sampler, self.batch_size, drop_last=False
            )
            from maskrcnn_benchmark.data import samplers
            batch_sampler = samplers.IterationBasedBatchSampler(
                batch_sampler, self.max_iter, start_iter
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                num_workers=self.num_workers,
                pin_memory=True,
                batch_sampler=batch_sampler,
                )
        else:
            loader = torch.utils.data.DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.test_batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
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
        else:
            assert self.convert_bn is None, self.convert_bn
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
        checkpointer = DetectronCheckpointer(cfg=DummyCfg(),
                model=model,
                save_dir=self.output_folder)
        checkpointer.load(model_file)

    def predict_iter(self, dataloader, model, softmax_func, meters):
        from tqdm import tqdm
        start = time.time()
        for i, (inputs, _, keys) in tqdm(enumerate(dataloader),
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
            output = model(inputs)
            meters.update(model=time.time() - start)
            start = time.time()
            if softmax_func is not None:
                output = softmax_func(output)
            meters.update(softmax=time.time() - start)
            for row in self.predict_output_to_tsv_row(output, keys):
                yield row
            start = time.time()

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

    def predict_output_to_tsv_row(self, output, keys):
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

