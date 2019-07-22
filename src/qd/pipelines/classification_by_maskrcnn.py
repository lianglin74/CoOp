import json
import logging
import os.path as op
import torch
from qd.qd_pytorch import ModelPipeline, torch_load
from qd.layers.loss import ModelLoss
from qd.qd_common import get_mpi_rank
from qd.tsv_io import tsv_writer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize
import shutil


class DummyCfg(object):
    # provide a signature of clone(), used by maskrcnn checkpointer
    def clone(self):
        return

class MaskClassificationPipeline(ModelPipeline):
    def __init__(self, **kwargs):
        super(MaskClassificationPipeline, self).__init__(**kwargs)

        self.step_lr = self.parse_iter(self.step_lr)
        self.max_iter = self.parse_iter(self.max_iter)

    def train(self):
        device = torch.device('cuda')
        num_class = len(self.get_labelmap())

        # prepare the model
        model = self._get_model(self.pretrained, num_class)
        criterion = self._get_criterion()
        # we need wrap model output and criterion into one model, to re-use
        # maskrcnn trainer
        model = ModelLoss(model, criterion)
        model.to(device)

        optimizer = self.get_optimizer(model)

        from apex import amp
        # Initialize mixed-precision training
        use_mixed_precision = False
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        logging.info('start to amp init')
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
        logging.info('end amp init')

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

        arguments = {}
        arguments['iteration'] = 0
        arguments.update(extra_checkpoint_data)

        # use the maskrcnn trainer engine
        train_loader = self._get_data_loader(data=self.data,
                split='train', stage='train',
                start_iter=arguments['iteration'])

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
        )

        model_final = op.join(self.output_folder, 'snapshot', 'model_final.pth')
        last_iter = self._get_checkpoint_file(iteration=self.max_iter)
        if self.mpi_rank == 0:
            if not op.isfile(last_iter):
                shutil.copy(model_final, last_iter)

        synchronize()
        return last_iter

    def _get_data_loader(self, data, split, stage, shuffle=True, start_iter=0):
        train_dataset = self._get_dataset(data, split, stage=stage,
                labelmap=self.get_labelmap())

        if self.distributed:
            from maskrcnn_benchmark.data import samplers
            train_sampler = samplers.DistributedSampler(train_dataset,
                    shuffle=shuffle,
                    length_divisible=self.batch_size)
        else:
            if shuffle:
                train_sampler = torch.utils.data.sampler.RandomSampler(
                        train_dataset)
            else:
                train_sampler = torch.utils.data.sampler.SequentialSampler(train_dataset)

        batch_sampler = torch.utils.data.sampler.BatchSampler(
            train_sampler, self.batch_size, drop_last=False
        )
        from maskrcnn_benchmark.data import samplers
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, self.max_iter, start_iter
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            )
        return train_loader

    def _get_test_data_loader(self, test_data, test_split, labelmap):
        test_dataset = self._get_dataset(test_data, test_split, stage='test',
                labelmap=labelmap)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.test_batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True)

        return test_dataset, test_loader

    def get_rank_specific_tsv(self, f, rank):
        return '{}_{}_{}.tsv'.format(f, rank, self.mpi_size)

    def predict(self, model_file, predict_result_file):
        if self.mpi_size > 1:
            sub_predict_file = self.get_rank_specific_tsv(predict_result_file,
                    self.mpi_rank)
        else:
            sub_predict_file = predict_result_file

        labelmap = self.get_labelmap()

        model = self._get_model(pretrained=False, num_class=len(labelmap))
        model = self._data_parallel_wrap(model)
        model.to('cuda')

        checkpointer = DetectronCheckpointer(cfg=DummyCfg(),
                model=model,
                save_dir=self.output_folder)
        checkpointer.load(model_file)

        dataloader = self._get_data_loader(
                data=self.test_data,
                split=self.test_split,
                stage='test',
                shuffle=False)

        softmax_func = self._get_test_normalize_module()

        model.eval()
        def gen_rows():
            from tqdm import tqdm
            for i, (inputs, labels, keys) in tqdm(enumerate(dataloader)):
                inputs = inputs.cuda()
                labels = labels.cuda()
                output = model(inputs)
                output = softmax_func(output)
                for row in self.predict_output_to_tsv_row(output, keys):
                    yield row

        tsv_writer(gen_rows(), sub_predict_file)

        if self.mpi_size > 1:
            from qd.process_tsv import concat_tsv_files
            cache_files = [self.get_rank_specific_tsv(predict_result_file, i)
                for i in range(self.mpi_size)]
            concat_tsv_files(cache_files, predict_result_file)
            from qd.process_tsv import delete_tsv_files
            delete_tsv_files(cache_files)
            # in distributed testing, some images might be predicted by
            # more than one worker since the distributed sampler only
            # garrantee each image will be processed at least once, not
            # exactly once. Thus, we have to remove the duplicate
            # predictions.
            ordered_keys = dataloader.dataset.get_keys()
            from qd.tsv_io import reorder_tsv_keys
            reorder_tsv_keys(predict_result_file, ordered_keys, predict_result_file)

        synchronize()
        return predict_result_file

    def predict_output_to_tsv_row(self, output, keys):
        topk = 50
        labelmap = self.get_labelmap()
        all_tops, all_top_indexes = output.topk(topk, dim=1,
                largest=True, sorted=False)

        for key, tops, top_indexes in zip(keys, all_tops, all_top_indexes):
            all_tag = [{'class': labelmap[i], 'conf': float(t)} for t, i in
                    zip(tops, top_indexes)]
            yield key, json.dumps(all_tag)

