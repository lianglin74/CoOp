from qd.qd_pytorch import TwoCropsTransform
from qd.pipelines.classification_by_maskrcnn import MaskClassificationPipeline
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import os.path as op
import logging
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import ImageFilter
from qd.qd_common import print_frame_info
from qd.qd_pytorch import GaussianBlur
from qd.qd_common import merge_dict_to_cfg
from qd.qd_common import load_from_yaml_file
from qd.qd_common import dict_update_nested_dict
from mmask.config import cfg
from qd.qd_common import (dict_has_path, dict_get_path_value,
                          dict_update_path_value)
from qd.qd_common import dump_to_yaml_str
import argparse
import os
import os.path as op
import json
import time
import datetime
import torch
from qd.tsv_io import convert_data_to_yaml

from mmask.utils.logger import setup_logger
from mmask.utils.comm import synchronize, is_main_process, get_rank
from mmask.utils.miscellaneous import mkdir, set_seed
from mmask.utils.metric_logger import AverageMeter
from mmask.modeling.captioning.utils import check_yaml_file
from mmask.modeling.captioning.utils_data import make_data_loader
from mmask.modeling.captioning.utils_solver import get_optimizer, get_scheduler
from mmask.layers.bert import BertTokenizer, BertConfig, BertImgForPreTraining
from qd.qd_common import get_mpi_rank, get_mpi_size


def save_checkpoint(model, tokenizer, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(
        args.output_dir,
        'snapshot',
        'model_iter_{:07d}'.format(iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logging.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logging.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def old_train(self, args, train_dataloader, model, tokenizer):
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    optimizer = self.get_optimizer(model)
    scheduler = get_scheduler(optimizer, args.scheduler,
        args.warmup_steps, max_iter
    )
    use_amp = self.use_amp
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    start_training_time = time.time()
    end = time.time()
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    log_start = time.time()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for iteration, (img_keys, batch) in enumerate(train_dataloader):
        iteration += 1
        data_time.update(time.time() - end)
        batch = tuple(t.cuda() for t in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'img_feats': batch[3],
            'masked_lm_labels': batch[4],
            'next_sentence_label': batch[5]
        }
        if use_amp:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**inputs)
                #loss, logits_mask, logits_seq = outputs[0], outputs[1], outputs[2]
                loss = outputs[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**inputs)
            #loss, logits_mask, logits_seq = outputs[0], outputs[1], outputs[2]
            loss = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        losses.update(loss.item())
        batch_time_update = time.time() - end
        batch_time.update(time.time() - end)
        end = time.time()
        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time_update * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            speed = get_mpi_size() * args.logging_steps * len(img_keys) / (time.time() - log_start)
            logging.info(
                ' '.join(
                ['eta: {eta}',
                 'iter: {iter}',
                'speed: {speed:.1f} images/sec',
                 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, iter=iteration, speed=speed,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                + '  loss: {:.4f} ({:.4f}), compute: {:.4f} ({:.4f}), data: {:.4f} ({:.4f}), lr: {:.6f}'.format(
                    losses.val, losses.avg, batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                    optimizer.param_groups[0]['lr'])
            )
            log_start = time.time()
        if (args.save_steps > 0 and iteration % args.save_steps == 0) or iteration == max_iter:
            epoch = iteration // iters_per_epoch
            checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, iteration)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logging.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    return checkpoint_dir


def check_arguments(args):
    check_yaml_file(op.join(args.data_dir, args.train_yaml))
    if args.effective_batch_size > 0:
        assert args.effective_batch_size % args.num_gpus == 0
        args.per_gpu_train_batch_size = int(args.effective_batch_size / args.num_gpus)
    if args.mask_od_labels:
        assert args.add_od_labels
    if args.add_od_labels:
        assert args.max_seq_length > args.max_seq_a_length
    else:
        assert args.max_seq_length == args.max_seq_a_length
    assert args.pert_caption_prob + args.pert_labels_prob <= 0.5

def main(self, args):
    check_arguments(args)
    mkdir(args.output_dir)
    set_seed(args.seed, args.num_gpus)
    logging.info("Using {} GPUs".format(args.num_gpus))

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertImgForPreTraining, BertTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name \
            else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                    else args.model_name_or_path, do_lower_case=args.do_lower_case)
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = 'frcnn'
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = 'classification'
    # update model structure if specified in arguments
    update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
    model_structure_changed = [False] * len(update_params)
    for idx, param in enumerate(update_params):
        arg_param = getattr(args, param)
        config_param = getattr(config, param)
        if arg_param > 0 and arg_param != config_param:
            logging.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
            setattr(config, param, arg_param)
            model_structure_changed[idx] = True
    if any(model_structure_changed):
        assert config.hidden_size % config.num_attention_heads == 0
        if args.load_partial_weights:
            # can load partial weights when changing layer only.
            assert not any(model_structure_changed[1:]), "Cannot load partial weights " \
                    "when any of ({}) is changed.".format(', '.join(update_params[1:]))
            model = model_class.from_pretrained(args.model_name_or_path,
                    from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
            logging.info("Load partial weights for bert layers.")
        else:
            model = model_class(config=config) # init from scratch
            logging.info("Init model from scratch.")
    else:
        model = model_class.from_pretrained(args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        logging.info("Load pretrained model: {}".format(args.model_name_or_path))

    if args.freeze_embedding:
        logging.info("Freeze word embedding")
        model.bert.embeddings.word_embeddings.weight.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    logging.info('Model total parameters: {}'.format(total_params))
    model.to(args.device)

    logging.info("Training parameters %s", args)
    train_dataloader = make_data_loader(args, args.train_yaml, tokenizer,
            args.distributed, is_train=True, is_pretrain=True)
    return old_train(self, args, train_dataloader, model, tokenizer)

class MMaskPretrainPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'bgr2rgb': True,
            'train_size_mode': 'random',
            'max_box': 300,
            'base_lr': 0.0002,
            'train_shuffle': True,
            'warmup_steps': 0,
            'num_workers': 16,
            'weight_decay': 0.05,
            'adam_epsilon': 1e-08,
        })


    def append_predict_param(self, cc):
        pass

    def get_train_data_loader(self, start_iter):
        pass

    def get_test_data_loader(self):
        pass

    def train(self):
        train_yaml = self.train_yaml
        if train_yaml is None:
            train_yaml = op.join(self.output_folder, 'train_yaml.yaml')
            if self.mpi_rank == 0:
                convert_data_to_yaml(
                    data=self.data,
                    split='train',
                    yaml=train_yaml,
                    label=self.train_label_tsv,
                    feature=self.train_feature_tsv,
                    label_version=self.train_label_version,
                    feature_version=self.train_feature_version,
                    qd_format=False,
                )
            synchronize()

        logging.info('train_yaml = {}'.format(train_yaml))

        param = {
            'train_yaml': train_yaml,
            'adam_epsilon': self.adam_epsilon,
            'add_od_labels': True,
            'config_name': '',
            'data_dir': 'data',
            'do_lower_case': True,
            'drop_out': 0.1,
            'effective_batch_size': self.effective_batch_size,
            'num_gpus': self.mpi_size,
            'distributed': self.distributed,
            'device': torch.device(self.device_id),
            'freeze_embedding': False,
            'hidden_size': -1,
            'img_feature_dim': self.img_feature_dim,
            'intermediate_size': -1,
            'learning_rate': self.base_lr,
            'load_partial_weights': False,
            'local_rank': self.mpi_local_rank,
            'logging_steps': self.log_step,
            'mask_loss_for_unmatched': False,
            'mask_od_labels': True,
            'mask_prob': 0.15,
            'mask_type': 'seq2seq',
            'max_img_seq_length': 50,
            'max_masked_tokens': 5,
            'max_seq_a_length': 20,
            'max_seq_length': 35,
            'model_name_or_path': 'models/captioning/bert-base-uncased-clean/',
            'no_sort_by_conf': False,
            'num_attention_heads': -1,
            'num_hidden_layers': 4,
            'num_train_epochs': self.max_epoch,
            'num_workers': self.num_workers,
            'od_label_conf': 0.8,
            'on_memory': False,
            'output_dir': self.output_folder,
            'per_gpu_train_batch_size': self.effective_batch_size // self.mpi_size,
            'pert_caption_prob': 0.0,
            'pert_labels_prob': 0.5,
            'save_steps': 20000,
            'scheduler': 'linear',
            'seed': 88,
            'tokenizer_name': '',
            'unique_labels_on': True,
            'warmup_steps': self.parse_iter(self.warmup_steps),
            'weight_decay': self.weight_decay,
            'train_shuffle': self.train_shuffle,
        }
        from pprint import pformat
        logging.info('param = \n{}'.format(pformat(param)))
        from qd.qd_common import make_namespace_by_dict
        args = make_namespace_by_dict(param)
        checkpoint_dir = main(self, args)
        last_model_link = self.get_last_model_link_file()
        from qd.qd_common import write_to_file
        write_to_file(op.relpath(checkpoint_dir, op.dirname(last_model_link)),
                      last_model_link)

    def get_train_model(self):
        pass

    def get_test_model(self):
        pass

    def get_optimizer(self, model):
        if self.optimizer_type is None:
            optimizer = get_optimizer(model, self.weight_decay,
                self.base_lr, self.adam_epsilon
            )
        elif self.optimizer_type in ['LAMB']:
            from pytorch_lamb import Lamb
            optimizer = Lamb(model.parameters(),
                             lr=self.base_lr,
                             weight_decay=self.weight_decay,
                             betas=(.9, .999),
                             adam=False)
        else:
            raise ValueError(self.optimizer_type)
        return optimizer

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        # we cannot get the correct max_iter, which uses (idx_osurce, idx_row,
        # idx_caption) format
        #scheduler = get_scheduler(
            #optimizer,
            #'linear',
            #self.parse_iter(self.warmup_steps),
            #max_iter
        #)
        pass

    def _get_test_normalize_module(self):
        return


