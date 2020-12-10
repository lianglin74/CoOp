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

from qd.qd_common import ensure_directory
from qd.mask.utils.comm import is_main_process
from qd.torch_common import synchronize
from qd.mask.utils.metric_logger import AverageMeter
from qd.mask.modeling.captioning.utils import check_yaml_file
from qd.mask.modeling.captioning.utils_data import make_data_loader
from qd.mask.modeling.captioning.utils_solver import get_optimizer, get_scheduler
from qd.mask.layers.bert import BertTokenizer, BertConfig, BertImgForPreTraining
from qd.qd_common import get_mpi_rank, get_mpi_size
from qd.torch_common import to, set_seed


def save_checkpoint(model, tokenizer, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(
        args.output_dir,
        'snapshot',
        'model_iter_{:07d}'.format(iteration))
    if not is_main_process():
        return checkpoint_dir
    ensure_directory(checkpoint_dir)
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
    if args.distributed and self.device == 'cuda':
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self.mpi_local_rank],
            output_device=self.mpi_local_rank,
            find_unused_parameters=True,
        )

    start_training_time = time.time()
    end = time.time()
    model.train()
    from qd.logger import MetricLogger
    meters = MetricLogger(delimiter="  ")
    log_start = time.time()
    debug_time = False
    #debug_time = True
    if debug_time:
        from qd.layers.forward_pass_time_checker import ForwardPassTimeChecker
        model = ForwardPassTimeChecker(model)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for iteration, batch in enumerate(train_dataloader):
        iteration += 1
        after_data_load = time.time()
        if iteration > 5:
            meters.update(data_time=after_data_load - end)
        batch = to(batch, self.device)
        after_to_device = time.time()
        meters.update(to_device=after_to_device - after_data_load)
        inputs = batch
        if use_amp:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = model(**inputs)
                loss = sum(v for _, v in loss_dict.items())
                #loss, logits_mask, logits_seq = outputs[0], outputs[1], outputs[2]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(**inputs)
            loss = sum(v for _, v in loss_dict.items())
            #loss, logits_mask, logits_seq = outputs[0], outputs[1], outputs[2]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        after_fwd_bkw = time.time()
        meters.update(fwd_bkw=after_fwd_bkw - after_to_device)

        meters.update(loss=loss)
        meters.update(**loss_dict)
        batch_time_update = time.time() - end
        if iteration > 5:
            meters.update(batch_time=batch_time_update)
        if debug_time and iteration > 5:
            from qd.qd_common import write_to_yaml_file
            speed_yaml = '/tmp/time.yaml'
            write_to_yaml_file(model.get_time_info(),
                speed_yaml,
            )
            from qd.qd_common import create_vis_net_file
            create_vis_net_file(speed_yaml,
                    op.splitext(speed_yaml)[0] + '.vis.txt')
            import ipdb;ipdb.set_trace(context=15)
        end = time.time()
        if iteration % args.logging_steps == 0 or iteration == max_iter:
            eta_seconds = batch_time_update * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            speed = get_mpi_size() * args.logging_steps * len(batch['input_ids']) / (time.time() - log_start)
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

def load_tokenizer_model(self, args):
    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertImgForPreTraining, BertTokenizer
    config = config_class.from_pretrained(args.config_name if args.config_name \
            else args.model_name_or_path)
    assert config is not None
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                    else args.model_name_or_path, do_lower_case=args.do_lower_case)
    config.img_feature_dim = args.img_feature_dim
    config.img_feature_type = 'frcnn'
    config.hidden_dropout_prob = args.drop_out
    config.loss_type = self.loss_type
    if self.prior_prob is not None:
        config.prior_prob = self.prior_prob
    # update model structure if specified in arguments
    update_params = [
        'num_hidden_layers',
        'hidden_size',
        'num_attention_heads',
        'intermediate_size',
        'use_img_layernorm',
        'img_layer_norm_eps',
    ]
    model_structure_changed = [False] * len(update_params)
    for idx, param in enumerate(update_params):
        arg_param = getattr(args, param)
        config_param = getattr(config, param) if hasattr(config, param) else -1
        if arg_param != -1 and arg_param != config_param:
            logging.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
            setattr(config, param, arg_param)
            model_structure_changed[idx] = True

    if args.model_name_or_path and op.isfile(
            op.join(args.model_name_or_path, 'pytorch_model.bin')):
        logging.info('init from {}'.format(args.model_name_or_path))
        model = model_class.from_pretrained(args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        model = model_class(config=config) # init from scratch
    assert model is not None
    logging.info("Load pretrained model: {}".format(args.model_name_or_path))

    #if any(model_structure_changed):
        #assert config.hidden_size % config.num_attention_heads == 0
        #if args.load_partial_weights:
            ## can load partial weights when changing layer only.
            #assert not any(model_structure_changed[1:]), "Cannot load partial weights " \
                    #"when any of ({}) is changed.".format(', '.join(update_params[1:]))
            #model = model_class.from_pretrained(args.model_name_or_path,
                    #from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
            #logging.info("Load partial weights for bert layers.")
        #else:
            #model = model_class(config=config) # init from scratch
            #logging.info("Init model from scratch.")
    #else:
        #model = model_class.from_pretrained(args.model_name_or_path,
                #from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        #logging.info("Load pretrained model: {}".format(args.model_name_or_path))
    return tokenizer, model

def main(self, args):
    check_arguments(args)
    ensure_directory(args.output_dir)
    set_seed(args.seed, args.num_gpus)
    logging.info("Using {} GPUs".format(args.num_gpus))

    tokenizer, model = load_tokenizer_model(self, args)
    logging.info(model)

    if args.freeze_embedding:
        logging.info("Freeze word embedding")
        model.bert.embeddings.word_embeddings.weight.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    logging.info('Model total parameters: {}'.format(total_params))
    model = model.to(self.device)

    logging.info("Training parameters %s", args)
    train_dataloader = make_data_loader(args, args.train_yaml, tokenizer,
            args.distributed, is_train=True, is_pretrain=True)
    logging.info(train_dataloader.dataset)
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
            'num_workers': 4,
            'weight_decay': 0.05,
            'adam_epsilon': 1e-08,
            'add_od_labels': True,
            'qa2caption': None,
            'basemodel': 'models/captioning/bert4-base-uncased-clean/',
            'tokenizer_name': '',
            # here we use 4 as default, but should be -1. This is only for back
            # compatibility.
            'num_hidden_layers': 4,
            'use_img_layernorm': False,
            'pert_labels_prob': 0.5,
            'on_memory': False,
            'train_split': 'train',
            'max_img_seq_length': 50,
            'od_label_conf': 0.8,
            'no_sort_by_conf': False,
            'unique_labels_on': True,
            'pert_caption_prob': 0.0,
            'loss_type': 'classification_classification',
            'dataset_type': None,
            'region_loss_for_unmatched': True,
        })

    def append_predict_param(self, cc):
        pass

    def get_train_data_loader(self, start_iter):
        # do not use this functions
        train_dataloader = make_data_loader(self, self.train_yaml, self.tokenizer,
                self.distributed, is_train=True, is_pretrain=True)
        return train_dataloader

    def get_test_data_loader(self):
        pass

    def train(self):
        train_yaml = self.train_yaml
        if train_yaml is None:
            train_yaml = op.join(self.output_folder, 'train_yaml.yaml')
            if self.mpi_rank == 0:
                convert_data_to_yaml(
                    data=self.data,
                    split=self.train_split,
                    yaml=train_yaml,
                    label=self.train_label_tsv,
                    feature=self.train_feature_tsv,
                    label_version=self.train_label_version,
                    feature_version=self.train_feature_version,
                    qd_format=False,
                )
            synchronize()

        logging.info('train_yaml = {}'.format(train_yaml))

        max_seq_length = 35

        if not self.add_od_labels:
            max_seq_a_length = max_seq_length
            mask_od_labels = False
        else:
            max_seq_a_length = 20
            mask_od_labels = True

        param = {
            'qd_format': self.qd_format,
            'data': self.data,
            'train_label_version': self.train_label_version,
            'train_feature_version': self.train_feature_version,
            'od_label_conf': self.od_label_conf,
            'no_sort_by_conf': self.no_sort_by_conf,
            'unique_labels_on': self.unique_labels_on,
            'qa2caption': self.qa2caption,
            'pert_caption_prob': self.pert_caption_prob,
            'pert_labels_prob': self.pert_labels_prob,
            'img_feature_dim': self.img_feature_dim,
            'adam_epsilon': self.adam_epsilon,
            'add_od_labels': self.add_od_labels,
            'effective_batch_size': self.effective_batch_size,
            'img_feat_label_type': self.img_feat_label_type,
            'distributed': self.distributed,

            'learning_rate': self.base_lr,
            'data_dir': 'data',
            'do_lower_case': True,
            'drop_out': 0.1,
            'num_gpus': self.mpi_size,
            'freeze_embedding': False,
            'hidden_size': -1,
            'intermediate_size': -1,
            'load_partial_weights': False,
            'local_rank': self.mpi_local_rank,
            'logging_steps': self.log_step,
            'mask_loss_for_unmatched': False,
            'mask_od_labels': mask_od_labels,
            'mask_prob': 0.15,
            'mask_type': 'seq2seq',
            'max_img_seq_length': self.max_img_seq_length,
            'max_masked_tokens': 5,
            'max_seq_a_length': max_seq_a_length,
            'max_seq_length': max_seq_length,
            #'model_name_or_path': 'models/captioning/bert-base-uncased-clean/',
            'model_name_or_path': self.basemodel,
            'num_attention_heads': -1,
            'num_hidden_layers': self.num_hidden_layers,
            'num_train_epochs': self.max_epoch,
            'num_workers': self.num_workers,
            'on_memory': self.on_memory,
            'output_dir': self.output_folder,
            'per_gpu_train_batch_size': self.effective_batch_size // self.mpi_size,
            'save_steps': 20000,
            'scheduler': 'linear',
            'seed': 88,
            'tokenizer_name': self.tokenizer_name,
            'warmup_steps': self.parse_iter(self.warmup_steps),
            'weight_decay': self.weight_decay,
            'train_shuffle': self.train_shuffle,
            'use_img_layernorm': self.use_img_layernorm,
            'img_layer_norm_eps': 1e-5,
            'dataset_type': self.dataset_type,
            'train_yaml': train_yaml,
            'config_name': '',
            'region_loss_for_unmatched': self.region_loss_for_unmatched,
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


