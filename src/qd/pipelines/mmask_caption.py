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
from tqdm import tqdm

from mmask.utils.comm import synchronize, is_main_process
from mmask.utils.comm import get_rank, get_world_size
from mmask.utils.miscellaneous import mkdir, set_seed
from mmask.utils.miscellaneous import delete_tsv_files, concat_tsv_files
from mmask.utils.metric_logger import MetricLogger
from mmask.structures.tsv_file_ops import tsv_writer, reorder_tsv_keys
from mmask.modeling.captioning.utils import check_yaml_file
from mmask.modeling.captioning.utils_data import make_data_loader
from mmask.modeling.captioning.utils_solver import get_optimizer, get_scheduler
from mmask.layers.bert import BertTokenizer, BertConfig, BertForImageCaptioning
from mmask.modeling.captioning.utils_caption_evaluate import (
        evaluate_on_coco_caption, ScstRewardCriterion)

from qd.tsv_io import convert_data_to_yaml

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


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data # argmax
    return logits == labels


def plot_validation_curve(res_file):
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    with open(res_file, 'r') as f:
        res = json.load(f)
    epochs = [r['epoch'] for r in res]
    iters = [r['iteration'] for r in res]
    iters_per_epoch = max(iters) / (max(epochs) + 1)
    epochs = [i / iters_per_epoch for i in iters]
    metrics = ['Bleu_4', 'METEOR', 'CIDEr', 'SPICE']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        plt.plot(epochs, [r[metric] for r in res], 'r*-')
        plt.title(metric)
    plt.savefig(op.splitext(res_file)[0] + '.png') 


def train(args, train_dataloader, val_dataloader, model, tokenizer):
    meters = MetricLogger(delimiter='  ')
    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs
    optimizer = get_optimizer(model, args.weight_decay, 
        args.learning_rate, args.adam_epsilon
    )
    scheduler = get_scheduler(optimizer, args.scheduler, 
        args.warmup_steps, max_iter
    )
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    if args.scst:
        scst_criterion = ScstRewardCriterion(
            cider_cached_tokens=op.join(args.data_dir, args.cider_cached_tokens),
            baseline_type=args.sc_baseline_type,
        )
        logging.info("  SCST training...")

    #eval_log = []
    #best_score = 0
    start_training_time = time.time()
    end = time.time()
    for iteration, (img_keys, batch) in enumerate(train_dataloader):
        iteration += 1
        data_time = time.time() - end
        batch = tuple(t.to(args.device) for t in batch)
        if not args.scst:
            model.train()
            inputs = {
                'input_ids': batch[0], 'attention_mask': batch[1],
                'token_type_ids': batch[2], 'img_feats': batch[3], 
                'masked_pos': batch[4], 'masked_ids': batch[5]
            }
            outputs = model(**inputs)
            loss, logits = outputs[:2]
            masked_ids = inputs['masked_ids']
            masked_ids = masked_ids[masked_ids != 0]
            batch_score = compute_score_with_logits(logits, masked_ids)
            batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])
        else:
            loss = scst_train_iter(args, train_dataloader, model, scst_criterion, 
                    img_keys, batch, tokenizer)
            batch_acc = scst_criterion.get_score()

        loss_dict = {'loss': loss, 'acc': batch_acc}
        meters.update_metrics({'loss': loss_dict})

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update_metrics(
            {'time_info': {'compute': batch_time, 'data': data_time}}
        )
        meters.update_params(
            {'params': {'learning_rate': optimizer.param_groups[0]['lr']}}
        )
        if iteration % args.logging_steps == 0 or iteration == max_iter:
            avg_time = meters.meters['time_info']['compute'].global_avg
            eta_seconds = avg_time * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logging.info(
                meters.delimiter.join(
                ['eta : {eta}', 'iter: {iter}', 'max mem : {memory:.0f}',]
                ).format(eta=eta_string, iter=iteration, 
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0) 
                + '\n    ' + meters.get_logs(iteration)
            )
        if (args.save_steps > 0 and iteration % args.save_steps == 0) or iteration == max_iter:
            epoch = iteration // iters_per_epoch
            checkpoint_dir = save_checkpoint(model, tokenizer, args, epoch, iteration)
            #if args.evaluate_during_training:
                #logging.info("Perform evaluation at iteration: %d" % (iteration))
                #evaluate_file = evaluate(args, val_dataloader, model, tokenizer, checkpoint_dir)
                #if get_world_size() > 1:
                    #torch.distributed.barrier()
                #if is_main_process():
                    #with open(evaluate_file, 'r') as f:
                        #res = json.load(f)
                    #best_score = max(best_score, res['CIDEr'])
                    #res['epoch'] = epoch
                    #res['iteration'] = iteration
                    #res['best_CIDEr'] = best_score
                    #eval_log.append(res)
                    #with open(op.join(args.output_dir, 'eval_logs.json'), 'w') as f:
                        #json.dump(eval_log, f)
                #if get_world_size() > 1:
                    #torch.distributed.barrier()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logging.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    return checkpoint_dir


def scst_train_iter(args, train_dataloader, model, scst_criterion, 
        img_keys, batch, tokenizer):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, 
        tokenizer.sep_token, tokenizer.pad_token, tokenizer.mask_token]
    )
    inputs = {'is_decode': True,
        'input_ids': batch[0], 'attention_mask': batch[1],
        'token_type_ids': batch[2], 'img_feats': batch[3],
        'masked_pos': batch[4],
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id],
        'mask_token_id': mask_token_id,
        # for adding od labels
        'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        'num_beams': 1,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": 1,
        "num_keep_best": 1,
    }

    def _ids_to_captions(all_ids):
        captions = []
        for ids in all_ids:
            c = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            captions.append(c)
        return captions

    if args.sc_baseline_type == 'greedy':
        model.eval()
        with torch.no_grad():
            greedy_res_raw, _ = model(**inputs)
            greedy_res_raw.squeeze_(1)  # batch_size * max_len
        greedy_res = _ids_to_captions(greedy_res_raw)
    else:
        greedy_res = None

    model.train()
    inputs['do_sample'] = True
    inputs['num_return_sequences'] = args.sc_train_sample_n
    sample_res_raw, sample_logprobs = model(**inputs)
    sample_res_raw.squeeze_(1)
    sample_logprobs.squeeze_(1)
    assert sample_logprobs.requires_grad == True
    assert sample_res_raw.requires_grad == False
    sample_res = _ids_to_captions(sample_res_raw)

    gt_res = [train_dataloader.dataset.get_captions_by_key(k) for k in img_keys]
    loss = scst_criterion(gt_res, greedy_res, sample_res, sample_logprobs)
    return loss


#def get_predict_file(output_dir, args, data_yaml_file):
    #cc = ['pred']
    ## example data_yaml_file: datasets/coco_caption/test.yaml
    #data = data_yaml_file.split('/')[-2]
    #if data != 'coco_caption':
        #cc.append(data)
    #cc.append(op.splitext(op.basename(data_yaml_file))[0])
    #cc.append('beam{}'.format(args.num_beams))
    #cc.append('max{}'.format(args.max_gen_length))
    #if args.add_od_labels:
        #cc.append('wlabels')
    #if args.num_keep_best != 1:
        #cc.append('best{}'.format(args.num_keep_best))
    #if args.use_cbs:
        #cc.append('cbs{}'.format(args.min_constraints_to_satisfy))
        #if args.use_hypo:
            #cc.append('hypo')
        #if args.decoding_constraint:
            #cc.append('dc')
        #if args.remove_bad_endings:
            #cc.append('rmbe')
    #if args.output_hidden_states:
        #cc.append('hidden')
    #return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))

def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    return op.splitext(predict_file)[0] + '.eval.json'

def evaluate(args,
             val_dataloader,
             model,
             tokenizer,
             output_dir):
    predict_file = args.predict_file
    #predict_file = get_predict_file(output_dir, args,
            #val_dataloader.dataset.yaml_file)
    if op.isfile(predict_file):
        logging.info('Skip predict. {} already exists'.format(predict_file))
    else:
        test(args, val_dataloader, model, tokenizer, predict_file)

    synchronize()

    #evaluate_file = get_evaluate_file(predict_file)
    #if is_main_process():
        ##caption_file = val_dataloader.dataset.get_caption_file_in_coco_format()
        #caption_file = args.caption_file_in_coco_format
        #if op.isfile(evaluate_file):
            #logging.info('Skip evaluation. {} already exists'.format(evaluate_file))
        #else:
            #data = val_dataloader.dataset.yaml_file.split('/')[-2]
            #if 'nocaps' not in data:
                #result = evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file)
                #logging.info('evaluation result: {}'.format(str(result)))
                #logging.info('evaluation result saved to {}'.format(evaluate_file))
    if get_world_size() > 1:
        torch.distributed.barrier()
    return evaluate_file

def test(args, test_dataloader, model, tokenizer, predict_file):
    if op.isfile(predict_file):
        logging.info('Skip predict. {} already exists'.format(predict_file))
        return

    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token, 
        tokenizer.pad_token, tokenizer.mask_token, '.'])
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        # local_rank would not work for cross-node distributed training
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(get_rank(), 
                world_size) + op.splitext(predict_file)[1]

    model.eval()
    inputs_param = {'is_decode': True,
        'do_sample': False,
        'bos_token_id': cls_token_id,
        'pad_token_id': pad_token_id,
        'eos_token_ids': [sep_token_id],
        'mask_token_id': mask_token_id,
        # for adding od labels
        'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,

        # hyperparameters of beam search
        'max_length': args.max_gen_length,
        'num_beams': args.num_beams,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_keep_best": args.num_keep_best,
    }
    if args.use_cbs:
        if args.remove_bad_endings:
            bad_endings = ['a','an','the','in','for','at','of','with','before',
                    'after','on','upon','near','to','is','are','am']
            bad_ending_ids = tokenizer.convert_tokens_to_ids(bad_endings)
        else:
            bad_ending_ids = []
        inputs_param.update({'use_cbs': True,
            'min_constraints_to_satisfy': args.min_constraints_to_satisfy,
            'use_hypo': args.use_hypo,
            'decoding_constraint_flag': args.decoding_constraint,
            'bad_ending_ids': bad_ending_ids,
        })
    def gen_rows():
        time_meter = 0
        # restore existing results for long running inference tasks
        exist_key2pred = {}
        tmp_file = cache_file + '.tmp.copy'
        if op.isfile(tmp_file):
            with open(tmp_file, 'r') as fp:
                for line in fp:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        exist_key2pred[parts[0]] = parts[1]

        with torch.no_grad():
            for step, (img_keys, batch) in tqdm(enumerate(test_dataloader)):
                is_exist = True
                for k in img_keys:
                    if k not in exist_key2pred:
                        is_exist = False
                        break
                if is_exist:
                    for k in img_keys:
                        yield k, exist_key2pred[k]
                    continue
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3],
                    'masked_pos': batch[4],
                }
                if args.use_cbs:
                    inputs.update({
                        'fsm': batch[5],
                        'num_constraints': batch[6],
                    })
                inputs.update(inputs_param)
                tic = time.time()
                # captions, logprobs
                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    if isinstance(img_key, torch.Tensor):
                        img_key = img_key.item()
                    yield img_key, json.dumps(res)

        logging.info("Inference model computing time: {} seconds per batch".format(time_meter / (step+1)))

    tsv_writer(gen_rows(), cache_file)
    synchronize()
    #if world_size > 1:
        #torch.distributed.barrier()
    if world_size > 1 and is_main_process():
        cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
            op.splitext(predict_file)[1] for i in range(world_size)]
        concat_tsv_files(cache_files, predict_file)
        delete_tsv_files(cache_files)
        reorder_tsv_keys(predict_file, test_dataloader.dataset.image_keys, predict_file)
    synchronize()
    #if world_size > 1:
        #torch.distributed.barrier()


def restore_training_settings(args):
    ''' Restore args for inference and SCST training
    '''
    if args.do_train:
        if not args.scst:
            return args
        checkpoint = args.model_name_or_path
    else:
        assert args.do_test or args.do_eval
        checkpoint = args.eval_model_dir
    # restore training settings, check hasattr for backward compatibility
    train_args = torch.load(op.join(checkpoint, 'training_args.bin'))
    if hasattr(train_args, 'max_seq_a_length'):
        if hasattr(train_args, 'scst') and train_args.scst:
            max_od_labels_len = train_args.max_seq_length - train_args.max_gen_length
        else:
            max_od_labels_len = train_args.max_seq_length - train_args.max_seq_a_length
        max_seq_length = args.max_gen_length + max_od_labels_len
        args.max_seq_length = max_seq_length
        logging.warning('Override max_seq_length to {} = max_gen_length:{} + od_labels_len:{}'.format(
                max_seq_length, args.max_gen_length, max_od_labels_len))

    override_params = ['max_seq_a_length', 'do_lower_case', 'add_od_labels',
            'max_img_seq_length', 'img_feature_dim', 'no_sort_by_conf',
            'od_label_conf', 'unique_labels_on']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logging.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args


def check_arguments(args):
    if args.do_train:
        check_yaml_file(op.join(args.data_dir, args.train_yaml))
        #if args.evaluate_during_training:
            #check_yaml_file(op.join(args.data_dir, args.val_yaml))
        if args.effective_batch_size > 0:
            assert args.effective_batch_size % args.num_gpus == 0, (
                args.effective_batch_size,
                args.num_gpus
            )
            args.per_gpu_train_batch_size = int(args.effective_batch_size / args.num_gpus)
        if args.add_od_labels:
            assert args.max_seq_length > args.max_seq_a_length
        else:
            assert args.max_seq_length == args.max_seq_a_length
    if args.do_test:
        for test_yaml in args.test_yaml:
            check_yaml_file(op.join(args.data_dir, test_yaml))

def main(args):
    # Setup CUDA, GPU & distributed training
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    check_arguments(args)
    mkdir(args.output_dir)
    set_seed(args.seed, args.num_gpus)
    logging.info("Using {} GPUs".format(args.num_gpus))

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
    if args.do_train:
        config = config_class.from_pretrained(args.config_name if args.config_name else \
                args.model_name_or_path, num_labels=2, finetuning_task='image_captioning')
        if args.scst:
            # avoid using too much memory
            config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
                else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_type = 'frcnn'
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = 'classification'
        config.tie_weights = args.tie_weights
        config.freeze_embedding = args.freeze_embedding
        config.label_smoothing = args.label_smoothing
        config.drop_worst_ratio = args.drop_worst_ratio
        config.drop_worst_after = args.drop_worst_after
        # update model structure if specified in arguments
        update_params = ['img_feature_dim', 'num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
        model_structure_changed = [False] * len(update_params)
        for idx, param in enumerate(update_params):
            arg_param = getattr(args, param)
            # bert-base-uncased do not have img_feature_dim
            config_param = getattr(config, param) if hasattr(config, param) else -1
            if arg_param > 0 and arg_param != config_param:
                logging.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                setattr(config, param, arg_param)
                model_structure_changed[idx] = True
        if any(model_structure_changed):
            assert config.hidden_size % config.num_attention_heads == 0
            if args.load_partial_weights:
                # can load partial weights when changing layer only.
                assert not any(model_structure_changed[2:]), "Cannot load partial weights " \
                    "when any of ({}) is changed.".format(', '.join(update_params[2:]))
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
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        config.output_hidden_states = args.output_hidden_states
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logging.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info('Model total parameters: {}'.format(total_params))
    model.to(args.device)

    args = restore_training_settings(args)
    logging.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataloader = make_data_loader(args, args.train_yaml, tokenizer, 
            args.distributed, is_train=True)
        args.max_iter = len(train_dataloader)
        val_dataloader = None
        #if args.evaluate_during_training:
            #val_dataloader = make_data_loader(args, args.val_yaml, tokenizer, 
            #args.distributed, is_train=False)
        last_checkpoint = train(args, train_dataloader, val_dataloader, model, tokenizer)
        #if args.evaluate_during_training and is_main_process():
            #plot_validation_curve(op.join(args.output_dir, 'eval_logs.json'))
        return last_checkpoint

        # test the last checkpoint after training
        #if args.do_test:
            #for test_yaml in args.test_yaml:
                #logging.info("Evaluate on dataset: " + test_yaml)
                #test_dataloader = make_data_loader(args, test_yaml, 
                    #tokenizer, args.distributed, is_train=False)
                #evaluate(args, test_dataloader, model, tokenizer, last_checkpoint)

    elif args.do_test or args.do_eval:
        for test_yaml in args.test_yaml:
            logging.info("Evaluate on dataset: " + test_yaml)
            test_dataloader = make_data_loader(args, test_yaml,
                tokenizer, args.distributed, is_train=False)

            assert not args.do_eval
            predict_file = args.predict_file
            #predict_file = get_predict_file(checkpoint, args,
                    #test_dataloader.dataset.yaml_file)
            test(args, test_dataloader, model, tokenizer, predict_file)
            #else:
                #evaluate(args, test_dataloader, model, tokenizer, checkpoint)


def iter_caption_to_json(iter_caption, json_file):
    key_captions = [(key, json.loads(p)) for key, p in iter_caption]

    info = {
        'info': 'dummy',
        'licenses': 'dummy',
        'type': 'captions',
    }
    info['images'] = [{'file_name': k, 'id': k} for k, _ in key_captions]
    n = 0
    annotations = []
    for k, cs in key_captions:
        for c in cs:
            annotations.append({
                'image_id': k,
                'caption': c['caption'],
                'id': n
            })
            n += 1
    info['annotations'] = annotations
    from qd.qd_common import write_to_file
    write_to_file(json.dumps(info), json_file)

class MMaskCaptionPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
        })

    def append_predict_param(self, cc):
        super().append_predict_param(cc)
        #def get_predict_file(output_dir, args, data_yaml_file):
            #cc = ['pred']
            ## example data_yaml_file: datasets/coco_caption/test.yaml
            #data = data_yaml_file.split('/')[-2]
            #if data != 'coco_caption':
                #cc.append(data)
            #cc.append(op.splitext(op.basename(data_yaml_file))[0])
            #cc.append('beam{}'.format(args.num_beams))
            #cc.append('max{}'.format(args.max_gen_length))
            #if args.add_od_labels:
                #cc.append('wlabels')
            #if args.num_keep_best != 1:
                #cc.append('best{}'.format(args.num_keep_best))
            #if args.use_cbs:
                #cc.append('cbs{}'.format(args.min_constraints_to_satisfy))
                #if args.use_hypo:
                    #cc.append('hypo')
                #if args.decoding_constraint:
                    #cc.append('dc')
                #if args.remove_bad_endings:
                    #cc.append('rmbe')
            #if args.output_hidden_states:
                #cc.append('hidden')
            #return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))

    def get_train_data_loader(self, start_iter):
        pass

    def get_test_data_loader(self):
        pass

    def train(self):
        train_yaml = self.train_yaml
        if train_yaml is None:
            train_yaml = op.join(self.output_folder, 'train_yaml.yaml')
            if self.mpi_rank == 0:
                assert self.train_label_tsv is None, 'no longer supported'
                assert self.train_feature_tsv is None, 'no longer supported'
                convert_data_to_yaml(
                    data=self.data,
                    split='train',
                    yaml=train_yaml,
                    label=self.train_label_tsv,
                    feature=self.train_feature_tsv,
                    label_version=self.train_label_version,
                    feature_version=self.train_feature_version,
                )
            synchronize()

        logging.info('train_yaml = {}'.format(train_yaml))

        param = {
            'adam_epsilon': 1e-08,
            'add_od_labels': True,
            'cider_cached_tokens': 'coco_caption/gt/coco-train-words.p',
            'train_shuffle': True,
            'config_name': '',
            'data_dir': '.',
            'decoding_constraint': False,
            'device': 'cuda',
            'do_eval': False,
            'do_lower_case': True,
            'do_test': True,
            'do_train': True,
            'drop_out': 0.1,
            'drop_worst_after': 0,
            'drop_worst_ratio': 0,
            'effective_batch_size': self.effective_batch_size,
            'eval_model_dir': '',
            'evaluate_during_training': False,
            'freeze_embedding': False,
            'hidden_size': -1,
            'img_feature_dim': 1030,
            'intermediate_size': -1,
            'label_smoothing': 0.1,
            'learning_rate': self.base_lr,
            'length_penalty': 1,
            'load_partial_weights': False,
            'local_rank': self.mpi_local_rank,
            'logging_steps': 20,
            'num_gpus': self.mpi_size,
            'mask_prob': 0.15,
            'mask_type': 'seq2seq',
            'max_gen_length': 20,
            'max_grad_norm': 1.0,
            'max_img_seq_length': 50,
            'max_masked_tokens': 3,
            'max_seq_a_length': 40,
            'max_seq_length': 70,
            'min_constraints_to_satisfy': 2,
            'model_name_or_path': self.basemodel,
            'no_sort_by_conf': False,
            'num_attention_heads': -1,
            'num_beams': 1,
            'num_hidden_layers': -1,
            'num_keep_best': 1,
            'num_return_sequences': 1,
            'num_train_epochs': self.max_epoch,
            'num_workers': self.num_workers,
            'od_label_conf': 0.2,
            'on_memory': False,
            'output_dir': self.output_folder,
            'output_hidden_states': False,
            'per_gpu_eval_batch_size': 64,
            'per_gpu_train_batch_size': self.effective_batch_size // self.mpi_size,
            'remove_bad_endings': False,
            'repetition_penalty': 1,
            'save_steps': 3000,
            'sc_baseline_type': 'greedy',
            'sc_train_sample_n': 5,
            'scheduler': 'linear',
            'scst': False,
            'seed': 88,
            'temperature': 1,
            'test_yaml': [
            ],
            'tie_weights': False,
            'tokenizer_name': '',
            'top_k': 0,
            'top_p': 1,
            'train_yaml': train_yaml,
            'unique_labels_on': False,
            'use_cbs': False,
            'use_hypo': False,
            'val_yaml': '',
            'warmup_steps': 0,
            'weight_decay': self.weight_decay,
        }

        from pprint import pformat
        logging.info('param = \n{}'.format(pformat(param)))
        from qd.qd_common import make_namespace_by_dict
        args = make_namespace_by_dict(param)
        checkpoint_dir = main(args)
        last_model_link = self.get_last_model_link_file()
        from qd.qd_common import write_to_file
        write_to_file(
            op.relpath(checkpoint_dir, op.dirname(last_model_link)),
            last_model_link)

    def predict(self, model_path, predict_result_file):
        yaml_file = op.join(self.output_folder, 'test_yaml', '{}_{}'.format(self.test_data, self.test_split))
        if self.mpi_rank == 0:
            convert_data_to_yaml(
                self.test_data, self.test_split, yaml_file, is_train=False,
                label_version=self.train_label_version,
                feature_version=self.train_feature_version,
            )
        synchronize()
        param = {
            'adam_epsilon': 1e-08,
            'add_od_labels': True,
            'cider_cached_tokens': 'coco_caption/gt/coco-train-words.p',
            'config_name': '',
            'data_dir': '.',
            'decoding_constraint': False,
            'device': 'cuda',
            'do_eval': False,
            'do_lower_case': True,
            'do_test': True,
            'do_train': False,
            'caption_file_in_coco_format': '',
            'drop_out': 0.1,
            'drop_worst_after': 0,
            'drop_worst_ratio': 0,
            'eval_model_dir': model_path,
            'evaluate_during_training': False,
            'freeze_embedding': False,
            'hidden_size': -1,
            'img_feature_dim': 1030,
            'intermediate_size': -1,
            'label_smoothing': 0.1,
            'learning_rate': self.base_lr,
            'length_penalty': 1,
            'load_partial_weights': False,
            'local_rank': self.mpi_local_rank,
            'logging_steps': 20,
            'num_gpus': self.mpi_size,
            'mask_prob': 0.15,
            'mask_type': 'seq2seq',
            'max_gen_length': 20,
            'max_grad_norm': 1.0,
            'max_img_seq_length': 50,
            'max_masked_tokens': 3,
            'max_seq_a_length': 40,
            'max_seq_length': 70,
            'min_constraints_to_satisfy': 2,
            'model_name_or_path': self.basemodel,
            'no_sort_by_conf': False,
            'num_attention_heads': -1,
            'num_beams': 1,
            'num_hidden_layers': -1,
            'num_keep_best': 1,
            'num_return_sequences': 1,
            'num_train_epochs': self.max_epoch,
            'num_workers': self.num_workers,
            'od_label_conf': 0.2,
            'on_memory': False,
            'output_dir': self.output_folder,
            'output_hidden_states': False,
            'per_gpu_eval_batch_size': 64,
            'per_gpu_train_batch_size': self.effective_batch_size // self.mpi_size,
            'remove_bad_endings': False,
            'repetition_penalty': 1,
            'save_steps': 3000,
            'sc_baseline_type': 'greedy',
            'sc_train_sample_n': 5,
            'scheduler': 'linear',
            'scst': False,
            'seed': 88,
            'temperature': 1,
            'test_yaml': [
                yaml_file
            ],
            'predict_file': predict_result_file,
            'tie_weights': False,
            'tokenizer_name': '',
            'top_k': 0,
            'top_p': 1,
            'train_yaml': '',
            'unique_labels_on': False,
            'use_cbs': False,
            'use_hypo': False,
            'val_yaml': '',
            'warmup_steps': 0,
            'weight_decay': self.weight_decay,
        }

        from pprint import pformat
        logging.info('param = \n{}'.format(pformat(param)))
        from qd.qd_common import make_namespace_by_dict
        args = make_namespace_by_dict(param)
        main(args)

    def get_train_model(self):
        pass

    def get_test_model(self):
        pass

    def evaluate(self, predict_file, evaluate_file):
        if 'nocaps' not in self.test_data:
            from qd.tsv_io import TSVDataset
            dataset = TSVDataset(self.test_data)
            json_caption = op.join(
                dataset._data_root,
                self.test_split + '.caption_coco_format.json')
            if not op.isfile(json_caption):
                if self.mpi_rank == 0:
                    iter_caption_to_json(
                        dataset.iter_data(
                            self.test_split, 'caption'),
                        json_caption)
            result = evaluate_on_coco_caption(predict_file, json_caption, outfile=evaluate_file)
            logging.info('evaluation result: {}'.format(str(result)))
            logging.info('evaluation result saved to {}'.format(evaluate_file))
        else:
            raise NotImplementedError

    def get_optimizer(self, model):
        pass

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        pass

    def _get_test_normalize_module(self):
        return



