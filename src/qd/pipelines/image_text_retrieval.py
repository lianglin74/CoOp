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

from qd.qd_common import get_mpi_rank, get_mpi_size
import argparse
import os.path as op
import random, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from qd.qd_common import qd_tqdm as tqdm
import logging

from mmask.utils.miscellaneous import set_seed
from mmask.layers.bert import BertTokenizer, BertConfig
from mmask.layers.bert.modeling_bert import ImageBertForSequenceClassification

from mmask.solver import AdamW, WarmupLinearSchedule
from mmask.solver import WarmupConstantSchedule

from qd.data_layer.dataset import TSVSplitProperty
from qd.qd_common import decode_np
from qd.torch_common import synchronize


def decode_feature_to_tensor(row):
    all_feature = [decode_np(f['zlib_feature']) for f in json.loads(row[1])]
    feature = torch.tensor(all_feature).float()
    return feature

class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, tokenizer, args, data, split='train', is_train=True,
                 feature_version=None,
                 label_version=None):
        """
        tokenizer: tokenizer to process caption text.
        args: configureation parameters including max_seq_length, etc.
        split: used to infer the data used for training or testing. 
             All files are in .pt format of a dictionary with image keys and 
             image features (pytorch tensors), captions (list of str, support multiple
             captions per image), labels (list of dictionary or str of all labels),

        """
        from qd.qd_common import print_frame_info
        print_frame_info()
        super().__init__()
        self.data = data

        self.caption_tsv = TSVSplitProperty( self.data, split=split, t='caption')
        self.feature_tsv = TSVSplitProperty( self.data, split=split, t='feature', version=args.feature_version)

        if args.add_od_labels:
            self.label_tsv = TSVSplitProperty(args.data, split=split, t='label', version=args.label_version,)

        self.num_captions_per_img = len(json.loads(self.caption_tsv[0][1]))
        logging.info('num_captions_per_img = {}'.format(self.num_captions_per_img))

        self.is_train = is_train
        self.output_mode = args.output_mode
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_length
        self.max_img_seq_len = args.max_img_seq_length
        self.args = args

    def get_image_caption_index(self, index):
        # return img_idx to access features and [img_key, cap_idx] to access caption
        if not self.is_train and self.args.cross_image_eval:
            img_idx = index // (self.num_captions_per_img *
                                len(self.caption_tsv))
            cap_idx = index % (self.num_captions_per_img * len(self.caption_tsv))
            img_idx1 = cap_idx // self.num_captions_per_img
            cap_idx1 = cap_idx % self.num_captions_per_img
            return img_idx, [img_idx1, cap_idx1]
        img_idx = index // self.num_captions_per_img
        cap_idx = index % self.num_captions_per_img
        return img_idx, [img_idx, cap_idx]

    def get_label(self, index):
        # only used in cross image task
        img_idx = index // (self.num_captions_per_img *
                            len(self.caption_tsv))
        cap_idx = index % (self.num_captions_per_img * len(self.caption_tsv))
        img_idx1 = cap_idx // self.num_captions_per_img
        return 1 if img_idx == img_idx1 else 0

    def get_od_labels(self, img_idx, img_key=None):
        if self.args.add_od_labels:
            label_row = self.label_tsv[img_idx]
            if img_key is not None:
                assert label_row[0] == img_key
            od_labels = ' '.join([l['class'] for l in json.loads(label_row[1])])
            return od_labels

    def tensorize_example(self, text_a, img_feat, text_b=None, 
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.args.max_seq_length - 2:
            tokens_a = tokens_a[:(self.args.max_seq_length - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        seq_a_len = len(tokens)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, :]
            img_len = img_feat.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        # generate attention_mask
        att_mask_type = self.args.att_mask_type
        if att_mask_type == "CLR":
            attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                             [1] * img_len + [0] * img_padding_len
        else:
            # use 2D mask to represent the attention
            max_len = self.max_seq_len + self.max_img_seq_len
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # full attention of C-C, L-L, R-R
            c_start, c_end = 0, seq_a_len
            l_start, l_end = seq_a_len, seq_len
            r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
            attention_mask[c_start : c_end, c_start : c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            if att_mask_type == 'CL':
                attention_mask[c_start : c_end, l_start : l_end] = 1
                attention_mask[l_start : l_end, c_start : c_end] = 1
            elif att_mask_type == 'CR':
                attention_mask[c_start : c_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, c_start : c_end] = 1
            elif att_mask_type == 'LR':
                attention_mask[l_start : l_end, r_start : r_end] = 1
                attention_mask[r_start : r_end, l_start : l_end] = 1
            else:
                raise ValueError("Unsupported attention mask type {}".format(att_mask_type))

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return (input_ids, attention_mask, segment_ids, img_feat)

    def __getitem__(self, index):
        if self.is_train:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            feature_row = self.feature_tsv[img_idx]
            feature = decode_feature_to_tensor(feature_row)
            caption_row = self.caption_tsv[cap_idxs[0]]
            caption_infos = json.loads(caption_row[1])
            caption = caption_infos[cap_idxs[1]]['caption']

            od_labels = self.get_od_labels(img_idx)
            example = self.tensorize_example(caption, feature, text_b=od_labels)

            # select a negative pair
            neg_img_indexs = list(range(0, img_idx)) + list(range(img_idx + 1,
                                                                  len(self.caption_tsv)))
            img_idx_neg = random.choice(neg_img_indexs)
            if random.random() <= 0.5:
                # randomly select a negative caption from a different image.
                cap_idx_neg = random.randint(0, self.num_captions_per_img - 1)
                caption_neg = json.loads(self.caption_tsv[img_idx_neg][1])[cap_idx_neg]['caption']
                example_neg = self.tensorize_example(caption_neg, feature, text_b=od_labels)
            else:
                # randomly select a negative image
                feature_neg = decode_feature_to_tensor(self.feature_tsv[img_idx_neg])
                od_labels_neg = self.get_od_labels(img_idx_neg)
                example_neg = self.tensorize_example(caption, feature_neg, text_b=od_labels_neg)

            example_pair = tuple(list(example) + [1] + list(example_neg) + [0])
            return index, example_pair
        else:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            feature_row = self.feature_tsv[img_idx]
            feature = decode_feature_to_tensor(feature_row)
            caption_row = self.caption_tsv[cap_idxs[0]]
            caption = json.loads(caption_row[1])[cap_idxs[1]]['caption']
            od_labels = self.get_od_labels(img_idx)
            example = self.tensorize_example(caption, feature, text_b=od_labels)
            label = 1 if img_idx == cap_idxs[0] else 0
            return index, tuple(list(example) + [label])

    def __len__(self):
        if not self.is_train and self.args.cross_image_eval:
            return len(self.caption_tsv) ** 2 * self.num_captions_per_img
        return len(self.caption_tsv) * self.num_captions_per_img


def compute_score_with_logits(logits, labels):
    if logits.shape[1] > 1:
        logits = torch.max(logits, 1)[1].data # argmax
        scores = logits == labels 
    else:
        scores = torch.zeros_like(labels).cuda()
        for i, (logit, label) in enumerate(zip(logits, labels)):
            logit_ = torch.sigmoid(logit)
            if (logit_ >= 0.5 and label == 1) or (logit_ < 0.5 and label == 0):
                scores[i] = 1
    return scores

def compute_ranks(labels, results, num_image, num_captions_per_img):
    #labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    similarities = np.array([results[i] for i in range(len(results))])
    #if dataset.has_caption_indexs:
        #num_captions_per_img = dataset.num_captions_per_img
    #else:
    num_captions_per_img = num_image * num_captions_per_img
    labels = np.reshape(labels, [-1, num_captions_per_img])
    similarities = np.reshape(similarities, [-1, num_captions_per_img])
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)

    #if not dataset.has_caption_indexs:
    labels = np.swapaxes(labels, 0, 1)
    similarities = np.swapaxes(similarities, 0, 1)
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks


def save_checkpoint(model, tokenizer, args, global_step, checkpoint_dir):
    from qd.qd_common import ensure_directory
    ensure_directory(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    save_num = 0
    while (save_num < 10):
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            tokenizer.save_pretrained(checkpoint_dir)
            logging.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            save_num += 1
    if save_num == 10:
        logging.info("Failed to save checkpoint after 10 trails.")
    return checkpoint_dir

def train(self, args, train_dataset, val_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size
    from qd.data_layer.samplers import DistributedSampler
    if self.max_epoch is not None and self.max_epoch > 0:
        t_total = len(train_dataset) // args.gradient_accumulation_steps \
                * self.max_epoch // (args.train_batch_size * self.mpi_size)
    else:
        t_total = self.max_iter

    train_sampler = DistributedSampler(
        train_dataset,
        length_divisible=args.train_batch_size)
    import torch
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        train_sampler, args.train_batch_size, drop_last=False
    )
    import qd.data_layer.samplers as samplers
    batch_sampler = samplers.IterationBasedBatchSampler(
        batch_sampler, t_total, start_iter=0,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=self.num_workers,
        pin_memory=True,
    )
    assert len(train_dataloader) == t_total

    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not \
            any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler == "constant":
        scheduler = WarmupConstantSchedule(
                optimizer, warmup_steps=args.warmup_steps)
    elif args.scheduler == "linear":
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        raise ValueError("Unknown scheduler type: {}".format(args.scheduler))

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logging.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    model.zero_grad()
    #log_json = []
    #best_score = 0
    from qd.logger import MetricLogger
    meters = MetricLogger(delimiter="  ")
    end = time.time()
    log_start = time.time()
    model.train()
    for iteration, (_, batch) in enumerate(train_dataloader):
        data_time = time.time() - end
        batch = tuple(t.to(self.device) for t in batch)
        inputs = {
            'input_ids':      torch.cat((batch[0], batch[5]), dim=0),
            'attention_mask': torch.cat((batch[1], batch[6]), dim=0),
            'token_type_ids': torch.cat((batch[2], batch[7]), dim=0),
            'img_feats':      torch.cat((batch[3], batch[8]), dim=0),
            'labels':         torch.cat((batch[4], batch[9]), dim=0)
        }
        outputs = model(**inputs)
        loss, logits = outputs[:2]
        loss = loss.mean() # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        batch_score = compute_score_with_logits(logits, inputs['labels']).sum()
        batch_acc = batch_score.item() / (args.train_batch_size * 2)
        meters.update(loss=loss, batch_acc=batch_acc)
        batch_time = time.time() - end
        end = time.time()
        if iteration > 5:
            # we will skip the first few iterations since the time cost
            # evaluation for those are not good
            meters.update(time=batch_time, data=data_time)
        scheduler.step()
        optimizer.step()
        model.zero_grad()
        if iteration % args.logging_steps == 0:
            speed = self.mpi_size * args.logging_steps * len(batch[0]) / (
                time.time() - log_start)

            if hasattr(meters, 'time'):
                eta_seconds = meters.time.global_avg * (t_total - iteration)
                import datetime
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
        if (args.save_steps > 0 and (iteration + 1) % args.save_steps == 0):
            if self.mpi_rank == 0:
                out_dir = op.join(args.output_dir, 'snapshot', 'model_iter_{:07d}'.format(iteration + 1))
                save_checkpoint(model, tokenizer, args, iteration + 1,
                                out_dir)
    out_dir = op.join(args.output_dir, 'snapshot', 'model_iter_{:07d}'.format(t_total))
    if self.mpi_rank == 0:
        out_dir = save_checkpoint(model, tokenizer, args, t_total, out_dir)
    synchronize()
    return out_dir

def iter_test(self, args, model, eval_dataset):
    #eval_sampler = SequentialSampler(eval_dataset)
    from qd.data_layer.samplers import DistributedSampler
    eval_sampler = DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
            batch_size=self.test_batch_size, num_workers=self.num_workers)

    logging.info("Num examples = {}".format(len(eval_dataset)))
    logging.info("Evaluation batch size = {}".format(self.test_batch_size))
    model.eval()
    softmax = nn.Softmax(dim=1)
    for indexs, batch in tqdm(eval_dataloader):
        batch = tuple(t.to(self.device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats':      batch[3],
                'labels':         batch[4]
            }
            _, logits = model(**inputs)[:2]
            if args.num_labels == 2:
                probs = softmax(logits)
                result = probs[:, 1] # the confidence to be a matched pair
            else:
                result = logits
            result = [_.to(torch.device("cpu")) for _ in result]

            for idx, res in zip(indexs, result):
                yield idx.item(), res.item()

def evaluate(labels, test_results, num_image, num_captions_per_img):
    i2t_ranks, t2i_ranks = compute_ranks(labels, test_results, num_image,
                                         num_captions_per_img)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logging.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logging.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result

def restore_training_settings(args):
    assert not args.do_train and (args.do_test or args.do_eval)
    train_args = torch.load(op.join(args.eval_model_dir, 'training_args.bin'))
    override_params = ['do_lower_case', 'img_feature_type', 'max_seq_length',
            'max_img_seq_length', 'add_od_labels']
    for param in override_params:
        if hasattr(train_args, param):
            train_v = getattr(train_args, param)
            test_v = getattr(args, param)
            if train_v != test_v:
                logging.warning('Override {} with train args: {} -> {}'.format(param,
                    test_v, train_v))
                setattr(args, param, train_v)
    return args

def main_retrieval(self, args):
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)
    logging.info('output_mode: {}, #Labels: {}'.format(args.output_mode, args.num_labels))

    config_class, tokenizer_class = BertConfig, BertTokenizer
    model_class = ImageBertForSequenceClassification
    if args.do_train:
        config = config_class.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=args.num_labels, finetuning_task='ir')
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name \
            else args.model_name_or_path, do_lower_case=args.do_lower_case)
        config.img_feature_dim = args.img_feature_dim
        config.img_feature_type = args.img_feature_type
        config.hidden_dropout_prob = args.drop_out
        config.loss_type = args.loss_type
        model = model_class.from_pretrained(args.model_name_or_path,
            from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    else:
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        config = config_class.from_pretrained(checkpoint)
        tokenizer = tokenizer_class.from_pretrained(checkpoint)
        logging.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)

    logging.info("Training/evaluation parameters %s", args)
    model = self._data_parallel_wrap(model)
    if args.do_train:
        train_dataset = RetrievalDataset(
            tokenizer, args,
            self.data, 'train', is_train=True,
            feature_version=self.train_feature_version,
            label_version=self.train_label_version,
        )
        val_dataset = None
        out_dir = train(self, args, train_dataset, val_dataset, model, tokenizer)
        return out_dir

    # inference and evaluation
    if args.do_test or args.do_eval:
        args = restore_training_settings(args)
        test_dataset = RetrievalDataset(
            tokenizer, args,
            self.test_data, self.test_split,
            is_train=False,
            feature_version=self.train_feature_version,
            label_version=self.train_label_version,
        )
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        logging.info("Evaluate the following checkpoint: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint, config=config)
        model.to(self.device)

        pred_file = args.pred_file
        if op.isfile(pred_file) and not self.force_predict:
            logging.info("Prediction file exist, skip inference.")
        else:
            curr_pred_file = pred_file
            if self.mpi_size > 1:
                curr_pred_file = op.splitext(pred_file)[0] + '_{}_{}'.format(
                    self.mpi_rank,
                    self.mpi_size) + op.splitext(pred_file)[1]
            from qd.tsv_io import tsv_writer
            tsv_writer(iter_test(self, args, model, test_dataset), curr_pred_file)
            synchronize()
            if self.mpi_size > 1:
                if self.mpi_rank == 0:
                    cache_files = [op.splitext(pred_file)[0] + '_{}_{}'.format(
                        i, self.mpi_size) + \
                        op.splitext(pred_file)[1] for i in range(self.mpi_size)]
                    from qd.process_tsv import concat_tsv_files
                    from qd.process_tsv import delete_tsv_files
                    concat_tsv_files(cache_files, pred_file)
                    delete_tsv_files(cache_files)
                    # no need to reorder the prediction file
            synchronize()

        if args.do_eval:
            from qd.tsv_io import tsv_reader
            test_result = dict((int(i), float(c)) for i, c in tsv_reader(pred_file))
            eval_result = evaluate(test_dataset, test_result)
            return eval_result

class ImageTextRetrievalPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'evaluate_method': 'ir_acc',
        })

    def append_predict_param(self, cc):
        pass

    def get_train_data_loader(self, start_iter):
        pass

    def get_test_data_loader(self):
        pass

    def train(self):
        param = {
            'data': self.data,

            'feature_version': self.train_feature_version,
            'label_version': self.train_label_version,

            'model_name_or_path': self.basemodel,
            'output_dir': self.output_folder,
            'loss_type': 'sfmx',
            'config_name': '',
            'tokenizer_name': '',
            'max_seq_length': 70,
            'do_train': True,
            'do_test': False,
            'do_eval': False,
            'test_split': 'test',
            'cross_image_eval': False,
            'add_od_labels': True,
            'att_mask_type': 'CLR',
            'do_lower_case': True,
            'drop_out': 0.1,
            'max_img_seq_length': 50,
            'img_feature_dim': 2054,
            'img_feature_type': 'frcnn',
            'per_gpu_train_batch_size': self.effective_batch_size // self.mpi_size,
            'output_mode': 'classification',
            'num_labels': 2,
            'gradient_accumulation_steps': 1,
            'learning_rate': self.base_lr,
            'weight_decay': self.weight_decay,
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,
            'warmup_steps': 0,
            'scheduler': 'linear',
            'logging_steps': self.log_step,
            'save_steps': -1,
            'eval_model_dir': '',
            'device': '',
            'seed': 88,
        }
        from pprint import pformat
        logging.info('param = \n{}'.format(pformat(param)))
        from qd.qd_common import make_namespace_by_dict
        args = make_namespace_by_dict(param)
        checkpoint_dir = main_retrieval(self, args)
        last_model_link = self.get_last_model_link_file()
        from qd.qd_common import write_to_file
        write_to_file(
            op.relpath(checkpoint_dir, op.dirname(last_model_link)),
            last_model_link)

    def get_train_model(self):
        pass

    def get_test_model(self):
        pass

    def get_optimizer(self, model):
        pass

    def predict(self, model_path, predict_result_file):
        param = {
            'data': self.test_data,
            'test_split': self.test_split,

            'feature_version': self.train_feature_version,
            'label_version': self.train_label_version,

            'model_name_or_path': model_path,
            'output_dir': self.output_folder,
            'loss_type': 'sfmx',
            'config_name': '',
            'tokenizer_name': '',
            'max_seq_length': 70,
            'do_train': False,
            'do_test': True,
            'do_eval': False,
            'cross_image_eval': True,
            'add_od_labels': True,
            'att_mask_type': 'CLR',
            'do_lower_case': True,
            'drop_out': 0.1,
            'max_img_seq_length': 50,
            'img_feature_dim': 2054,
            'img_feature_type': 'frcnn',
            'output_mode': 'classification',
            'num_labels': 2,
            'gradient_accumulation_steps': 1,
            'learning_rate': self.base_lr,
            'weight_decay': self.weight_decay,
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,
            'warmup_steps': 0,
            'scheduler': 'linear',
            'max_steps': -1,
            'logging_steps': 20,
            'save_steps': -1,
            'eval_model_dir': model_path,
            'device': '',
            'seed': 88,
            'pred_file': predict_result_file,
        }
        from pprint import pformat
        logging.info('param = \n{}'.format(pformat(param)))
        from qd.qd_common import make_namespace_by_dict
        args = make_namespace_by_dict(param)
        main_retrieval(self, args)

    def evaluate(self, predict_file, evaluate_file):
        from qd.tsv_io import tsv_reader
        test_result = dict((int(i), float(c)) for i, c in tsv_reader(predict_file))

        caption_tsv = TSVSplitProperty(self.test_data, self.test_split, 'caption')
        num_captions_per_img = len(json.loads(caption_tsv[0][-1]))

        def get_label(index):
            # only used in cross image task
            img_idx = index // (num_captions_per_img * len(caption_tsv))
            cap_idx = index % (num_captions_per_img * len(caption_tsv))
            img_idx1 = cap_idx // num_captions_per_img
            return 1 if img_idx == img_idx1 else 0

        labels = np.array([get_label(i) for i in range(
            len(caption_tsv) ** 2 * num_captions_per_img)])
        eval_result = evaluate(labels, test_result, len(caption_tsv),
                               num_captions_per_img)
        from qd.qd_common import write_to_yaml_file
        write_to_yaml_file(eval_result, evaluate_file)
        from pprint import pformat
        logging.info(pformat(eval_result))

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        pass

    def _get_test_normalize_module(self):
        return

