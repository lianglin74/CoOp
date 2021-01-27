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
from qd.qd_common import func_retry_agent

from qd.qd_common import get_mpi_rank, get_mpi_size
import argparse
import os.path as op
import random, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from qd.qd_common import qd_tqdm as tqdm

from qd.torch_common import set_seed
from qd.mask.layers.bert import BertTokenizer, BertConfig
from qd.layers.bert.modeling_bert import ImageBertForSequenceClassification

from qd.mask.solver import AdamW, WarmupLinearSchedule
from qd.mask.solver import WarmupConstantSchedule

from torch.utils.data.distributed import DistributedSampler

from qd.data_layer.dataset import TSVSplitProperty
from qd.qd_common import decode_np
from qd.torch_common import synchronize


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class InputInstance(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, score=None, img_key=None, q_id=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """

        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.score = score
        self.img_key = img_key
        self.q_id = q_id

def detect_feature_dim(feat_tsv):
    for row in feat_tsv:
        rects = json.loads(row[-1])
        for r in rects:
            return len(decode_np(r['zlib_feature']))

class VQATSVDataset(Dataset):
    """ VQA Dataset """

    def __init__(self, data, split,
                 feature_version,
                 label_version,
                 tokenizer,
                 max_seq_length,
                 max_img_seq_length,
                 img_feature_dim,
                 caption_version=None,
                 od_label_conf=0.,
                 ):
        super().__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()

        logging.info('loading features')
        self.max_seq_length = max_seq_length
        self.max_img_seq_length = max_img_seq_length
        self.feat_tsv = TSVSplitProperty(
            data,
            split, 'feature',
            version=feature_version)
        self.hw_tsv = TSVSplitProperty(
            data,
            split, 'hw')
        self.cap_tsv = TSVSplitProperty(data, split, 'caption',
                                        version=caption_version)
        self.label_tsv = TSVSplitProperty(
            data, split, 'label', version=label_version)

        num_cap_tsv = TSVSplitProperty(data, split, 'num_caption',
                                       version=caption_version,
                                       )
        num_caps = [int(n) for _, n in num_cap_tsv]
        self.caption_linelist = [(idx_img, idx_cap) for idx_img, n in enumerate(num_caps) for idx_cap in range(n)]

        self.tokenizer = tokenizer

        from qd.tsv_io import TSVDataset
        dataset = TSVDataset(data)
        from qd.qd_common import load_list_file
        answers = load_list_file(dataset.get_txt('answermap'))
        self.answer2idx = {a: i for i, a in enumerate(answers)}
        self.answers = answers
        self.od_label_conf = od_label_conf
        # with 6 spatial dimension
        self.img_feature_dim = img_feature_dim

    def get_spatial_features(self, feat_info, img_idx):
        key, str_hw = self.hw_tsv[img_idx]
        img_height, img_width = map(int, str_hw.split(' '))
        spatial_feats = []
        for f in feat_info:
            # spatial features follow OSCAR pre-processing
            box = f['rect'] # xyxy
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            scaled_width = box_width / img_width
            scaled_height = box_height / img_height
            scaled_x = box[0] / img_width
            scaled_y = box[1] / img_height
            spatial_feat = np.array([scaled_x, scaled_y, scaled_x + scaled_width,
                scaled_y + scaled_height, scaled_width, scaled_height], dtype=np.float32)
            spatial_feats.append(spatial_feat)
        return key, spatial_feats

    def get_example_info(self, idx_image, idx_question):
        key, str_cap = self.cap_tsv[idx_image]
        cap = json.loads(str_cap)[idx_question]
        label_key, str_tags = self.label_tsv[idx_image]
        tags = ' '.join([r['class'] for r in json.loads(str_tags)
                         if r['conf'] >= self.od_label_conf])
        tags = tags.replace(';', ' ').strip()
        guid = "%s-%s" % (idx_image, idx_question)
        text_a = cap['question']
        text_b = tags
        # during test, we don't have these two fields.
        label = cap.get('answers')
        if label is not None:
            label = [self.answer2idx.get(l, -1)  for l in label]
        score = cap.get('confs')
        # during test, we need this for result submission
        q_id = cap.get('question_id', 0)
        example = InputInstance(guid=guid, text_a=text_a, text_b=text_b, label=label, score=score, img_key=key, q_id=q_id)
        return {
            'key': key,
            'example': example,
        }

    def get_feature_info(self, idx_image):
        row = self.feat_tsv[idx_image]
        feat_info = json.loads(row[1])
        if len(feat_info) == 0:
            return {'feat': torch.zeros((0, self.img_feature_dim)),
                    'key': row[0]}
        if all('feature' in f for f in feat_info):
            import base64
            feats = [np.frombuffer(base64.b64decode(f['feature']), np.float32) for f in feat_info]
        else:
            feats = [decode_np(f['zlib_feature']).astype(np.float32) for f in feat_info]
        key, spatial_feats = self.get_spatial_features(feat_info, idx_image)
        assert key == row[0]
        return {'feat': torch.Tensor(np.concatenate((feats, spatial_feats), 1)),
                'key': row[0]}

    def get_answers(self, index):
        # by evaluation code
        idx_img, idx_question = self.caption_linelist[index]
        key, str_cap = self.cap_tsv[idx_img]
        cap = json.loads(str_cap)[idx_question]
        return cap.get('answers')

    def __getitem__(self, index):
        idx_img, idx_cap = self.caption_linelist[index]
        entry = self.get_example_info(idx_img, idx_cap)

        pad_token=0
        sequence_a_segment_id=0
        sequence_b_segment_id=1
        mask_padding_with_zero=True

        cls_token_at_end=False
        cls_token=self.tokenizer.cls_token
        sep_token=self.tokenizer.sep_token
        cls_token_segment_id= 0
        pad_on_left=False # pad on the left for xlnet
        pad_token_segment_id = 0
        example = entry['example']

        tokens_a = self.tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        feat_info = self.get_feature_info(idx_img)
        img_feat = feat_info['feat']
        assert feat_info['key'] == entry['key']

        if img_feat.shape[0] > self.max_img_seq_length:
            img_feat = img_feat[0:self.max_img_seq_length, ]
            if self.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
        else:
            if self.max_img_seq_length > 0:
                input_mask = input_mask + [1 if mask_padding_with_zero else 0] * img_feat.shape[0]
                # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
            padding_matrix = torch.zeros((self.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)
            if self.max_img_seq_length > 0:
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_matrix.shape[0])
                # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]

        if (example.label is None):
            label_id = [0]
            score = [0]
        elif len(example.label) == 0:
            label_id = [0]
            score = [0]
        else:
            #label_id = [self.label_map[l] for l in example.label]
            label_id = example.label
            score = example.score

        new_scores = target_tensor(len(self.answer2idx), label_id, score)

        net_in = (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_mask, dtype=torch.long),
                torch.tensor(segment_ids, dtype=torch.long),
                torch.tensor([label_id[0]], dtype=torch.long),
                torch.tensor(new_scores, dtype=torch.float),
                img_feat,
                torch.tensor([example.q_id], dtype=torch.long))
        return {
            'index': index,
            'net_in': net_in,
            'idx_img': idx_img,
            'idx_cap': idx_cap,
        }

    def __len__(self):
        return len(self.caption_linelist)

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def save_checkpoint(model, tokenizer, args, output_dir):
    from qd.qd_common import ensure_directory
    ensure_directory(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model

    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    tokenizer.save_pretrained(output_dir)

def train(self, args, train_dataset, model, tokenizer):
    from qd.data_layer.samplers import DistributedSampler
    if self.max_epoch is not None and self.max_epoch > 0:
        t_total = int(len(train_dataset) * self.max_epoch //
                      (self.effective_batch_size))
    else:
        t_total = self.max_iter
    bs_each_gpu = self.effective_batch_size // self.mpi_size
    train_sampler = DistributedSampler(
        train_dataset,
        length_divisible=self.effective_batch_size)
    import torch
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        train_sampler, bs_each_gpu, drop_last=False
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

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.base_lr, eps=args.adam_epsilon)

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    model = self._data_parallel_wrap(model)

    from qd.logger import MetricLogger
    meters = MetricLogger(delimiter="  ")

    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Total optimization steps = %d", t_total)

    #global_step = 0
    #tr_loss = 0.0
    model.zero_grad()
    log_start = time.time()

    end = time.time()
    model.train()
    for iteration, loaded in enumerate(train_dataloader):
        data_time = time.time() - end
        batch = loaded['net_in']
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'labels':         batch[4],
                  'img_feats':      batch[5]}
        outputs = model(**inputs)

        loss, logits = outputs[:2]
        loss = loss.mean()

        meters.update(loss=loss)
        batch_time = time.time() - end
        end = time.time()
        if iteration > 5:
            # we will skip the first few iterations since the time cost
            # evaluation for those are not good
            meters.update(time=batch_time, data=data_time)

        batch_score = compute_score_with_logits(logits, batch[4]).sum() / len(logits)
        meters.update(acc=batch_score)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scheduler.step()  # Update learning rate schedule
        optimizer.step()
        model.zero_grad()
        if iteration % self.log_step == 0:
            speed = self.mpi_size * self.log_step * len(batch[0]) / (
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
                out_dir = op.join(self.output_folder, 'snapshot', 'model_iter_{:07d}'.format(iteration + 1))
                func_retry_agent({'retry_times': 5, 'throw_if_fail': False},
                                 save_checkpoint, model, tokenizer, args, out_dir)

    out_dir = op.join(self.output_folder, 'snapshot', 'model_iter_{:07d}'.format(t_total))
    if self.mpi_rank == 0:
        func_retry_agent({'retry_times': -1}, save_checkpoint, model, tokenizer, args, out_dir)
    synchronize()
    return out_dir

def iter_test(self, args, model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    for loaded in tqdm(eval_dataloader):
        batch = loaded['net_in']
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels':         None,
                      'img_feats':      batch[5]}
            outputs = model(**inputs)
            logits = outputs[0]

            val, max_idx = logits.max(1)

        assert len(max_idx) == len(loaded['index'])
        for i, idx in enumerate(loaded['index']):
            result = {}
            result['question_id'] = batch[6][i].item()
            result['answer'] = eval_dataset.answers[max_idx[i].item()]
            from qd.qd_common import json_dump
            yield int(idx), json_dump(result)

def target_tensor(len, labels, scores):
    """ create the target by labels and scores """
    target = [0]*len
    for id, l in enumerate(labels):
        target[l] = scores[id]

    return target

def main_vqa(self, args):
    from qd.torch_common import ensure_init_process_group
    ensure_init_process_group()
    device = torch.device("cuda")
    args.n_gpu = self.mpi_size

    args.device = device

    set_seed(88, self.mpi_size)

    from qd.tsv_io import TSVDataset
    from qd.qd_common import load_list_file
    dataset = TSVDataset(self.data)
    label_list = load_list_file(dataset.get_txt('answermap'))
    num_labels = len(label_list)

    synchronize()

    config_class, model_class, tokenizer_class = BertConfig, ImageBertForSequenceClassification, BertTokenizer
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, finetuning_task='vqa_text',
    )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

    # discrete code
    config.img_feature_dim = self.img_feature_dim
    config.img_feature_type = 'faster_r-cnn'
    config.code_voc = 512
    config.hidden_dropout_prob = self.drop_out
    config.loss_type = self.loss_type
    config.classifier = args.classifier
    config.cls_hidden_scale = args.cls_hidden_scale
    if self.prior_prob is not None:
        config.prior_prob = self.prior_prob

    config.use_img_layernorm = self.use_img_layernorm
    config.img_layer_norm_eps = 1e-5

    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    logging.info(model)

    synchronize()
    model.to(args.device)
    logging.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataset = VQATSVDataset(
            data=self.data, split='train',
            feature_version=self.train_feature_version,
            label_version=self.train_label_version,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            max_img_seq_length=args.max_img_seq_length,
            img_feature_dim=self.img_feature_dim,
            caption_version=self.train_caption_version,
            od_label_conf=self.od_label_conf,
        )
        return train(self, args, train_dataset, model, tokenizer)

    if args.do_test:
        test_dataset = VQATSVDataset(
            data=self.test_data, split=self.test_split, tokenizer=tokenizer,
            feature_version=self.train_feature_version,
            label_version=self.train_label_version,
            max_seq_length=args.max_seq_length,
            max_img_seq_length=args.max_img_seq_length,
            img_feature_dim=self.img_feature_dim,
            od_label_conf=self.od_label_conf,
        )
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
                    before_reorder = pred_file + '.before.reorder.tsv'
                    concat_tsv_files(cache_files, before_reorder)
                    delete_tsv_files(cache_files)
                    ordered_keys = [str(i) for i in range(len(test_dataset))]
                    from qd.tsv_io import reorder_tsv_keys
                    reorder_tsv_keys(before_reorder, ordered_keys, pred_file)
            synchronize()

class VQAPipeline(MaskClassificationPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            # only used to generate evaluate file name
            'evaluate_method': 'vqa_acc',
            'loss_type': 'bce',
            'drop_out': 0.3,
            'od_label_conf': 0.,
            # set od_label_conf = 1.1 to ignore od labels
        })

    def append_predict_param(self, cc):
        pass

    def get_train_data_loader(self, start_iter):
        pass

    def get_test_data_loader(self):
        pass

    def train(self):
        param = {
            'adam_epsilon': 1e-08,
            'classifier': 'linear',
            'cls_hidden_scale': 3,
            'config_name': '',
            'do_lower_case': True,
            'do_test': False,
            'do_train': True,
            'max_grad_norm': 1.0,
            'max_img_seq_length': 50,
            'max_seq_length': 128,
            'model_name_or_path': self.basemodel,
            'per_gpu_eval_batch_size': self.test_batch_size,
            'save_steps': 5000,
            'scheduler': 'linear',
            'task_name': 'vqa_text',
            'tokenizer_name': '',
            'warmup_steps': 0,
            'weight_decay': 0.05,
        }
        from pprint import pformat
        logging.info('param = \n{}'.format(pformat(param)))
        from qd.qd_common import make_namespace_by_dict
        args = make_namespace_by_dict(param)
        checkpoint_dir = main_vqa(self, args)
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
            'adam_epsilon': 1e-08,
            'classifier': 'linear',
            'cls_hidden_scale': 3,
            'config_name': '',
            'do_lower_case': True,
            'do_test': True,
            'do_train': False,
            'img_feature_dim': self.img_feature_dim,
            'max_grad_norm': 1.0,
            'max_img_seq_length': 50,
            'max_seq_length': 128,
            'model_name_or_path': model_path,
            'per_gpu_eval_batch_size': self.test_batch_size,
            'save_steps': 5000,
            'scheduler': 'linear',
            'task_name': 'vqa_text',
            'tokenizer_name': '',
            'pred_file': predict_result_file,
            'warmup_steps': 0,
            'weight_decay': 0.05,
        }
        from pprint import pformat
        logging.info('param = \n{}'.format(pformat(param)))
        from qd.qd_common import make_namespace_by_dict
        args = make_namespace_by_dict(param)
        main_vqa(self, args)

    def evaluate(self, predict_file, evaluate_file):
        if self.test_split in ['test_dev', 'test_std']:
            # we only convert the pred to json and then we should manually
            # upload the json file
            out_file = predict_file + '.server.json'
            from qd.tsv_io import tsv_reader
            result = [json.loads(s) for _, s in tsv_reader(predict_file)]
            from qd.qd_common import write_to_file, json_dump
            write_to_file(json_dump(result), out_file)
        else:
            return self.evaluate_acc(self, predict_file, evaluate_file)

    def evaluate_acc(self, predict_file, evaluate_file):
        from qd.tsv_io import TSVDataset
        dataset = TSVDataset(self.test_data)
        all_qa = [json.loads(s_cap) for key, s_cap in dataset.iter_data(
            self.test_split,
            'caption')]
        num_caps = [len(qa) for qa in all_qa]
        caption_linelist = [(idx_img, idx_cap) for idx_img, n in enumerate(num_caps) for idx_cap in range(n)]
        correctness = []
        from qd.tsv_io import tsv_reader
        for index, s_pred in tqdm(tsv_reader(predict_file)):
            pred = json.loads(s_pred)['answer']
            index = int(index)
            idx_img, idx_cap = caption_linelist[index]
            gt = all_qa[idx_img][idx_cap]['answers']
            if len(gt) == 0:
                # this case, we ignore it to follow the released code in oscar
                continue
            if pred in gt:
                idx = gt.index(pred)
                correctness.append(all_qa[idx_img][idx_cap]['confs'][idx])
            else:
                correctness.append(0.)
        acc = torch.tensor(correctness).mean()
        from qd.qd_common import write_to_yaml_file
        logging.info(acc)
        write_to_yaml_file({'acc': float(acc)}, evaluate_file)

    def get_lr_scheduler(self, optimizer, last_epoch=-1):
        pass

    def _get_test_normalize_module(self):
        return

