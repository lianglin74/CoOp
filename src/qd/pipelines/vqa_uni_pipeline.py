from qd.qd_common import qd_tqdm as tqdm
from qd.qd_common import json_dump
from qd.tsv_io import TSVDataset
from qd.qd_common import load_list_file
from qd.mask.layers.bert import BertForImageCaptioning
from qd.qd_common import execute_func
from qd.mask.layers.bert import BertTokenizer, BertConfig
from qd.mask.layers.bert.modeling_bert import ImageBertForSequenceClassification
import os.path as op
import json
import logging
import torch
from qd.data_layer.builder import collate_fn
from torchvision.transforms import transforms
from qd.data_layer.transform import (
    LoadLabel,
    LoadHW,
    LoadFeature,
    LoadImage,
    LoadCaption,
    IdentifyTextAB,
    RandomPairNegative,
    TokenizeTransform,
    NoChange,
    PrepareLabel,
    RemoveUselessKeys,
    RenameKey,
    AppendDummyFeature,
)
from qd.data_layer.dataset import CaptionIdxTSVDataset, ImageIdxTSVDataset
from qd.pipelines.uni_pipeline import UniPipeline
from torch import nn


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

def target_tensor(len, labels, scores):
    """ create the target by labels and scores """
    target = [0]*len
    for id, l in enumerate(labels):
        target[l] = scores[id]

    return target

class AggregateInputVQA(object):
    def __init__(self, answers, tokenizer, max_seq_length,
                 max_img_seq_length,
                 img_feature_dim,
                 od_label_conf=0.,
                 pad_to_max=True,
                 ):
        self.answer2idx = {a: i for i, a in enumerate(answers)}
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_img_seq_length = max_img_seq_length
        self.img_feature_dim = img_feature_dim
        self.od_label_conf = od_label_conf
        self.pad_to_max = pad_to_max

    def __call__(self, data):
        idx_img, idx_cap = data['idx_img'], data['idx_cap']
        cap = data['caption']
        tags = data['label']
        tags = ' '.join([r['class'] for r in tags
                         if r['conf'] >= self.od_label_conf])
        tags = tags.replace(';', ' ').strip()
        guid = "%s-%s" % (idx_img, idx_cap)
        text_a = cap['question']
        text_b = tags
        # during test, we don't have these two fields.
        label = cap.get('answers')
        if label is not None:
            label = [self.answer2idx.get(l, -1)  for l in label]
        score = cap.get('confs')
        # during test, we need this for result submission
        q_id = cap.get('question_id', 0)
        key = data['key']
        example = InputInstance(guid=guid, text_a=text_a, text_b=text_b,
                                label=label, score=score, img_key=key, q_id=q_id)
        entry = {
            'key': key,
            'example': example,
        }

        pad_token=0
        sequence_a_segment_id=0
        sequence_b_segment_id=1

        cls_token_at_end=False
        cls_token=self.tokenizer.cls_token
        sep_token=self.tokenizer.sep_token
        cls_token_segment_id= 0
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
        input_mask = [1] * len(input_ids)

        if self.pad_to_max:
            # Zero-pad up to the sequence length.
            padding_length = self.max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

        if 'img_feats' in data:
            # we always do the padding for the image features, no matter what
            # the value of self.pad_to_max is
            img_feat = data['img_feats']
            if img_feat.shape[0] > self.max_img_seq_length:
                img_feat = img_feat[0:self.max_img_seq_length, ]
                if self.max_img_seq_length > 0:
                    input_mask = input_mask + [1] * img_feat.shape[0]
                    # segment_ids += [sequence_b_segment_id] * img_feat.shape[0]
            else:
                if self.max_img_seq_length > 0:
                    input_mask = input_mask + [1] * img_feat.shape[0]
                    # segment_ids = segment_ids + [sequence_b_segment_id] * img_feat.shape[0]
                padding_matrix = torch.zeros((self.max_img_seq_length - img_feat.shape[0], img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                if self.max_img_seq_length > 0:
                    input_mask = input_mask + ([0] * padding_matrix.shape[0])
                    # segment_ids = segment_ids + [pad_token_segment_id] * padding_matrix.shape[0]
            data['img_feats'] = img_feat

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

        update = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(input_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(segment_ids, dtype=torch.long),
            #'label_id': torch.tensor([label_id[0]], dtype=torch.long),
            'labels': torch.tensor(new_scores, dtype=torch.float),
            'question_id': torch.tensor([example.q_id], dtype=torch.long)
        }
        for k in update:
            assert k not in data
        data.update(update)
        return data

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

class VQAModel(nn.Module):
    def __init__(self, model, image_encoder=None):
        super().__init__()
        self.module = model
        self.iter = 0
        self.image_encoder = image_encoder

    def construct_attn_mask(self, data):
        img_feats = data['img_feats']
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        batch_size = img_feats.shape[0]

        num_img_feats = img_feats.shape[1]
        device = input_ids.device
        full_attention_mask = torch.cat(
            (attention_mask,
             torch.ones((batch_size, num_img_feats), device=device)), dim=1)
        data['attention_mask'] = full_attention_mask

    def forward(self, data):
        verbose = (self.iter % 100) == 0
        self.iter += 1
        data = dict(data.items())
        if self.image_encoder:
            assert 'img_feats' not in data
            data['img_feats'] = self.image_encoder(data.pop('image'))
            self.construct_attn_mask(data)

        if self.training:
            for k in ['idx', 'key', 'question_id']:
                data.pop(k)
            outputs =  self.module(**data)
            loss, logits = outputs[:2]

            if verbose:
                batch_score = compute_score_with_logits(logits, data['labels']).sum() / len(logits)
                logging.info('acc = {}'.format(batch_score))
            return {'vqa_loss': loss.mean()}
        else:
            data.pop('question_id')
            data.pop('key')
            data.pop('idx')
            # the model will have different paths if labels is not None
            data['labels'] = None
            outputs = self.module(**data)
            return outputs[0]

class VQAUniPipeline(UniPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            # only used to generate evaluate file name
            'evaluate_method': 'vqa_acc',
            'loss_type': 'bce',
            'drop_out': 0.3,
            'od_label_conf': 0.,
            'cls_hidden_scale': 3,
            # with 8 number workers, it is quite slow
            'num_workers': 16,
            # set od_label_conf = 1.1 to ignore od labels
            'classifier': 'linear',
            'gradient_clip': 1.,

            'max_seq_length': 128,
            'max_img_seq_length': 50,
            'optimizer_type': 'MAdamW',
            'bias_no_weight_decay': True,
            'ln_no_weight_decay': True,
            'scheduler_type': 'linear',
            'warmup_steps': 0,

            'ignore_project_image': False,
            'pad_to_max': True,
            'fusion_timm_param_drop_out_all': None,
        })
        self._tokenizer = None
        self._answermap = None

        if not self.cfg.pad_to_max:
            self.train_collate_fn = collate_fn
            self.test_collate_fn = collate_fn

    def get_len_dataset(self, is_train):
        data = self.cfg.data if is_train else self.cfg.test_data
        split = 'train' if is_train else self.cfg.test_split
        caption_version = self.cfg.train_version if is_train else None
        dataset = CaptionIdxTSVDataset(
            data=data,
            split=split,
            caption_version=caption_version,
        )
        return dataset

    def get_transform(self, is_train):
        data = self.cfg.data if is_train else self.cfg.test_data
        split = 'train' if is_train else self.cfg.test_split
        caption_version = self.cfg.train_version if is_train else None

        all_trans = []
        cache_policy = None
        hw_loader = LoadHW(
            data=data,
            split=split,
            cache_policy=cache_policy,
        )
        all_trans.append(hw_loader)
        max_img_seq_len = self.cfg.max_img_seq_length
        load_feature = max_img_seq_len > 0
        if load_feature:
            feature_loader = LoadFeature(
                data=data,
                split=split,
                version=self.cfg.train_feature_version,
                img_feature_dim=self.cfg.img_feature_dim,
                max_len=max_img_seq_len,
                sort_by_conf=not self.cfg.no_sort_by_conf,
            )
        else:
            # load image and we will extract the features online. This is mainly
            # used for end-to-end training or inference.
            image_loader = LoadImage(data, split)
            from qd.pipelines.uni_pipeline import get_transform_image
            image_transform = get_transform_image(self, is_train)
            from qd.data_layer.transform import ImageTransform2Dict
            image_transform = ImageTransform2Dict(image_transform)
            feature_loader = transforms.Compose([
                image_loader,
                image_transform,
            ])
        all_trans.append(feature_loader)

        caption_loader = LoadCaption(
            data=data, split=split, version=caption_version,
            cache_policy=cache_policy,
        )
        all_trans.append(caption_loader)

        label_loader = LoadLabel(
            data=data, split=split,
            version=caption_version)
        all_trans.append(label_loader)

        process = AggregateInputVQA(
            answers=self.answermap,
            tokenizer=self.tokenizer,
            max_seq_length=self.cfg.max_seq_length,
            max_img_seq_length=self.cfg.max_img_seq_length,
            od_label_conf=self.cfg.od_label_conf,
            img_feature_dim=self.cfg.img_feature_dim,
            pad_to_max=self.cfg.pad_to_max,
        )
        all_trans.append(process)

        useless_keys = [
            #'idx',
            'idx_img',
            'idx_cap',
            'dataset',
            'label',
            'caption',
            'text_ab_type',
            'text_a',
            'text_b',
            'width',
            'height',
            'text_changed',
            'text_a_or_b_changed',
            'img_feat',
            'max_seq_a_len',
            'seq_a_padded_len',
            'feats_conf',
            'feats_class',
            'teacher_feats_conf',
            'teacher_feats_class',
            'vocab_size',
            'feats_class_token_ids',
            'feats_class_tokens',
            'origin_input_ids',
        ]
        all_trans.extend([
            RemoveUselessKeys(useless_keys),
            RenameKey({'segment_ids': 'token_type_ids'}),
        ])
        return transforms.Compose(all_trans)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = BertTokenizer.from_pretrained(
                self.cfg.text_encoder_type,
                do_lower_case=True)
        return self._tokenizer

    @property
    def answermap(self):
        if self._answermap is None:
            dataset = TSVDataset(self.cfg.data)
            self._answermap = load_list_file(dataset.get_txt('answermap'))
        return self._answermap

    def get_fusion_config(self, is_train):
        config = BertConfig.from_pretrained(
            self.cfg.text_encoder_type,
            num_labels=len(self.answermap),
            finetuning_task='vqa_text',
        )
        if self.cfg.fusion_timm_param_drop_out_all:
            if not hasattr(config, 'timm_param'):
                config.timm_param = {}
            config.timm_param['drop_rate'] = self.cfg.fusion_timm_param_drop_out_all
            config.timm_param['attn_drop_rate'] = self.cfg.fusion_timm_param_drop_out_all
            config.timm_param['drop_path_rate'] = self.cfg.fusion_timm_param_drop_out_all
        # discrete code
        config.img_feature_dim = self.cfg.img_feature_dim
        config.img_feature_type = 'faster_r-cnn'
        config.code_voc = 512
        config.hidden_dropout_prob = self.cfg.drop_out
        config.loss_type = self.cfg.loss_type
        config.classifier = self.cfg.classifier
        config.cls_hidden_scale = self.cfg.cls_hidden_scale
        if self.cfg.prior_prob is not None:
            config.prior_prob = self.cfg.prior_prob
        config.use_img_layernorm = self.cfg.use_img_layernorm
        config.img_layer_norm_eps = 1e-5
        config.ignore_project_image = self.cfg.ignore_project_image
        return config

    def get_image_encoder(self, is_train):
        from qd.pipelines.uni_pipeline import get_image_encoder_model
        return get_image_encoder_model(self, is_train)

    def get_raw_model(self, is_train):
        config = self.get_fusion_config(is_train)

        image_encoder = None
        if self.cfg.max_img_seq_length == 0:
            image_encoder = self.get_image_encoder(is_train)

        if op.isfile(op.join(self.cfg.text_encoder_type, 'pytorch_model.bin')):
            model = ImageBertForSequenceClassification.from_pretrained(
                self.cfg.text_encoder_type,
                from_tf=False, config=config)
        else:
            model = ImageBertForSequenceClassification(config)
        model = VQAModel(
            model, image_encoder=image_encoder)
        return model

    def predict_output_to_tsv_row(self, data, output):
        val, max_idx = output.max(1)

        for i, idx in enumerate(data['idx']):
            result = {}
            result['question_id'] = data['question_id'][i].item()
            result['answer'] = self.answermap[max_idx[i].item()]
            yield int(idx), json_dump(result)

    def evaluate(self, predict_file, evaluate_file):
        if self.cfg.test_split in ['test_dev', 'test_std']:
            # we only convert the pred to json and then we should manually
            # upload the json file
            out_file = predict_file + '.server.json'
            from qd.tsv_io import tsv_reader
            result = [json.loads(s) for _, s in tsv_reader(predict_file)]
            from qd.qd_common import write_to_file, json_dump
            write_to_file(json_dump(result), out_file)
        else:
            return self.evaluate_acc(predict_file, evaluate_file)

    def evaluate_acc(self, predict_file, evaluate_file):
        from qd.tsv_io import TSVDataset
        dataset = TSVDataset(self.cfg.test_data)
        all_qa = [json.loads(s_cap) for key, s_cap in dataset.iter_data(
            self.cfg.test_split,
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

