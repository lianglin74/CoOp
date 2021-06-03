import random
from vilt.modules import heads, objectives, vilt_utils
from vilt.modules import ViLTransformerSS
from qd.qd_common import json_dump
import functools
from qd.torch_common import torch_load, torch_save
from qd.mask.layers.bert import BertForImageCaptioning
from qd.qd_common import execute_func
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
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)
from qd.data_layer.dataset import CaptionIdxTSVDataset, ImageIdxTSVDataset
from qd.pipelines.uni_pipeline import UniPipeline
from torch import nn
from qd.data_layer.dataset import CaptionIdxTSVDataset, ImageIdxTSVDataset
from .uni_pipeline import UniPipeline


def collate_pert(batch, mlm_collator):
    # this function is ok to replace collate(), but we keep two functions in
    # case there is bug in this new function
    for b in batch:
        if isinstance(b['image'], torch.Tensor):
            b['image'] = [b['image']]
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    task = dict_batch['current_tasks'][0]

    if 'itm' in task and 'false_image_0' not in dict_batch:
        import copy
        images = dict_batch['image']
        idx = list(range(len(images)))
        random.shuffle(idx)
        dict_batch['false_image_0'] = [copy.deepcopy(images[i]) for i in idx]
        dict_batch['correct_false'] = torch.tensor([
            int(dict_batch['idx_img'][i] != dict_batch['idx_img'][ii]) for i, ii in enumerate(idx)])

    img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
    img_sizes = list()

    for img_key in img_keys:
        img = dict_batch[img_key]
        img_sizes += [ii.shape for i in img if i is not None for ii in i]

    for size in img_sizes:
        assert (
            len(size) == 3
        ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

    if len(img_keys) != 0:
        max_height = max([i[1] for i in img_sizes])
        max_width = max([i[2] for i in img_sizes])

    for img_key in img_keys:
        img = dict_batch[img_key]
        view_size = len(img[0])

        new_images = [
            torch.zeros(batch_size, 3, max_height, max_width)
            for _ in range(view_size)
        ]

        for bi in range(batch_size):
            orig_batch = img[bi]
            for vi in range(view_size):
                if orig_batch is None:
                    new_images[vi][bi] = None
                else:
                    orig = img[bi][vi]
                    new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

        dict_batch[img_key] = new_images

    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

    if len(txt_keys) != 0:
        texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        #draw_text_len = len(encodings)
        flatten_encodings = [e for encoding in encodings for e in encoding]
        flatten_mlms = mlm_collator(flatten_encodings)

        for i, txt_key in enumerate(txt_keys):
            texts, encodings = (
                [d[0] for d in dict_batch[txt_key]],
                [d[1] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = torch.zeros_like(mlm_ids)
            attention_mask = torch.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                _input_ids, _attention_mask = (
                    torch.tensor(encoding["input_ids"]),
                    torch.tensor(encoding["attention_mask"]),
                )
                input_ids[_i, : len(_input_ids)] = _input_ids
                attention_mask[_i, : len(_attention_mask)] = _attention_mask

            dict_batch[txt_key] = texts
            dict_batch[f"{txt_key}_ids"] = input_ids
            dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
            dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            dict_batch[f"{txt_key}_masks"] = attention_mask
    dict_batch['current_tasks'] = task
    return dict_batch

def collate(batch, mlm_collator):
    for b in batch:
        if isinstance(b['image'], torch.Tensor):
            b['image'] = [b['image']]
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
    img_sizes = list()

    for img_key in img_keys:
        img = dict_batch[img_key]
        img_sizes += [ii.shape for i in img if i is not None for ii in i]

    for size in img_sizes:
        assert (
            len(size) == 3
        ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

    if len(img_keys) != 0:
        max_height = max([i[1] for i in img_sizes])
        max_width = max([i[2] for i in img_sizes])

    for img_key in img_keys:
        img = dict_batch[img_key]
        view_size = len(img[0])

        new_images = [
            torch.zeros(batch_size, 3, max_height, max_width)
            for _ in range(view_size)
        ]

        for bi in range(batch_size):
            orig_batch = img[bi]
            for vi in range(view_size):
                if orig_batch is None:
                    new_images[vi][bi] = None
                else:
                    orig = img[bi][vi]
                    new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

        dict_batch[img_key] = new_images

    txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

    if len(txt_keys) != 0:
        texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
        #draw_text_len = len(encodings)
        flatten_encodings = [e for encoding in encodings for e in encoding]
        flatten_mlms = mlm_collator(flatten_encodings)

        for i, txt_key in enumerate(txt_keys):
            texts, encodings = (
                [d[0] for d in dict_batch[txt_key]],
                [d[1] for d in dict_batch[txt_key]],
            )

            mlm_ids, mlm_labels = (
                flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
            )

            input_ids = torch.zeros_like(mlm_ids)
            attention_mask = torch.zeros_like(mlm_ids)
            for _i, encoding in enumerate(encodings):
                _input_ids, _attention_mask = (
                    torch.tensor(encoding["input_ids"]),
                    torch.tensor(encoding["attention_mask"]),
                )
                input_ids[_i, : len(_input_ids)] = _input_ids
                attention_mask[_i, : len(_attention_mask)] = _attention_mask

            dict_batch[txt_key] = texts
            dict_batch[f"{txt_key}_ids"] = input_ids
            dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
            dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
            dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
            dict_batch[f"{txt_key}_masks"] = attention_mask

    dict_batch['current_tasks'] = dict_batch['current_tasks'][0]
    return dict_batch

class PrepareText(object):
    def __init__(self, tokenizer, max_text_len, pad_to_max=True):
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.token_padding = 'max_length' if pad_to_max else 'do_not_pad'

    def get_extra_field(self, data):
        return {}

    def get_text(self, data):
        if 'caption' in data['caption']:
            text = data['caption']['caption']
        elif 'shortAnswer' in data['caption'] and 'question' in data['caption']:
            # vqa dataset
            text = data['caption']['question'] + data['caption']['shortAnswer']
        elif 'question' in data['caption'] and 'answer' in data['caption']:
            # gqa dataset
            text = data['caption']['question'] + data['caption']['answer']
        elif 'question' in data['caption'] and 'answers' in data['caption']:
            # vq-qa dataset
            if len(data['caption']['answers']) > 0:
                text = data['caption']['question'] + random.choice(data['caption']['answers'])
            else:
                text = data['caption']['question']
        else:
            raise NotImplementedError(data['caption'])
        return text

    def __call__(self, data):
        # vilt/datasets/base_dataset.py:get_text

        text = self.get_text(data)
        #'Is the airplane about to take off?'
        encoding = self.tokenizer(
            text,
            padding=self.token_padding,
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        update = {
            "text": (text, encoding),
            #"img_index": index,
            #"cap_index": caption_index,
            #"raw_index": raw_index,
        }
        update.update(self.get_extra_field(data))

        for k in update:
            assert k not in data
        data.update(update)
        return data

class PrepareQuestion(PrepareText):
    def __init__(self, tokenizer, max_text_len, pad_to_max):
        super().__init__(tokenizer, max_text_len, pad_to_max)

    def get_text(self, data):
        text = data['caption']['question']
        return text

    def get_extra_field(self, data):
        return {'question_id': data['caption'].get('question_id', 0)}

class PrepareAnswer(object):
    def __init__(self, answers):
        self.answer2idx = {a: i for i, a in enumerate(answers)}

    def __call__(self, data):
        update = {}
        cap = data['caption']
        answers = cap.get('answers', [])
        update['vqa_answer'] = answers
        update['vqa_labels'] = [self.answer2idx.get(l, -1)  for l in answers]
        update['vqa_scores'] = cap.get('confs', [])

        for k in update:
            assert k not in data
        data.update(update)
        return data

class ViLTUniModel(ViLTransformerSS):
    def forward(self, data):
        self.current_tasks = data['current_tasks']
        #vilt_utils.set_task(self)
        dtype = next(self.named_parameters())[1].dtype
        img_keys = [k for k in data if 'image' in k]
        for k in img_keys:
            data[k] = [i.to(dtype) for i in data[k]]
        ret = super().forward(data)
        if self.training:
            ret = dict([(k, v) for k, v in ret.items() if 'loss' in k])
        return ret

class ViltUniPipeline(UniPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'transform_key': 'pixelbert',
            'tokenizer': 'bert-base-uncased',
            'max_text_len': 40,
            'test_transform_key': 'pixelbert',
            'train_transform_key': 'pixelbert_randaug',
            'mlm_prob': 0.15,
            'whole_word_masking': False,
            'end_lr': 0.,
            'lr_mult': 10,
            'warmup_steps': 0.1,
            'optim_type': 'adamw',
            'weight_decay': 0.01,
            'decay_power': 1,
            'max_image_len': -1,
            'draw_false_image': 0,
            'loss_names': {
                'irtr': 0,
                'itm': 0,
                'mlm': 0,
                'mpp': 0,
                'nlvr2': 0,
                'vqa': 0,
            },
            'attach_iter_in_sampler': True,
            'vit': 'vit_base_patch32_384',
        })

        self._tokenizer = None
        self._mlm_collator = None

    def get_optimizer(self, model):
        lr = self.cfg.base_lr
        wd = self.cfg.weight_decay

        no_decay = [
            "bias",
            "LayerNorm.bias",
            "LayerNorm.weight",
            "norm.bias",
            "norm.weight",
            "norm1.bias",
            "norm1.weight",
            "norm2.bias",
            "norm2.weight",
        ]
        head_names = ["vqa_classifier", "nlvr2_classifier"]
        lr_mult = self.cfg.lr_mult
        optim_type = self.cfg.optim_type

        #names = [n for n, p in model.named_parameters()]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                'param_names': [
                    n
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                "weight_decay": wd,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                'param_names': [
                    n
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                ],
                'param_names': [
                    n
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                ],
                'param_names': [
                    n
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult,
            },
        ]
        optimizer_grouped_parameters = [
            x for x in optimizer_grouped_parameters if len(x['params']) > 0
        ]

        if optim_type == "adamw":
            from transformers.optimization import AdamW
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
            )
        elif optim_type == "adam":
            optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optim_type == "sgd":
            optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
        return optimizer

    def get_lr_scheduler(self, optimizer):
        max_steps = self.max_iter

        warmup_steps = self.cfg.warmup_steps
        if isinstance(self.cfg.warmup_steps, float):
            warmup_steps = int(max_steps * warmup_steps)

        from transformers import (
            get_polynomial_decay_schedule_with_warmup,
            get_cosine_schedule_with_warmup,
        )
        end_lr = self.cfg.end_lr
        decay_power = self.cfg.decay_power
        if decay_power == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
            )
        else:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
                lr_end=end_lr,
                power=decay_power,
            )

        return scheduler

    def get_collate_fn(self, is_train):
        return functools.partial(
            collate,
            mlm_collator=self.mlm_collator,
        )

    #@property
    #def train_collate_fn(self):
        #return self.collate

    #@property
    #def test_collate_fn(self):
        #return self.collate

    @property
    def mlm_collator(self):
        if self._mlm_collator is None:
            collator = (
                DataCollatorForWholeWordMask
                if self.cfg.whole_word_masking
                else DataCollatorForLanguageModeling
            )
            self._mlm_collator = collator(
                tokenizer=self.tokenizer,
                mlm=True,
                mlm_probability=self.cfg.mlm_prob
            )
        return self._mlm_collator

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from vilt.datamodules.datamodule_base import get_pretrained_tokenizer
            self._tokenizer = get_pretrained_tokenizer(self.cfg.tokenizer)
        return self._tokenizer

    def get_vilt_config(self, is_train):
        # config based on vqa/train
        config = {
            'batch_size': 256,
            'data_root': '',
            'datasets': ['vqa'],
            'draw_false_text': 0,
            'draw_false_image': self.cfg.draw_false_image,
            'drop_rate': 0.1,
            'end_lr': 0,
            'exp_name': 'finetune_vqa_randaug',
            'fast_dev_run': False,
            'get_recall_metric': False,
            'image_only': False,
            'image_size': 384,
            'learning_rate': 0.0001,
            'log_dir': 'result',
            'max_image_len': self.cfg.max_image_len,
            'max_steps': None,
            'mlm_prob': 0.15,
            'mlp_ratio': 4,
            'num_gpus': 1,
            'num_heads': 12,
            'num_layers': -1, # this parameter is not used actually
            'num_nodes': 1,
            'num_workers': 8,
            'per_gpu_batchsize': 64,
            'precision': 16,
            'resume_from': None,
            'seed': 0,
            'test_only': False,
            'train_transform_keys': ['pixelbert_randaug'],
            'val_check_interval': 0.1,
            'val_transform_keys': ['pixelbert'],
            'vit': self.cfg.vit,
            'vocab_size': 30522,
            'vqav2_label_size': 3129,
            # the following is confirmed to be here.
            'load_path': '', # always load it from imagenet pretrained model
            'max_text_len': self.cfg.max_text_len,
            'tokenizer': self.cfg.tokenizer,
            'loss_names': self.cfg.loss_names,
        }
        if config['vit'] in ['vit_base_patch32_384']:
            config['patch_size'] = 32
            config['hidden_size'] = 768
        elif config['vit'] in ['vit_large_patch32_384']:
            config['patch_size'] = 32
            config['hidden_size'] = 1024
        else:
            raise NotImplementedError(config['vit'])
        return config

    def get_raw_model(self, is_train):
        _config = self.get_vilt_config(is_train)
        if not is_train:
            _config['test_only'] = True
        else:
            _config['test_only'] = False
        model = ViLTUniModel(_config)
        return model

class DrawFalseImage(object):
    def __init__(self, image_loader, image_transform, num):
        self.image_loader = image_loader
        self.image_transform = image_transform
        self.num = num

    def __call__(self, data):
        max_len = len(self.image_loader)
        for i in range(self.num):
            random_index = random.randint(0, max_len - 1)
            false_image = self.image_loader({'idx_img': random_index})
            false_image = self.image_transform(false_image)
            data[f"false_image_{i}"] = [false_image['image']]
        return data

class SelectTask(object):
    def __init__(self, is_train, loss_names):
        self.is_train = is_train
        self.loss_names = loss_names

    def __call__(self, data):
        if not self.is_train:
            current_tasks = [
                k for k, v in self.loss_names.items() if v >= 1
            ]
        else:
            sampling_pools = list()
            for k, v in self.loss_names.items():
                sampling_pools.extend([k] * v)
            old_state = random.getstate()
            random.seed(data['iteration'])
            current_tasks = [random.choice(sampling_pools)]
            random.setstate(old_state)
        data['current_tasks'] = current_tasks
        return data

class VLPViltUniPipeline(ViltUniPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'ignore_predict': True,
            'max_image_len': 200,
            'draw_false_image': 1,
            'loss_names': {
                'irtr': 0,
                'itm': 1,
                'mlm': 1,
                'mpp': 0,
                'nlvr2': 0,
                'vqa': 0,
            },
            'whole_word_masking': True,
            'collate_pert': False,
            'pad_to_max': True,
            'train_transform_key': 'pixelbert',
        })

    def get_collate_fn(self, is_train):
        if not self.cfg.collate_pert:
            return functools.partial(
                collate,
                mlm_collator=self.mlm_collator,
            )
        else:
            return functools.partial(
                collate_pert,
                mlm_collator=self.mlm_collator,
            )

    def get_len_dataset(self, is_train):
        data = self.cfg.data if is_train else self.cfg.test_data
        split = 'train' if is_train else self.cfg.test_split
        caption_version = self.cfg.train_version if is_train else None
        # no matter it is training or testing
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

        image_loader = LoadImage(data, split, backend='pil')
        #from qd.pipelines.uni_pipeline import get_transform_image
        #image_transform = get_transform_image(self, is_train)
        from vilt.transforms import keys_to_transforms
        image_size = self.cfg.train_crop_size if is_train else self.cfg.test_crop_size
        transform_key = self.cfg.train_transform_key if is_train else self.cfg.test_transform_key
        image_transform = keys_to_transforms(
            [transform_key],
            size=image_size,
        )[0]
        from qd.data_layer.transform import ImageTransform2Dict
        image_transform = ImageTransform2Dict(image_transform)
        image_load_trans = transforms.Compose([image_loader, image_transform])
        all_trans.append(image_load_trans)

        if self.cfg.draw_false_image:
            assert self.cfg.draw_false_image == 1
            if not self.cfg.collate_pert:
                draw_false_image = DrawFalseImage(image_loader, image_transform,
                                                  self.cfg.draw_false_image)
                all_trans.append(draw_false_image)

        caption_loader = LoadCaption(
            data=data, split=split, version=caption_version,
            cache_policy=cache_policy,
        )
        all_trans.append(caption_loader)

        p = PrepareText(
            self.tokenizer,
            self.cfg.max_text_len,
            self.cfg.pad_to_max,
        )
        all_trans.append(p)

        p = SelectTask(is_train, self.cfg.loss_names)
        all_trans.append(p)

        useless_keys = [
            #'idx',
            #'idx_img',
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


class VQAViltUniPipeline(ViltUniPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'loss_names': {
                'irtr': 0,
                'itm': 0,
                'mlm': 0,
                'mpp': 0,
                'nlvr2': 0,
                'vqa': 1,
            },
            'evaluate_method': 'vqa_acc',
        })
        self._answermap = None

    def get_len_dataset(self, is_train):
        data = self.cfg.data if is_train else self.cfg.test_data
        split = 'train' if is_train else self.cfg.test_split
        caption_version = self.cfg.train_version if is_train else None
        # no matter it is training or testing
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

        image_loader = LoadImage(data, split, backend='pil', add_key=True)
        #from qd.pipelines.uni_pipeline import get_transform_image
        #image_transform = get_transform_image(self, is_train)
        from vilt.transforms import keys_to_transforms
        image_size = self.cfg.train_crop_size if is_train else self.cfg.test_crop_size
        transform_key = self.cfg.train_transform_key if is_train else self.cfg.test_transform_key
        image_transform = keys_to_transforms(
            [transform_key],
            size=image_size,
        )[0]
        from qd.data_layer.transform import ImageTransform2Dict
        image_transform = ImageTransform2Dict(image_transform)
        all_trans.append(image_loader)
        all_trans.append(image_transform)

        caption_loader = LoadCaption(
            data=data, split=split, version=caption_version,
            cache_policy=cache_policy,
        )
        all_trans.append(caption_loader)

        p = PrepareQuestion(
            self.tokenizer,
            self.cfg.max_text_len,
            self.cfg.pad_to_max,
        )
        all_trans.append(p)

        p = PrepareAnswer(self.answermap)
        all_trans.append(p)

        p = SelectTask(is_train, self.cfg.loss_names)
        all_trans.append(p)

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
    def answermap(self):
        if self._answermap is None:
            from qd.tsv_io import TSVDataset
            dataset = TSVDataset(self.cfg.data)
            from qd.qd_common import load_list_file
            self._answermap = load_list_file(dataset.get_txt('answermap'))
        return self._answermap

    def predict_output_to_tsv_row(self, data, output):
        val, max_idx = output["vqa_logits"].max(1)
        for i, idx in enumerate(data['idx']):
            result = {}
            result['question_id'] = data['question_id'][i]
            result['answer'] = self.answermap[max_idx[i].item()]
            yield int(idx), json_dump(result)

    def evaluate(self, predict_file, evaluate_file):
        from qd.pipelines.uni_pipeline import evaluate_vqa
        return evaluate_vqa(self, predict_file, evaluate_file)

