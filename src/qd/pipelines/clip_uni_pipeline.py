from qd.qd_common import execute_func
from qd.mask.layers.bert import  BertConfig
import math
from qd.qd_common import write_to_yaml_file
from qd.torch_common import recursive_to_device
from qd.qd_common import qd_tqdm as tqdm
from qd.data_layer.builder import collate_fn
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
    LogSystemInfo,
    AppendDummyFeature,
)
from qd.data_layer.dataset import CaptionIdxTSVDataset, ImageIdxTSVDataset
from qd.mask.modeling.captioning.utils_data import make_batch_data_sampler
from qd.mask.data.build import make_data_sampler
from qd.layers.image_text_align import ImageTextAligner
from qd.process_tsv import iter_caption_to_json
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
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import json

#from qd.mask.utils.metric_logger import MetricLogger
from qd.mask.layers.bert import BertTokenizer
from qd.mask.modeling.captioning.utils_caption_evaluate import (
        evaluate_on_coco_caption, ScstRewardCriterion)

from qd.pipelines.uni_pipeline import UniPipeline
from torch import nn
from qd.layers.CLIP.clip import _tokenizer


class CLIPTokenizer(object):
    def __init__(self, context_length=77):
        self.context_length=context_length

    def tokenize(self, text):
        return _tokenizer.encode(text)

    @property
    def sep_token(self):
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        return eot_token

    @property
    def cls_token(self):
        sot_token = _tokenizer.encoder["<|startoftext|>"]
        return sot_token

    @property
    def mask_token(self):
        sot_token = _tokenizer.encoder["<|startoftext|>"]
        return sot_token

    @property
    def vocab_size(self):
        return 49408

    @property
    def pad_token(self):
        return 0

    def convert_tokens_to_ids(self, token):
        import copy
        return copy.deepcopy(token)

class CLIPTransformer(nn.Module):
    def __init__(self, width, layers, heads, embed_dim, context_length,
                 vocab_size):
        super().__init__()

        from qd.layers.CLIP.model import Transformer
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask(context_length)
        )
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.randn(width, embed_dim) * 1.  / math.sqrt(width))

    def build_attention_mask(self, context_length):
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, input_ids,
                position_ids=None, token_type_ids=None, attention_mask=None):
        #ipdb> pp text
        #tensor([[49406,   320, 22697, 49407,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0],
                #[49406,   320,  1929, 49407,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0],
                #[49406,   320,  2368, 49407,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                     #0,     0,     0,     0,     0,     0,     0]], device='cuda:0')
        x = self.token_embedding(input_ids)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding[:x.shape[1]]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)] @ self.text_projection

        return {'pooled_output': x}

def create_bert_model_encoder(config,
                              pre_trained,
                              output_dim,
                              pooler_bias=True,
                              ):
    from qd.mask.layers.bert.modeling_bert import BertPlainModel
    model = BertPlainModel(config, output_dim)
    if op.isfile(pre_trained):
        from qd.torch_common import torch_load
        state = torch_load(pre_trained)
        from qd.opt.checkpoint import load_state_dict
        load_state_dict(model, state)
    return model

class TimmImageEncoder(nn.Module):
    def __init__(self, model, output_dim, pooler_type=None):
        super().__init__()

        self.model = model
        in_features = self.model.norm.weight.shape[0]

        if pooler_type is None:
            self.proj = nn.Linear(in_features, output_dim, bias=False)
        elif pooler_type == 'i':
            self.proj = nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, image):
        feats = self.model(image)
        return self.proj(feats)

def create_timm_image_encoder(output_dim, pooler_type=None, **kwargs):
    import timm
    model = timm.create_model(**kwargs)
    return TimmImageEncoder(model, output_dim, pooler_type)

class CLIPPipeline(UniPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'max_seq_length': 70,
            'tie_weights': False,
            'log_step': 100,

            'mask_prob': 0.15,
            'replace_by_mask_prob': 0.8,
            'replace_by_rand_prob': 0.1,

            # e2e
            'train_crop_size': 224,
            'test_crop_size': 224,
            'image_encoder_pretrained': False,
            'pad_to_max': False,
            'align_loss': None,
            'align_loss_weight': 1.,
            'mask_type': 'seq2seq',
            'qd_format': True,
            'max_gen_length': 20,

            'max_masked_tokens': 3,

            'evaluate_method': 'top1',
            'temperature': 0.2,
            'loss_style': 'batch',
            'queue_size': 1024,
            'text_encoder_pooler_bias': True,

            'cluster_sinkhorn_eps': 0.05,
            'batch_sink_weight': 1.,
            'batch_weight_sink': 1.,
        })

        # data layer
        self._tokenizer = None
        self._test_caption_tensorizer = None
        self._train_caption_tensorizer = None
        self._test_captionmap = None

        self.train_collate_fn = collate_fn
        self.test_collate_fn = collate_fn

    def get_len_dataset(self, is_train):
        if is_train:
            #'idx': idx,
            #'idx_img': idx_img,
            #'idx_cap': idx_cap,
            #'key': key,
            #'dataset': self,
            dataset = CaptionIdxTSVDataset(
                data=self.cfg.data,
                split='train',
                caption_version=self.cfg.train_version,
            )
        else:
            dataset = ImageIdxTSVDataset(
                data=self.cfg.test_data,
                split=self.cfg.test_split,
            )
        return dataset

    def get_transform(self, is_train):
        data = self.cfg.data if is_train else self.cfg.test_data
        split = 'train' if is_train else self.cfg.test_split
        all_trans = []
        cache_policy = None
        load_feature = False
        if load_feature:
            hw_loader = LoadHW(
                data=data, split=split,
                cache_policy=cache_policy,
            )
            all_trans.append(hw_loader)

        if load_feature:
            feature_loader = LoadFeature(
                data=data,
                split=split,
                version=self.cfg.feature_version,
                img_feature_dim=self.cfg.img_feature_dim,
                max_len=self.cfg.max_img_seq_len,
                sort_by_conf=self.cfg.feat_sort_by_conf,
            )
        else:
            # load image and we will extract the features online. This is mainly
            # used for end-to-end training or inference.
            normalize = None
            backend = 'cv'
            if self.cfg.text_encoder_type == 'CLIP':
                normalize = transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                )
                backend = 'pil'
            if self.cfg.sampler_type == 'splitBysplit':
                hold_buffer = self.cfg.splitbysplitsample_buffer_size
            else:
                assert self.cfg.sampler_type in [
                    None, 'ranksplit', 'nodesplit']
                hold_buffer = 0
            image_loader = LoadImage(
                data,
                split,
                add_key=True,
                backend=backend,
                hold_buffer=hold_buffer,
            )
            if is_train:
                from qd.data_layer.transform import get_inception_train_transform
                image_transform = get_inception_train_transform(
                    bgr2rgb=True,
                    crop_size=self.cfg.train_crop_size,
                    small_scale=self.cfg.input_small_scale,
                    normalize=normalize,
                    backend=backend,
                )
            else:
                if self.cfg.test_resize_size is None:
                    resize_size = 256 * self.cfg.test_crop_size // 224
                else:
                    resize_size = self.cfg.test_resize_size
                from qd.data_layer.transform import get_inception_test_transform
                from PIL import Image
                image_transform = get_inception_test_transform(
                    bgr2rgb=True,
                    resize_size=resize_size,
                    crop_size=self.cfg.test_crop_size,
                    normalize=normalize,
                    interpolation=Image.BICUBIC,
                    backend=backend,
                )
            from qd.data_layer.transform import ImageTransform2Dict
            image_transform = ImageTransform2Dict(image_transform)
            feature_loader = transforms.Compose([
                image_loader,
                image_transform,
            ])

        all_trans.append(feature_loader)

        if is_train:
            caption_loader = LoadCaption(
                data=data, split=split, version=self.cfg.train_version,
                cache_policy=cache_policy,
            )
            all_trans.append(caption_loader)

        #if add_od_labels:
            #label_loader = LoadLabel(
                #data=data, split=split,
                #version=label_version)
            #all_trans.append(label_loader)

        text_ab = IdentifyTextAB(
            add_od_labels=False,
            od_label_conf=2.,
            label_sort_by_conf=False,
            unique_labels_on=False,
            qa2caption=False,
            sep_token=self.tokenizer.sep_token,
        )
        all_trans.append(text_ab)

        tensorizer = (self.train_caption_tensorizer if is_train else
                      self.test_caption_tensorizer)
        from qd.data_layer.transform import CaptionTensorizer
        trans_tensorizer = CaptionTensorizer(
            tensorizer,
            with_img_feats=load_feature,
            pad_to_max=False,
        )
        all_trans.append(trans_tensorizer)

        useless_keys = [
                'idx',
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
                'feats_conf',
                'feats_class',
                'vocab_size',
                'feats_class_token_ids',
                'feats_class_tokens',
        ]
        all_trans.extend([
            RemoveUselessKeys(useless_keys),
            RenameKey({'segment_ids': 'token_type_ids'}),
            #LogSystemInfo(),
        ])
        all_trans = transforms.Compose(all_trans)
        return all_trans

    def get_raw_model(self, is_train):
        text_encoder = self.get_text_encoder_fd()
        from qd.qd_common import dict_has_path
        if self.cfg.embed_dim is None:
            if dict_has_path(text_encoder, 'param$config'):
                embed_dim = text_encoder['param']['config'].hidden_size
            else:
                embed_dim = 1024
        else:
            embed_dim = self.cfg.embed_dim

        image_encoder = self.get_image_encoder(is_train, embed_dim)

        if self.cfg.share_backbone:
            logging.info('share the backbone')
            assert str(image_encoder.model.blocks) == str(image_encoder.model.blocks)
            text_encoder.encoder.blocks = image_encoder.model.blocks

        model = ImageTextAligner(
            image_encoder,
            text_encoder,
            self.cfg.align_loss,
            temperature=self.cfg.temperature,
            loss_style=self.cfg.loss_style,
            queue_size=self.cfg.queue_size,
            feat_dim=embed_dim,
            sinkhorn_eps=self.cfg.cluster_sinkhorn_eps,
            batch_sink_weight=self.cfg.batch_sink_weight,
            batch_weight_sink=self.cfg.batch_weight_sink,
        )

        if is_train:
            if self.cfg.init_last_bias:
                if hasattr(model.image_encoder, 'fc') and \
                        hasattr(model.text_encoder, 'pooler'):
                    logging.info('setting the bias as 0.1')
                    torch.nn.init.constant_(
                        model.image_encoder.fc.bias,
                        self.cfg.init_last_bias,
                    )
                    torch.nn.init.constant_(
                        model.text_encoder.pooler.dense.bias,
                        self.cfg.init_last_bias,
                    )
                else:
                    raise NotImplementedError
        if not is_train:
            model = model.eval()
        return model

    def append_predict_param(self, cc):
        super().append_predict_param(cc)
        if self.cfg.test_version:
            # normally, prediction does not depend on the test version, and
            # only evaluation depends on it. But in CLIP, the prediction
            # depends on the version as we need the labelmap
            cc.append('v{}'.format(self.cfg.test_version))

    def predict_output_to_tsv_row(self, data, output, **kwargs):
        topk = 10
        if output.shape[1] < topk:
            topk = output.shape[1]
        labelmap = self.test_captionmap
        all_tops, all_top_indexes = output.topk(topk, dim=1,
                largest=True, sorted=False)
        keys = data['key']
        for key, tops, top_indexes in zip(keys, all_tops, all_top_indexes):
            all_tag = [{'caption': labelmap[i], 'conf': float(t)} for t, i in
                       zip(tops, top_indexes)]
            yield key, json.dumps(all_tag)

    def get_text_encoder_fd(self):
        text_encoder_type = self.cfg.text_encoder_type
        if op.isdir(text_encoder_type):
            pre_trained = op.isfile(op.join(text_encoder_type, 'pytorch_model.bin'))
            config = BertConfig.from_pretrained(text_encoder_type)
            config.vocab_size = self.tokenizer.vocab_size
            config.pooler_type = self.cfg.text_pooler_type
            config.return_dict = True
            # pooler_activate is not used.
            config.pooler_activate = False
            return execute_func({
                'from': 'qd.pipelines.clip_uni_pipeline',
                'import': 'create_bert_model_encoder',
                'param': {
                    'config': config,
                    'pre_trained': pre_trained,
                    'output_dim': self.cfg.embed_dim,
                    'pooler_bias': self.cfg.text_encoder_pooler_bias,
                },
            })
        elif text_encoder_type == 'CLIP':
            return execute_func({
                'from': 'qd.pipelines.clip_uni_pipeline',
                'import': 'CLIPTransformer',
                'param': {
                    'width': 512,
                    'layers': 12,
                    'heads': 8,
                    # 1024 for resnet50; 512 for vit-s
                    'embed_dim': self.cfg.embed_dim or 1024,
                    'vocab_size': self.tokenizer.vocab_size,
                    'context_length': 77,
                },
            })
        else:
            raise NotImplementedError

    def get_image_encoder(self, is_train, hidden_size):
        from qd.pipelines.uni_pipeline import get_image_encoder
        return get_image_encoder(self, is_train, hidden_size)

    @property
    def test_captionmap(self):
        if self._test_captionmap is None:
            from qd.tsv_io import TSVDataset
            dataset = TSVDataset(self.cfg.test_data)
            assert dataset.has(self.cfg.test_split,
                               'captionmap',
                               version=self.cfg.test_version)
            self._test_captionmap = [c for c, in dataset.iter_data(
                self.cfg.test_split, 'captionmap',
                version=self.cfg.test_version)]
        return self._test_captionmap

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self.cfg.text_encoder_type == 'CLIP':
                tokenizer = CLIPTokenizer()
            else:
                tokenizer = BertTokenizer.from_pretrained(
                    self.cfg.text_encoder_type, do_lower_case=True)
            self._tokenizer = tokenizer
        return self._tokenizer

    @property
    def train_caption_tensorizer(self):
        if self._train_caption_tensorizer is None:
            from qd.mask.data.datasets.caption_tensorizer import CaptionTensorizer
            caption_tensorizer = CaptionTensorizer(
                self.tokenizer,
                max_img_seq_length=0,
                max_seq_length=self.cfg.max_gen_length,
                max_seq_a_length=self.cfg.max_gen_length,
                mask_prob=self.cfg.mask_prob,
                max_masked_tokens=self.cfg.max_masked_tokens,
                mask_type=self.cfg.mask_type,
                is_train=True,
                mask_b=False,
                replace_by_mask_prob=self.cfg.replace_by_mask_prob,
                replace_by_rand_prob=self.cfg.replace_by_rand_prob,
            )
            self._train_caption_tensorizer = caption_tensorizer
        return self._train_caption_tensorizer

    @property
    def test_caption_tensorizer(self):
        if self._test_caption_tensorizer is None:
            from qd.mask.data.datasets.caption_tensorizer import CaptionTensorizer
            caption_tensorizer = CaptionTensorizer(
                self.tokenizer,
                max_img_seq_length=0,
                max_seq_length=self.cfg.max_gen_length,
                max_seq_a_length=self.cfg.max_gen_length,
                mask_type=self.cfg.mask_type,
                is_train=False,
            )
            self._test_caption_tensorizer = caption_tensorizer
        return self._test_caption_tensorizer

    def post_load_model_surgery(self, model, model_file):
        all_data = [self.test_caption_tensorizer.tensorize_ab(
            caption,
            pad_to_max=False,
        ) for caption in self.test_captionmap]
        model = model.to(self.cfg.device)
        with torch.no_grad():
            all_abstracted_text = []
            for i, d in tqdm(enumerate(all_data)):
                d = recursive_to_device(d, self.cfg.device)
                x = model.feed_test_texts(d)
                all_abstracted_text.append(x)
            all_abstracted_text = torch.cat(all_abstracted_text)
        model.all_abstracted_text = all_abstracted_text
        model = super().post_load_model_surgery(model, model_file)
        return model

    def evaluate(self, predict_file, evaluate_file):
        from qd.tsv_io import TSVDataset, tsv_reader
        dataset = TSVDataset(self.cfg.test_data)
        iter_label = dataset.iter_data(self.cfg.test_split, 'caption',
                self.cfg.test_version)
        from qd.torch_common import evaluate_topk
        top1 = evaluate_topk(tsv_reader(predict_file), iter_label, 'caption')
        logging.info('top1 = {}'.format(top1))
        write_to_yaml_file({'top1': top1}, evaluate_file)

