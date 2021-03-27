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
from qd.data_layer.dataset import CaptionIdxTSVDataset, ImageIdxTSVDataset
from qd.pipelines.uni_pipeline import UniPipeline
from torch import nn


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data # argmax
    return logits == labels

def construct_basemodel_image_joint(image_path, joint_path, out):
    from qd.torch_common import torch_load, torch_save
    if op.isfile(out):
        logging.info('{} exists'.format(out))
        return
    from qd.qd_common import save_frame_yaml
    save_frame_yaml(out + '.yaml')

    image_model = torch_load(image_path)
    from qd.torch_common import remove_prefix
    image_model = remove_prefix(image_model['model'], 'module.')

    image_encoder = [('image_encoder.module.{}'.format(k), v) for k, v in image_model.items()]
    image_encoder = dict(image_encoder)

    if joint_path is not None:
        assert op.isdir(joint_path), 'not supported'
        joint = op.join(joint_path, 'pytorch_model.bin')
        joint = torch_load(joint)
        joint_bert = [('module.{}'.format(k), v) for k, v in joint.items()]
        joint_bert = dict(joint_bert)
        image_encoder.update(joint_bert)
    torch_save({'model': image_encoder}, out)

class ImageCaptioning(nn.Module):
    def __init__(self,
                 model,
                 test_extra_input=None,
                 image_encoder=None,
                 cfg=None,
                 ):
        super().__init__()
        self.module = model
        self.iter = 0
        self.test_extra_input = test_extra_input
        self.image_encoder = image_encoder
        self.cfg = cfg

    def construct_attn_mask(self, data):
        img_feats = data['img_feats']
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        batch_size = img_feats.shape[0]

        num_img_feats = img_feats.shape[1]
        num_token = input_ids.shape[-1]
        device = input_ids.device
        top_right = torch.ones((batch_size, num_token, num_img_feats), device=device)
        if self.cfg.mask_type in ['seq2seq', 'seq2seq_off']:
            bottom_left = torch.zeros((batch_size, num_img_feats, num_token), device=device)
        else:
            assert self.cfg.mask_type == 'bidirectional'
            bottom_left = torch.ones((batch_size, num_img_feats, num_token), device=device)
        bottom_right = torch.ones((batch_size, num_img_feats, num_img_feats), device=device)
        bottom = torch.cat((bottom_left, bottom_right), dim=2)

        top = torch.cat((attention_mask, top_right), dim=2)
        full_attention_mask = torch.cat((top, bottom), dim=1)
        data['attention_mask'] = full_attention_mask

    def forward(self, data):
        data = dict(data.items())
        # this is required in test, but not in train
        data.pop('key')
        if self.image_encoder:
            assert 'img_feats' not in data
            data['img_feats'] = self.image_encoder(data.pop('image'))
            self.construct_attn_mask(data)
            #from qd.tsv_io import TSVDataset
            #dataset = TSVDataset('TaxCocoCaption')
            #all_feats = []
            #for key in keys:
                #_, srects = dataset.seek_by_key(key, 'test', 'feature', 'ViTB16_224')
                #rects = json.loads(srects)
                #from qd.qd_common import decode_np
                #features = torch.tensor([decode_np(r['zlib_feature']) for r in rects])
                #all_feats.append(features)
            #features = torch.cat(all_feats).to(data['img_feats'].device)
            #diff = (data['img_feats'] - features).abs().sum()
            #x = data['img_feats'].abs().sum()
            #logging.info(diff)
            #logging.info(diff / x)
        if self.training:
            result = self.module(**data, return_dict=True)
            verbose = (self.iter % 100) == 0
            self.iter += 1
            if verbose:
                masked_ids = data['masked_ids']
                masked_ids = masked_ids[masked_ids != 0]
                batch_score = compute_score_with_logits(result['class_logits'], masked_ids)
                batch_acc = torch.sum(batch_score.float()) / torch.sum(data['masked_pos'])
                logging.info('acc = {}'.format(batch_acc))
            return {
                'masked_loss': result['masked_loss']
            }
        else:
            data.update(self.test_extra_input)
            result = self.module(**data)
            return result

class CaptionUniPipeline(UniPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'mask_type': 'seq2seq',
            'max_seq_length': 70,
            'add_od_labels': True,
            'od_label_conf': 0.2,
            'drop_out': 0.1,
            'tie_weights': True,
            'label_smoothing': 0.1,
            'img_layer_norm_eps': 1e-5,
            'max_img_seq_length': 50,
            'max_gen_length': 20,

            'output_isvalid': False,

            'max_masked_tokens': 3,
            'cider_cached_tokens': 'data/coco_caption/gt/coco-train-words.p',

            'num_beams': 1,

            'mask_prob': 0.15,
            'replace_by_mask_prob': 0.8,
            'replace_by_rand_prob': 0.1,

            'temperature': 1,
            'top_k': 0,
            'top_p': 1,
            'gradient_clip': 1.,

            'optimizer_type': 'MAdamW',
            'bias_no_weight_decay': True,
            'ln_no_weight_decay': True,

            'unique_labels_on': False,

            'scheduler_type': 'linear',
            'pad_to_max': True,
            'no_sort_by_conf': False,

            'ignore_project_image': False,
            'real_text_a_in_test': False
        })
        self._tokenizer = None
        self._test_caption_tensorizer = None
        self._train_caption_tensorizer = None

        if self.cfg.pad_to_max:
            from torch.utils.data.dataloader import default_collate
            self.train_collate_fn = default_collate
            self.test_collate_fn = default_collate
        else:
            self.train_collate_fn = collate_fn
            self.test_collate_fn = collate_fn

        max_seq_length = self.cfg.max_seq_length
        if not self.cfg.add_od_labels:
            assert self.cfg.max_seq_a_length == max_seq_length
        else:
            assert self.cfg.max_seq_a_length == 40

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

        if is_train:
            caption_loader = LoadCaption(
                data=data, split=split, version=None,
                cache_policy=cache_policy,
            )
            all_trans.append(caption_loader)

        if self.cfg.add_od_labels:
            label_loader = LoadLabel(
                data=data, split=split,
                version=self.cfg.train_label_version)
            all_trans.append(label_loader)

        text_ab = IdentifyTextAB(
            self.cfg.add_od_labels,
            self.cfg.od_label_conf,
            label_sort_by_conf=not self.cfg.no_sort_by_conf,
            unique_labels_on=self.cfg.unique_labels_on,
            qa2caption=None,
            sep_token=self.tokenizer.sep_token,
        )
        all_trans.append(text_ab)

        tensorizer = (self.train_caption_tensorizer if is_train else
                      self.test_caption_tensorizer)
        from qd.data_layer.transform import TransCaptionTensorizer
        if not is_train:
            assert self.cfg.pad_to_max, 'not ready'
        trans_tensorizer = TransCaptionTensorizer(
            tensorizer,
            with_img_feats=load_feature,
            pad_to_max=self.cfg.pad_to_max,
            pad_image_to_max=True,
            real_text_a_in_test=self.cfg.real_text_a_in_test
        )
        all_trans.append(trans_tensorizer)

        useless_keys = [
            'idx',
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

    def get_raw_model(self, is_train):
        from qd.mask.layers.bert import BertConfig
        config = BertConfig.from_pretrained(
            self.cfg.text_encoder_type,
            num_labels=2,
            finetuning_task='image_captioning',
        )
        config.img_feature_type = 'frcnn'
        config.hidden_dropout_prob = self.cfg.drop_out
        config.loss_type = 'classification'
        config.tie_weights = self.cfg.tie_weights
        config.freeze_embedding = False
        config.label_smoothing = self.cfg.label_smoothing
        config.drop_worst_ratio = 0
        config.drop_worst_after = 0
        config.img_feature_dim = self.cfg.img_feature_dim
        config.use_img_layernorm = self.cfg.use_img_layernorm
        config.img_layer_norm_eps = self.cfg.img_layer_norm_eps
        config.ignore_project_image = self.cfg.ignore_project_image

        image_encoder = None
        if self.cfg.max_img_seq_length == 0:
            image_encoder = self.get_image_encoder(is_train)

        model = BertForImageCaptioning(config=config) # init from scratch
        if is_train:
            model = ImageCaptioning(model, image_encoder=image_encoder,
                                    cfg=self.cfg)
        else:
            tokenizer = self.tokenizer
            cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
                tokenizer.convert_tokens_to_ids([
                    tokenizer.cls_token,
                    tokenizer.sep_token,
                    tokenizer.pad_token,
                    tokenizer.mask_token,
                    '.',
                ])
            test_extra_input = {
                'is_decode': True,
                'do_sample': False,
                'bos_token_id': cls_token_id,
                'pad_token_id': pad_token_id,
                'eos_token_ids': [sep_token_id],
                'mask_token_id': mask_token_id,
                # for adding od labels
                'add_od_labels': self.cfg.add_od_labels,
                'od_labels_start_posid': self.cfg.max_seq_a_length,

                # hyperparameters of beam search
                'max_length': self.cfg.max_gen_length,
                'num_beams': self.cfg.num_beams,
                "temperature": self.cfg.temperature,
                "top_k": self.cfg.top_k,
                "top_p": self.cfg.top_p,
                "repetition_penalty": 1,
                "length_penalty": 1,
                "num_return_sequences": 1,
                "num_keep_best": 1,
            }
            model = ImageCaptioning(
                model, test_extra_input, image_encoder=image_encoder,
                cfg=self.cfg,
            )

        return model

    def get_image_encoder(self, is_train):
        if self.cfg.image_encoder_type.startswith('timm_'):
            net = self.cfg.image_encoder_type[5:]
            import timm
            model = timm.create_model(
                net,
                output_grid=True,
                pretrained=False,
            )
            if not is_train:
                model.eval()
            from qd.torch_common import InputAsDict
            model = InputAsDict(model)
        elif self.cfg.image_encoder_type.startswith('vit'):
            parts = list(self.cfg.image_encoder_type.split('_'))[1:]
            depth, embed_dim, patch_size, num_heads = 12, 386, 16, 12
            for p in parts:
                if p.startswith('d'):
                    depth = int(p[1:])
                elif p.startswith('h'):
                    embed_dim = int(p[1:])
                elif p.startswith('p'):
                    patch_size = int(p[1:])
                elif p.startswith('a'):
                    num_heads = int(p[1:])
                else:
                    raise NotImplementedError
            if depth == 0:
                # image encoder has done projection
                assert self.cfg.ignore_project_image
                assert not self.cfg.use_img_layernorm
            model_kwargs = dict(patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                                num_heads=num_heads)
            img_size = self.cfg.train_crop_size if is_train else self.cfg.test_crop_size
            from timm.models.vision_transformer import VisionTransformer
            model = VisionTransformer(img_size=img_size, num_classes=-1, output_grid=True, **model_kwargs)
            if not is_train:
                model.eval()
            from qd.torch_common import InputAsDict
            model = InputAsDict(model)
        else:
            raise NotImplementedError(self.cfg.image_encoder_type)
        return model

    def predict_output_to_tsv_row(self, data, output):
        all_caps = output[0]  # batch_size * num_keep_best * max_len
        all_confs = torch.exp(output[1])

        for img_key, caps, confs in zip(data['key'], all_caps, all_confs):
            res = []
            for cap, conf in zip(caps, confs):
                cap = self.tokenizer.decode(
                    cap.tolist(), skip_special_tokens=True)
                res.append({'caption': cap, 'conf': conf.item()})
            yield img_key, json.dumps(res)

    def evaluate(self, predict_file, evaluate_file):
        from qd.tsv_io import TSVDataset
        dataset = TSVDataset(self.cfg.test_data)
        json_caption = op.join(
            dataset._data_root,
            self.cfg.test_split + '.caption_coco_format.json')
        if not op.isfile(json_caption):
            from qd.process_tsv import iter_caption_to_json
            iter_caption_to_json(
                dataset.iter_data(
                    self.cfg.test_split, 'caption'),
                json_caption)
        from qd.mask.modeling.captioning.utils_caption_evaluate import evaluate_on_coco_caption
        result = evaluate_on_coco_caption(predict_file, json_caption, outfile=evaluate_file)
        logging.info('evaluation result: {}'.format(str(result)))
        logging.info('evaluation result saved to {}'.format(evaluate_file))

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from qd.mask.layers.bert import BertTokenizer
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
                max_img_seq_length=self.cfg.max_img_seq_length,
                max_seq_length=self.cfg.max_seq_length,
                max_seq_a_length=self.cfg.max_seq_a_length,
                mask_prob=self.cfg.mask_prob,
                max_masked_tokens=self.cfg.max_masked_tokens,
                mask_type=self.cfg.mask_type,
                is_train=True,
                mask_b=False,
                replace_by_mask_prob=self.cfg.replace_by_mask_prob,
                replace_by_rand_prob=self.cfg.replace_by_rand_prob,
                output_isvalid=self.cfg.output_isvalid,
            )
            self._train_caption_tensorizer = caption_tensorizer
        return self._train_caption_tensorizer

    @property
    def test_caption_tensorizer(self):
        if self._test_caption_tensorizer is None:
            max_seq_length = self.cfg.max_seq_length if self.cfg.add_od_labels else self.cfg.max_gen_length
            max_od_labels_len = self.cfg.max_seq_length - self.cfg.max_seq_a_length
            max_seq_length = self.cfg.max_gen_length + max_od_labels_len
            from qd.mask.data.datasets.caption_tensorizer import CaptionTensorizer
            caption_tensorizer = CaptionTensorizer(
                self.tokenizer,
                max_img_seq_length=self.cfg.max_img_seq_length,
                max_seq_length=max_seq_length,
                max_seq_a_length=self.cfg.max_gen_length,
                is_train=False,
            )
            self._test_caption_tensorizer = caption_tensorizer
        return self._test_caption_tensorizer

