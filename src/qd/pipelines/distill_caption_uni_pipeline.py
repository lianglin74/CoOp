from qd.mask.layers.bert import BertConfig
import os.path as op
import logging
from torchvision.transforms import transforms
from qd.data_layer.transform import TransCaptionTensorizer
import torch
from qd.pipelines.caption_uni_pipeline import CaptionUniPipeline
from qd.data_layer.transform import (
    LoadLabel,
    LoadHW,
    LoadFeature,
    PadFeature,
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
    StudentTensorizer,
)
from qd.data_layer.dataset import CaptionIdxTSVDataset, ImageIdxTSVDataset
from qd.pipelines.uni_pipeline import UniPipeline
from torch import nn
from qd.torch_common import torch_load, torch_save


def merge_teacher_student_as_base(
        teacher_model, student_model, basemodel):
    if op.isfile(basemodel):
        logging.info('ignore to generate {}'.format(basemodel))
        return
    teacher_model = op.join(teacher_model, 'pytorch_model.bin')
    if op.isfile(student_model):
        prefix = 'module.'
        student = {}
        for k, v in torch_load(student_model)['model'].items():
            while k.startswith(prefix):
                k = k[len(prefix):]
            if 'teacher.' in k:
                continue
            student[k] = v
    else:
        student_model = op.join(student_model, 'pytorch_model.bin')
        student = torch_load(student_model)

    teacher = torch_load(teacher_model)
    teacher = dict(('teacher.' + k, v) for k, v in teacher.items())
    student = dict(('module.' + k, v) for k, v in student.items())
    student.update(teacher)
    torch_save(
        {'model': student},
        basemodel,
    )
    from qd.qd_common import write_to_file
    write_to_file(
        'converted from {} \nand {}'.format(teacher_model, student_model),
        basemodel + '.info',
    )

class DistillCaptionBert(nn.Module):
    def __init__(self,
                 teacher_config,
                 config,
                 teacher_temperature=1.,
                 temperature=1.,
                 distill_cls_weight=1.,
                 distill_attn_weight=None,
                 distill_hiddn_weight=None,
                 distill_attn_style=None,
                 ):
        super().__init__()
        from qd.qd_common import print_frame_info
        print_frame_info()
        if distill_attn_weight is not None:
            config.output_attentions = True
            teacher_config.output_attentions = True
        if distill_hiddn_weight is not None:
            config.output_hidden_states = True
            teacher_config.output_hidden_states = True

        from qd.mask.layers.bert.modeling_bert import BertForImageCaptioning
        self.teacher = BertForImageCaptioning(teacher_config)
        self.teacher.eval()
        for _, p in self.teacher.named_parameters():
            p.requires_grad = False
        self.module = BertForImageCaptioning(config)
        self.teacher_temperature = teacher_temperature
        self.temoperature = temperature

        self.distill_cls_loss = distill_cls_weight
        self.distill_attn_weight = distill_attn_weight
        self.distill_hiddn_weight = distill_hiddn_weight

        self.distill_attn_style = distill_attn_style

        self.iter = 0

    def distill_ce_loss(self, logits, teacher_logits):
        teacher_logits = teacher_logits / self.teacher_temperature
        logits = logits / self.temoperature

        target = teacher_logits.softmax(dim=1)
        x = -target * logits.log_softmax(dim=1)
        return x.sum(dim=1).mean()

    def distill_hiddn_loss(self, hidden, teacher_hidden,
                           valid, teacher_valid):
        hidden = hidden[-1]
        teacher_hidden = teacher_hidden[-1]
        #ipdb> pp hidden.shape
        #torch.Size([2, 120, 384])
        #ipdb> pp teacher_hidden.shape
        #torch.Size([2, 120, 768])
        hidden = torch.nn.functional.normalize(hidden, dim=2)
        teacher_hidden = torch.nn.functional.normalize(teacher_hidden, dim=2)
        sim = torch.bmm(
            hidden, hidden.permute(0, 2, 1))
        teacher_sim = torch.bmm(
            teacher_hidden,
            teacher_hidden.permute(0, 2, 1))
        mask = (1 - torch.bmm(valid[:, :, None].half(), valid[:, None, :].half())) * -10000
        sim += mask
        teacher_sim += mask
        loss = -(teacher_sim.softmax(dim=2) * sim.log_softmax(dim=2)).sum(dim=2).mean()
        return loss

    def distill_attn_loss(self, attn, teacher_attn,
                           valid, teacher_valid, data):
        #ipdb> attn[-1].shape
        #torch.Size([2, 12, 120, 120])
        #ipdb> teacher_attn[-1].shape
        #torch.Size([2, 12, 120, 120])

        if self.distill_attn_style is None:
            attn = attn[-1]
            teacher_attn = teacher_attn[-1]
            attn = attn.permute(0, 2, 3, 1)
            teacher_attn = teacher_attn.permute(0, 2, 3, 1)
            attn = attn[valid]
            teacher_attn = teacher_attn[teacher_valid]
            loss = -(teacher_attn * (attn + 1e-5).log()).sum(dim=1)
            loss = loss.mean()
        elif self.distill_attn_style == 'avg':
            attn = attn[-1]
            teacher_attn = teacher_attn[-1]
            attn = attn.mean(dim=1)
            teacher_attn = teacher_attn.mean(dim=1)
            attn = attn[valid]
            teacher_attn = teacher_attn[teacher_valid]
            loss = -(teacher_attn * (attn + 1e-5).log()).sum(dim=1)
            loss = loss.mean()
        elif self.distill_attn_style == 'avgmask':
            attn = attn[-1]
            teacher_attn = teacher_attn[-1]
            attn = attn.mean(dim=1)
            teacher_attn = teacher_attn.mean(dim=1)

            attn = attn[:, :data['masked_pos'].shape[1]]
            teacher_attn = teacher_attn[:, :data['masked_pos'].shape[1]]

            assert (data['masked_pos'] - data['teacher_masked_pos']).abs().sum() == 0
            attn = attn[data['masked_pos'] == 1]
            teacher_attn = teacher_attn[data['teacher_masked_pos'] == 1]
            loss = -(teacher_attn * (attn + 1e-5).log()).sum(dim=1)
            loss = loss.mean()
        return loss

        #x = torch.bmm(valid[:, :, None].half(), valid[:, None, :].half())
        #x = x > 0.1
        #attn = attn[x]
        #teacher_attn = teacher_attn[x]
        #(attn - teacher_attn).abs().mean()


    def top1_acc(self, logits, label):
        with torch.no_grad():
            logits = torch.max(logits, -1)[1].data # argmax
            return (logits == label).float().mean()

    def forward(self, data):
        valid = data.pop('valid')
        teacher_valid = data.pop('teacher_valid')
        verbose = (self.iter % 100) == 0
        self.iter += 1
        # in test, we need the key, so key is not removed in transform
        key = data.pop('key')
        assert self.training
        student_data = dict([(k, v) for k, v in data.items()
                             if not k.startswith('teacher')])

        output = self.module(**student_data, return_dict=True)

        with torch.no_grad():
            prefix = 'teacher_'
            teacher_data = dict([(k[len(prefix):], v) for k, v in data.items() if
                            k.startswith(prefix)])
            teacher_output = self.teacher(**teacher_data, return_dict=True)
            if verbose:
                acc = self.top1_acc(teacher_output['class_logits'], teacher_output['masked_ids'])
                logging.info('teacher acc = {}'.format(acc))

        ce_loss = self.distill_ce_loss(
            output['class_logits'], teacher_output['class_logits'])
        if verbose:
            acc = self.top1_acc(output['class_logits'], output['masked_ids'])
            logging.info('student acc = {}'.format(acc))

        result = {
            'masked_loss': output['masked_loss'],
            'distill_ce_loss': self.distill_cls_loss * ce_loss,
        }

        if self.distill_hiddn_weight or self.distill_attn_weight:
            assert (valid != teacher_valid).sum() == 0, key

        if self.distill_hiddn_weight is not None:
            #ipdb> hiddn[-1].shape
            #torch.Size([2, 120, 384])
            hidden = output['inter_info'][0]
            #ipdb> pp teacher_hidden[-1].shape
            #torch.Size([2, 120, 768])
            teacher_hidden = teacher_output['inter_info'][0]
            hl = self.distill_hiddn_loss(
                hidden, teacher_hidden,
                valid, teacher_valid,
            )
            result['distill_hiddn_loss'] = self.distill_hiddn_weight * hl
        if self.distill_attn_weight is not None:
            if self.distill_hiddn_weight is not None:
                attn = output['inter_info'][1]
                teacher_attn = teacher_output['inter_info'][1]
            else:
                attn = output['inter_info'][0]
                teacher_attn = teacher_output['inter_info'][0]
            al = self.distill_attn_loss(
                attn,
                teacher_attn,
                valid,
                teacher_valid,
                data,
            )
            result['distill_attn_loss'] = self.distill_attn_weight * al

        return result

class DistillCaptionUniPipeline(CaptionUniPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._default.update({
            'teacher_tie_weights': False,
            'distill_attn_weight': None,
            'distill_hiddn_weight': None,
            'output_isvalid': True,
        })

    def get_transform(self, is_train):
        if not is_train:
            return super().get_transform(is_train)
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

        # load teacher's information
        teacher_feature_loader = self.get_train_tran_load_feature(
            self.cfg.teacher_train_feature_version,
            self.cfg.teacher_img_feature_dim,
        )
        all_trans.append(teacher_feature_loader)

        if self.cfg.add_od_labels:
            teacher_label_loader = self.get_train_tran_load_label(
                self.cfg.teacher_train_label_version
            )
            all_trans.append(teacher_label_loader)

        caption_loader = LoadCaption(
            data=data, split=split, version=None,
            cache_policy=cache_policy,
        )
        all_trans.append(caption_loader)

        text_ab = IdentifyTextAB(
            self.cfg.add_od_labels,
            self.cfg.od_label_conf,
            label_sort_by_conf=self.cfg.no_sort_by_conf,
            unique_labels_on=False,
            qa2caption=None,
            sep_token=self.tokenizer.sep_token,
        )
        all_trans.append(text_ab)
        assert self.cfg.pad_to_max, 'not ready'
        trans_tensorizer = TransCaptionTensorizer(
            self.train_caption_tensorizer,
            with_img_feats=self.cfg.max_img_seq_length > 0,
            pad_to_max=self.cfg.pad_to_max,
            pad_image_to_max=True,
        )
        all_trans.append(trans_tensorizer)

        teacher_remove = RemoveUselessKeys([
            'label',
            'text_ab_type',
            'text_changed',
            'text_a_or_b_changed',
            'img_feat',
            'max_seq_a_len',
            'seq_a_padded_len',
            'feats_conf',
            'feats_class',
            'vocab_size',
            'feats_class_token_ids',
            'feats_class_tokens',
        ])
        all_trans.append(teacher_remove)
        teacher_rename = RenameKey({
            'img_feats': 'teacher_img_feats',
            'segment_ids': 'teacher_token_type_ids',
            'input_ids': 'teacher_input_ids',
            'img_feats': 'teacher_img_feats',
            'attention_mask': 'teacher_attention_mask',
            'masked_ids': 'teacher_masked_ids',
            'masked_pos': 'teacher_masked_pos',
            'text_a': 'teacher_text_a',
            'text_b': 'teacher_text_b',
            'valid': 'teacher_valid',
        })
        all_trans.append(teacher_rename)

        # load student
        feature_loader = self.get_train_tran_load_feature(
            self.cfg.train_feature_version,
            self.cfg.img_feature_dim,
        )
        all_trans.append(feature_loader)

        if self.cfg.add_od_labels:
            label_loader = LoadLabel(
                data=data, split=split,
                version=self.cfg.train_label_version)
            all_trans.append(label_loader)

        all_trans.append(text_ab)

        # no need to do random masking for student as teacher has already done
        # that. student should inherit teacher's masking result
        trans_tensorizer = StudentTensorizer(
            self.train_caption_tensorizer,
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
            # the follow was used by student transforms to check or
            # verification. so it should not be removed in teacher's transform
            'teacher_text_a',
            'teacher_text_b',
        ]
        remove = RemoveUselessKeys(useless_keys)
        all_trans.extend([
            remove,
            RenameKey({'segment_ids': 'token_type_ids'}),
        ])
        return transforms.Compose(all_trans)

    def get_train_tran_load_feature(self, version, dim):
        return LoadFeature(
            data=self.cfg.data,
            split='train',
            version=version,
            img_feature_dim=dim,
            max_len=self.cfg.max_img_seq_length,
            sort_by_conf=self.cfg.no_sort_by_conf,
        )

    def get_train_tran_load_label(self, version):
        return LoadLabel(
                data=self.cfg.data, split='train',
                version=version)

    def deploy(self, model_file=None):
        if model_file is None:
            model_file = self.get_checkpoint_file()
        out_folder = op.splitext(model_file)[0]
        from qd.qd_common import ensure_copy_folder
        ensure_copy_folder(self.cfg.text_encoder_type, out_folder)

        # we need to overwreite the config as we may customize it
        config = self.get_student_config()
        config.save_pretrained(out_folder)

        # need to do some surgery on model param
        model = torch_load(model_file)
        model = dict([(k, v) for k, v in model['model'].items() if 'teacher.' not in
              k])
        prefix = 'module.'
        out_model = {}
        for k, v in model.items():
            while k.startswith(prefix):
                k = k[len(prefix): ]
            out_model[k] = v
        torch_save(out_model, op.join(out_folder, 'pytorch_model.bin'))

        # we need to save training_args.bin
        args = {
            'max_seq_length': self.cfg.max_seq_length,
            'max_seq_a_length': self.cfg.max_seq_a_length,
            'max_gen_length': self.cfg.max_gen_length,
            'do_lower_case': True,
            'add_od_labels': self.cfg.add_od_labels,
            'max_img_seq_length': self.cfg.max_img_seq_length,
            'img_feature_dim': self.cfg.img_feature_dim,
            'od_label_conf': self.cfg.od_label_conf,
            'unique_labels_on': self.cfg.unique_labels_on,
            'no_sort_by_conf': self.cfg.no_sort_by_conf,
        }
        from qd.qd_common import make_namespace_by_dict
        torch_save(make_namespace_by_dict(args), op.join(
            out_folder,
            'training_args.bin'), _use_new_zipfile_serialization=False)

        return out_folder

    def get_raw_model(self, is_train):
        if not is_train:
            return super().get_raw_model(is_train)

        teacher_config = BertConfig.from_pretrained(
            self.cfg.teacher_net, num_labels=2, finetuning_task='image_captioning')
        teacher_config.use_img_layernorm = self.cfg.use_img_layernorm
        teacher_config.img_feature_dim = self.cfg.teacher_img_feature_dim
        teacher_config.img_feature_type = 'frcnn'
        teacher_config.hidden_dropout_prob = self.cfg.drop_out
        teacher_config.loss_type = 'classification'
        teacher_config.tie_weights = self.cfg.teacher_tie_weights
        teacher_config.freeze_embedding = False
        teacher_config.label_smoothing = self.cfg.label_smoothing
        teacher_config.drop_worst_ratio = 0
        teacher_config.drop_worst_after = 0
        teacher_config.img_layer_norm_eps = self.cfg.img_layer_norm_eps

        config = self.get_student_config()
        model = DistillCaptionBert(
            teacher_config,
            config,
            distill_cls_weight=self.cfg.distill_cls_weight,
            distill_attn_weight=self.cfg.distill_attn_weight,
            distill_hiddn_weight=self.cfg.distill_hiddn_weight,
            distill_attn_style=self.cfg.distill_attn_style,
        )

        return model

    def get_student_config(self):
        config = BertConfig.from_pretrained(
            self.cfg.text_encoder_type,
            num_labels=2,
            finetuning_task='image_captioning')
        config.use_img_layernorm = self.cfg.use_img_layernorm
        config.img_feature_dim = self.cfg.img_feature_dim

        config.img_feature_type = 'frcnn'
        config.hidden_dropout_prob = self.cfg.drop_out
        config.loss_type = 'classification'
        config.tie_weights = self.cfg.tie_weights
        config.freeze_embedding = False
        config.label_smoothing = self.cfg.label_smoothing
        config.drop_worst_ratio = 0
        config.drop_worst_after = 0
        config.img_layer_norm_eps = self.cfg.img_layer_norm_eps
        return config

