import json
import base64
import random
import torch
import random
import os.path as op
import numpy as np

from qd.mask.structures.tsv_file import CompositeTSVFile
from qd.tsv_io import TSVFile
from qd.tsv_io import tsv_reader
from qd.mask.modeling.captioning.utils_cbs import (ConstraintFilter,
                       ConstraintBoxesReader, FiniteStateMachineBuilder)
from .utils.load_files import load_from_yaml_file, find_file_path_in_yaml
from .utils.load_files import load_box_linelist_file


class CaptionTSVDataset(object):
    def __init__(self, yaml_file, tensorizer, tokenizer=None, 
                 add_od_labels=True, od_label_conf=0.0, sort_by_conf=True,
                 unique_labels_on=False, is_train=True, on_memory=False,
                 img_feature_dim=None,
                 ):
        """Constructor.
        Args:
            yaml file with all required data (caption, feature, labels, etc)
            tensorizer: process inputs to tensor formats, add mask, etc.
            tokenizer: tokenizer for text processing.
            add_od_labels: whether to add labels from yaml file to BERT. 
            od_label_conf: threshold to select od labels.
            is_train: train or test mode.
            sort_by_conf: sort feature/labels by confidence (truncate feature or 
                 labels with lower confidence.)
            unique_labels_on: use only unique labels as tags. 
            on_memory: load caption/labels into memory, it can save some computation 
                during training. 
        """
        self.yaml_file = yaml_file
        self.cfg = load_from_yaml_file(yaml_file)
        # gradually to use qd_format, which is better to decouple the data
        # format and the training logic
        self.qd_format = self.cfg.get('qd_format', False)
        self.is_composite = self.cfg.get('composite', False)
        self.root = op.dirname(yaml_file)
        self.feat_file = self.cfg['feature']
        self.hw_file = self.cfg['hw']
        self.label_file = self.cfg.get('label', None)
        self.cap_file = self.cfg.get('caption', None)

        if not self.qd_format:
            self.cap_linelist_file = find_file_path_in_yaml(self.cfg.get('caption_linelist', None), self.root)

        self.feat_tsv = self.get_tsv_file(self.feat_file)
        self.hw_tsv = self.get_tsv_file(self.hw_file, on_memory)
        self.label_tsv = self.get_tsv_file(self.label_file)
        self.cap_tsv = self.get_tsv_file(self.cap_file, on_memory)
        # set it back always to false
        on_memory = False

        if not self.qd_format:
            if self.is_composite:
                assert op.isfile(self.cap_linelist_file)
                self.cap_line_list = [int(row[2]) for row in tsv_reader(self.cap_linelist_file)]
                self.img_line_list = [i for i in range(len(self.cap_line_list))]
            elif self.cap_linelist_file:
                line_list = load_box_linelist_file(self.cap_linelist_file)
                self.img_line_list = line_list[0]
                self.cap_line_list = line_list[1]
            else:
                # one caption per image
                self.img_line_list = [i for i in range(self.feat_tsv.num_rows())]
                self.cap_line_list = [0 for i in range(self.feat_tsv.num_rows())]

        if add_od_labels: assert self.label_tsv is not None
        if is_train:
            assert self.cap_tsv is not None
            assert tokenizer is not None

        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.add_od_labels = add_od_labels
        self.od_label_conf = od_label_conf
        self.sort_by_conf = sort_by_conf
        self.unique_labels_on = unique_labels_on
        self.is_train = is_train
        self._image_keys = None
        self._key2index = None
        self.img_feature_dim = img_feature_dim

        self.on_memory = on_memory
        self.labels_loaded_to_memory = False
        if on_memory:
            if self.add_od_labels:
                self.load_labels_to_memory()
                self.labels_loaded_to_memory = True
            if self.cap_tsv is not None:
                self.load_caption_to_memory()

    @property
    def image_keys(self):
        if self._image_keys is None:
            self._image_keys = self.prepare_image_keys()
        return self._image_keys

    @property
    def key2index(self):
        if self._key2index is None:
            self._key2index = self.prepare_image_key_to_index()
        return self._key2index

    def get_image_key(self, idx):
        tsv = self.get_valid_tsv()
        return tsv.get_key(idx)

    def load_labels_to_memory(self):
        self.labels_on_memory = {}
        for img_idx in set(self.img_line_list):
            self.labels_on_memory[img_idx] = self.get_od_labels(img_idx)

    def load_caption_to_memory(self):
        self.caption_on_memory = {}
        for img_idx in set(self.img_line_list):
            row = self.get_row_from_tsv(self.cap_tsv, img_idx)
            for cap_idx, data in enumerate(json.loads(row[1])):
                self.caption_on_memory[(img_idx, cap_idx)] = data['caption']

    def get_tsv_file(self, tsv_file, on_memory=False):
        if self.qd_format:
            if isinstance(tsv_file, str):
                tsv_path = find_file_path_in_yaml(tsv_file, self.root)
                return TSVFile(tsv_path)
            elif isinstance(tsv_file, dict):
                from qd.data_layer.dataset import TSVSplitProperty
                return TSVSplitProperty(
                    **tsv_file,
                    cache_policy=None)
        else:
            # this will be deprecated
            if tsv_file:
                if self.is_composite:
                    return CompositeTSVFile(tsv_file, self.cap_linelist_file,
                            root=self.root, cache_policy='memory' if on_memory
                                            else None)
                tsv_path = find_file_path_in_yaml(tsv_file, self.root)
                return TSVFile(tsv_path, cache_policy='memory' if on_memory
                               else None)

    def get_valid_tsv(self):
        if self.is_train:
            return self.cap_tsv
        # sorted by file size
        if self.hw_tsv:
            return self.hw_tsv
        if self.cap_tsv:
            return self.cap_tsv
        if self.label_tsv:
            return self.label_tsv
        if self.feat_tsv:
            return self.feat_tsv

    def prepare_image_keys(self):
        tsv = self.get_valid_tsv()
        return [tsv.get_key(i) for i in range(tsv.num_rows())]

    def prepare_image_key_to_index(self):
        tsv = self.get_valid_tsv()
        return {tsv.get_key(i) : i for i in range(tsv.num_rows())}

    def get_image_cap_index(self, idx):
        return self.img_line_list[idx], self.cap_line_list[idx]

    def get_row_from_tsv(self, tsv, img_idx):
        row = tsv[img_idx]
        #if self.is_composite:
            #assert self.image_keys[img_idx].endswith(row[0]), (
                #row[0], img_idx,
                #self.image_keys[img_idx])
        #else:
            #assert row[0] == self.image_keys[img_idx]
        return row

    def get_hw_info(self, img_idx):
        row = self.get_row_from_tsv(self.hw_tsv, img_idx)
        try:
            hw_info = json.loads(row[1])
        except:
            h, w = map(int, row[1].split(' '))
            hw_info = {'height': h, 'width': w}
        if type(hw_info) == dict:
            return hw_info
        if type(hw_info) == list:
            return hw_info[0]

    def get_spatial_features(self, feat_info, img_idx):
        hw_info = self.get_hw_info(img_idx)
        img_height, img_width = hw_info['height'], hw_info['width']
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
        return spatial_feats

    def get_image_features_dict(self, img_idx):
        if not self.qd_format:
            row = self.get_row_from_tsv(self.feat_tsv, img_idx)
            feat_info = json.loads(row[1])
            if self.sort_by_conf and any('conf' in f for f in feat_info):
                feat_info = sorted(feat_info, key = lambda x : -x['conf'])

            max_img_seq_len = self.tensorizer.max_img_seq_len
            if len(feat_info) > max_img_seq_len:
                if any('conf' in f for f in feat_info) and self.sort_by_conf:
                    feat_info = feat_info[:max_img_seq_len]
                else:
                    step = len(feat_info) // max_img_seq_len
                    feat_info = feat_info[0:(step * max_img_seq_len):step]
                    assert len(feat_info) == max_img_seq_len, (
                        max_img_seq_len,
                        step, len(feat_info))

            if len(feat_info) == 0:
                return {'feat': torch.zeros((0, self.img_feature_dim)),
                        'key': row[0]}

            if all('feature' in f for f in feat_info):
                feats = [np.frombuffer(base64.b64decode(f['feature']), np.float32) for f in feat_info]
            else:
                from qd.qd_common import decode_np
                feats = [decode_np(f['zlib_feature']).astype(np.float32) for f in feat_info]
            if any('rect' in f for f in feat_info):
                spatial_feats = self.get_spatial_features(feat_info, img_idx)
                return {'feat': torch.Tensor(np.concatenate((feats, spatial_feats), 1)),
                        'key': row[0]}
            else:
                # grid feature, where there is no box location
                return {'feat': torch.Tensor(feats),
                        'key': row[0]}
        else:
            row = self.feat_tsv[img_idx]
            feat_info = row[1]
            if self.sort_by_conf:
                feat_info = sorted(feat_info, key = lambda x : -x['conf'])
            if all('feature' in f for f in feat_info):
                feats = [np.frombuffer(base64.b64decode(f['feature']), np.float32) for f in feat_info]
            else:
                from qd.qd_common import decode_np
                feats = [decode_np(f['zlib_feature']).astype(np.float32)
                         for f in feat_info]
            if len(feats) == 0:
                return {'feat': torch.zeros((0, self.img_feature_dim)),
                        'key': row[0]}
            if any('rect' in f for f in feat_info):
                spatial_feats = self.get_spatial_features(feat_info, img_idx)
                return {'feat': torch.Tensor(np.concatenate((feats, spatial_feats), 1)),
                        'key': row[0]}
            else:
                # grid feature, where there is no box location
                return {'feat': torch.Tensor(feats),
                        'key': row[0]}

    def get_image_features(self, img_idx, key=None):
        # prefer to use get_image_features_dict
        feat_info = self.get_image_features_dict(img_idx)
        if key:
            assert key.endswith(feat_info['key'])
        return feat_info['feat']

        #if not self.qd_format:
            #row = self.get_row_from_tsv(self.feat_tsv, img_idx)
            #if key:
                #assert key.endswith(row[0])
            #feat_info = json.loads(row[1])
            #if self.sort_by_conf and any('conf' in f for f in feat_info):
                #feat_info = sorted(feat_info, key = lambda x : -x['conf'])
            #if all('feature' in f for f in feat_info):
                #feats = [np.frombuffer(base64.b64decode(f['feature']), np.float32) for f in feat_info]
            #else:
                #from qd.qd_common import decode_np
                #feats = [decode_np(f['zlib_feature']).astype(np.float32) for f in feat_info]
            #spatial_feats = self.get_spatial_features(feat_info, img_idx)
            #if len(feats) == 0:
                #return torch.zeros((0, self.img_feature_dim))
            #return torch.Tensor(np.concatenate((feats, spatial_feats), 1))
        #else:
            #assert key is None
            #feat_info = self.feat_tsv[img_idx][1]
            #if self.sort_by_conf:
                #feat_info = sorted(feat_info, key = lambda x : -x['conf'])
            #if all('feature' in f for f in feat_info):
                #feats = [np.frombuffer(base64.b64decode(f['feature']), np.float32) for f in feat_info]
            #else:
                #from qd.qd_common import decode_np
                #feats = [decode_np(f['zlib_feature']).astype(np.float32)
                         #for f in feat_info]
            #spatial_feats = self.get_spatial_features(feat_info, img_idx)
            #if len(feats) == 0:
                #return torch.zeros((0, self.img_feature_dim))
            #return torch.Tensor(np.concatenate((feats, spatial_feats), 1))

    def get_caption_dict(self, img_idx, cap_idx):
        if self.is_train:
            if not self.qd_format:
                if self.on_memory:
                    raise NotImplementedError(
                        'return value type is not correct yet')
                    return self.caption_on_memory[(img_idx, cap_idx)]
                row = self.get_row_from_tsv(self.cap_tsv, img_idx)
                r = json.loads(row[1])[cap_idx]
                r['key'] = row[0]
                return r
            else:
                row = self.cap_tsv[img_idx]
                rects = json.loads(row[1])
                if len(rects) == 0:
                    return {'caption': '', 'key': row[0]}
                else:
                    cap_idx = random.choices(range(len(rects)))
                    r = json.loads(row[1])[cap_idx]
                    r['key'] = row[0]
                    return r
        return {'caption': ''}

    def get_caption(self, img_idx, cap_idx, key=None):
        # use get_caption_dict since some caption here has no 'caption' field
        # and there is question-answer
        if self.is_train:
            if not self.qd_format:
                if self.on_memory:
                    return self.caption_on_memory[(img_idx, cap_idx)]
                row = self.get_row_from_tsv(self.cap_tsv, img_idx)
                if key:
                    assert key.endswith(row[0])
                return json.loads(row[1])[cap_idx]['caption']
            else:
                assert key is None
                row = self.cap_tsv[img_idx]
                rects = json.loads(row[1])
                if len(rects) == 0:
                    return ''
                else:
                    cap_idx = random.choices(range(len(rects)))
                    return json.loads(row[1])[cap_idx]['caption']
        return ""

    def get_od_labels_dict(self, img_idx):
        if self.add_od_labels:
            if self.labels_loaded_to_memory:
                raise NotImplementedError(
                    'return type is not correct yet')
                return self.labels_on_memory[img_idx]

            if not self.qd_format:
                row = self.get_row_from_tsv(self.label_tsv, img_idx)
            else:
                row = self.label_tsv[img_idx]
            label_info = json.loads(row[1])
            for lab in label_info:
                if 'conf' not in lab:
                    lab['conf'] = 1.0
            if len(label_info) > 0 and self.od_label_conf > 0 and 'conf' in label_info[0]:
                # select labels based on confidence
                label_info = [l for l in label_info if l['conf'] >= self.od_label_conf]
            if self.sort_by_conf:
                label_info = sorted(label_info, key = lambda x : -x['conf'])
            if self.unique_labels_on:
                # keep the order so it is deterministic
                label_list = []
                for lab in label_info:
                    if lab['class'].lower() not in label_list:
                        label_list.append(lab['class'].lower())
                return {'label': ' '.join(label_list),
                        'key': row[0]}
            else:
                return {'label': ' '.join([l['class'] for l in label_info]),
                        'key': row[0]}

    def get_od_labels(self, img_idx, key=None):
        # prefer to use get_od_labels_dict
        if self.add_od_labels:
            if self.labels_loaded_to_memory:
                return self.labels_on_memory[img_idx]

            if not self.qd_format:
                row = self.get_row_from_tsv(self.label_tsv, img_idx)
            else:
                row = self.label_tsv[img_idx]
            label_info = json.loads(row[1])
            if key is not None:
                assert key.endswith(row[0])
            for lab in label_info:
                if 'conf' not in lab:
                    lab['conf'] = 1.0
            if len(label_info) > 0 and self.od_label_conf > 0 and 'conf' in label_info[0]:
                # select labels based on confidence
                label_info = [l for l in label_info if l['conf'] >= self.od_label_conf]
            if self.sort_by_conf:
                label_info = sorted(label_info, key = lambda x : -x['conf'])
            if self.unique_labels_on:
                # keep the order so it is deterministic
                label_list = []
                for lab in label_info:
                    if lab['class'].lower() not in label_list:
                        label_list.append(lab['class'].lower())
                return ' '.join(label_list)
            else:
                return ' '.join([l['class'] for l in label_info])

    def get_caption_file_in_coco_format(self):
        # for evaluation
        cap_file_coco_format = find_file_path_in_yaml(self.cfg.get('caption_coco_format', 
            None), self.root)
        if cap_file_coco_format:
            return cap_file_coco_format
        test_split = op.basename(self.yaml_file).split('.')[0]
        return op.join(self.root, test_split + '_caption_coco_format.json')

    def get_captions_by_key(self, key):
        # get a list of captions for image (by key)
        img_idx = self.key2index[key]
        cap_info = json.loads(self.cap_tsv[img_idx][1])
        return [c['caption'] for c in cap_info]

    def __getitem__(self, idx):
        if not self.qd_format:
            img_idx, cap_idx = self.get_image_cap_index(idx)
            img_key = self.image_keys[img_idx]
            #features = self.get_image_features(img_idx, key=img_key)
            features_dict = self.get_image_features_dict(img_idx)
            features = features_dict['feat']
            assert features_dict['key'] in img_key
            caption = self.get_caption(img_idx, cap_idx, key=img_key)
            od_labels = self.get_od_labels(img_idx, key=img_key)
            example = self.tensorizer.tensorize_example(caption, features, text_b=od_labels)
        else:
            raise NotImplementedError
            img_key = self.hw_tsv[idx][0]
            features = self.get_image_features(idx)
            caption = self.get_caption(idx, None)
            od_labels = self.get_od_labels(idx)

        return img_key, example

    def __len__(self):
        if not self.qd_format:
            return len(self.img_line_list)
        else:
            return len(self.hw_tsv)

class PretrainCaptionTSVDataset(CaptionTSVDataset):
    def __init__(self, yaml_file, tensorizer, tokenizer=None, 
                 add_od_labels=True, od_label_conf=0.0, sort_by_conf=True,
                 unique_labels_on=False, is_train=True,
                 pert_caption_prob=0.25, pert_labels_prob=0.25,
                 mask_loss_for_unmatched=False, on_memory=False,
                 img_feature_dim=None,
                 qa2caption=None,
                 label_type=None,
                 ):
        """
           pert_caption_prob: probability to pertube caption for contrastive loss.
           pert_labels_prob: probability to pertube labels for contrastive loss.
           mask_loss_for_unmatched: mask caption/label even if it is not matched to the image.
           if both probabilities are 0, it is equal to no contrastive loss.
        """
        self.pert_caption_prob = pert_caption_prob
        self.pert_labels_prob = pert_labels_prob
        self.pert_prob = pert_caption_prob + pert_labels_prob
        self.mask_loss_for_unmatched = mask_loss_for_unmatched
        super(PretrainCaptionTSVDataset, self).__init__(
                yaml_file,
                tensorizer=tensorizer,
                tokenizer=tokenizer,
                add_od_labels=add_od_labels,
                od_label_conf=od_label_conf,
                sort_by_conf=sort_by_conf,
                is_train=is_train,
                unique_labels_on=unique_labels_on,
                on_memory=on_memory,
            img_feature_dim=img_feature_dim
        )
        assert qa2caption in [None, 'QT_A', 'Q_TA',
                              'QST_A', 'Q_TSA', 'QA_T']
        self.qa2caption = qa2caption
        self.label_type = label_type


    def load_text_ab(self, img_idx, cap_idx):
        # currently, this function is used to load information for current
        # instance and the negative instance. If we'd like to have different
        # behaviors for the negative instance, we can add options to this
        # function
        caption_dict = self.get_caption_dict(img_idx, cap_idx)
        if self.add_od_labels:
            od_labels_dict = self.get_od_labels_dict(img_idx)
            od_labels = od_labels_dict['label']
            assert od_labels_dict['key'] == caption_dict['key']
        else:
            od_labels = ''
        img_key = caption_dict['key']

        if 'caption' in caption_dict:
            caption = caption_dict['caption']
        else:
            question = caption_dict['question']
            if 'shortAnswer' in caption_dict:
                answer = caption_dict['shortAnswer']
            elif 'answer' in caption_dict:
                answer = caption_dict['answer']
            else:
                answer = ' '.join(caption_dict['answers'])
            if self.qa2caption is None:
                caption = question
                od_labels = answer
            elif self.qa2caption == 'QT_A':
                caption = question + od_labels
                od_labels = answer
            elif self.qa2caption == 'Q_TA':
                caption = question
                # no need to add whitespace around sep-token
                od_labels = od_labels + self.tensorizer.tokenizer.sep_token + answer
            elif self.qa2caption == 'QST_A':
                # no need to add whitespace around sep-token
                caption = question + self.tensorizer.tokenizer.sep_token + od_labels
                od_labels = answer
            elif self.qa2caption == 'QA_T':
                caption = question + answer
            else:
                raise NotImplementedError
        if not self.add_od_labels:
            od_labels = None
        return {
            'key': img_key,
            'text_a': caption,
            'text_b': od_labels,
        }

    def __getitem__(self, idx):
        img_idx, cap_idx = self.get_image_cap_index(idx)

        features_dict = self.get_image_features_dict(img_idx)
        features = features_dict['feat']
        text_ab_info = self.load_text_ab(img_idx, cap_idx)
        caption = text_ab_info['text_a']
        od_labels = text_ab_info['text_b']
        assert features_dict['key'] == text_ab_info['key']

        rand_num = random.random()
        cap_changed = False
        if rand_num <= self.pert_prob:
            # randomly select caption or labels from a different image to form a negative pair
            find_negative = False
            while not find_negative:
                rand_idx = random.randint(0, len(self.img_line_list) - 1)
                neg_img_idx, neg_cap_idx = self.get_image_cap_index(rand_idx)
                if neg_img_idx != img_idx:
                    find_negative = True
            neg_text_ab = self.load_text_ab(neg_img_idx, neg_cap_idx)
            if rand_num <= self.pert_caption_prob:
                caption = neg_text_ab['text_a']
                cap_changed = True
            else:
                od_labels = neg_text_ab['text_b']
            is_matched = 0
        else:
            is_matched = 1

        x = self.tensorizer.tensorize_example(
                caption, features, text_b=od_labels, return_dict=True,
        )
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        segment_ids = x['segment_ids']
        img_feats = x['img_feats']
        masked_pos = x['masked_pos']
        masked_ids = x['masked_ids']

        # pretrain model needs a slightly different format of inputs, -1 is ignored in loss computation
        lm_labels_id_img = torch.ones(img_feats.shape[0], dtype=torch.long) * (-1)
        lm_labels_id_text = torch.ones_like(input_ids) * (-1)
        if not self.mask_loss_for_unmatched and is_matched == 0:
            # no masked loss for unmatched part
            num_masked_cap = sum(masked_pos[:self.tensorizer.max_seq_a_len]).item()
            if cap_changed:
                # no masked loss for caption
                masked_pos[:self.tensorizer.max_seq_a_len] = 0
                masked_ids[:num_masked_cap] = 0
            else:
                # no masked loss for labels
                masked_pos[self.tensorizer.max_seq_a_len:] = 0
                masked_ids[num_masked_cap:] = 0
        lm_labels_id_text[masked_pos==1] = masked_ids[masked_ids!=0]
        lm_labels_id = torch.cat((lm_labels_id_text, lm_labels_id_img))

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': segment_ids,
            'img_feats': img_feats,
            'masked_lm_labels': lm_labels_id,
            'next_sentence_label': torch.tensor(is_matched),
        }

        if self.label_type == 'hot':
            from qd.torch_common import convert_single_label_to_one_hot_label
            result['masked_lm_labels'] = convert_single_label_to_one_hot_label(
                lm_labels_id, self.tensorizer.tokenizer.vocab_size)
            result['next_sentence_label'] = convert_single_label_to_one_hot_label(
                torch.tensor(is_matched).long(), 2)

        return result

class CaptionTSVDatasetWithConstraints(CaptionTSVDataset):
    r"""
    Providing inputs for inference with Constraint Beam Search

    nms_threshold: float, optional (default = 0.85)
        NMS threshold for suppressing generic object class names during constraint filtering,
        for two boxes with IoU higher than this threshold, "dog" suppresses "animal".
    max_given_constraints: int, optional (default = 3)
        Maximum number of constraints which can be specified for CBS decoding. Constraints are
        selected based on the prediction confidence score of their corresponding bounding boxes.
    """

    def __init__(
        self, yaml_file, tensorizer,
        nms_threshold=0.85,
        max_given_constraints=3, **kwargs
    ):
        super().__init__(yaml_file, tensorizer, **kwargs)
        boxes_tsvpath = find_file_path_in_yaml(self.cfg['cbs_box'], self.root)
        constraint2tokens_tsvpath = find_file_path_in_yaml(self.cfg['cbs_constraint'], self.root)
        tokenforms_tsvpath = find_file_path_in_yaml(self.cfg['cbs_tokenforms'], self.root)
        hierarchy_jsonpath = find_file_path_in_yaml(self.cfg['cbs_hierarchy'], self.root)

        self._boxes_reader = ConstraintBoxesReader(boxes_tsvpath)
        self._constraint_filter = ConstraintFilter(
            hierarchy_jsonpath, nms_threshold, max_given_constraints
        )
        self._fsm_builder = FiniteStateMachineBuilder(self.tokenizer,
                constraint2tokens_tsvpath, tokenforms_tsvpath,
                max_given_constraints)

    def __getitem__(self, index):
        img_key, example = super().__getitem__(index)

        # Apply constraint filtering to object class names.
        constraint_boxes = self._boxes_reader[img_key]

        candidates = self._constraint_filter(
            constraint_boxes["boxes"], constraint_boxes["class_names"], constraint_boxes["scores"]
        )
        num_constraints = len(candidates)
        fsm, nstates = self._fsm_builder.build(candidates)

        return img_key, example + (fsm, num_constraints, )


