import torch
import random
import os.path as op
import numpy as np


class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70, 
            max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3,
            mask_type='seq2seq', is_train=True, mask_b=False):
        """Constructor.
        Args:
            tokenizer: tokenizer for text processing.
            max_img_seq_length: max image sequence length.
            max_seq_length: max text sequence length.
            max_seq_a_length: max caption sequence length.
            is_train: train or test mode.
            mask_prob: probability to mask a input token.
            max_masked_tokens: maximum number of tokens to be masked in one sentence.
            mask_type: attention mask type, support seq2seq/bidirectional/cap_s2s/cap_bidir.
            mask_b: whether to mask text_b or not during training.
        """
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_img_seq_len = max_img_seq_length
        self.max_seq_len = max_seq_length
        self.max_seq_a_len = max_seq_a_length
        self.mask_prob = mask_prob
        self.max_masked_tokens = max_masked_tokens
        self.mask_type = mask_type
        self.mask_b = mask_b
        if is_train:
            assert mask_type in ('seq2seq', 'bidirectional', 'cap_s2s', 'cap_bidir')
        else:
            assert mask_type == 'seq2seq'
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))

    def tensorize_example(self, text_a, img_feat, text_b=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1,
                          return_dict=False):
        if self.is_train:
            tokens_a = self.tokenizer.tokenize(text_a)
        else:
            # fake tokens to generate masks
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]

        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            padding_a_len = self.max_seq_a_len - seq_a_len
            tokens += [self.tokenizer.pad_token] * padding_a_len
            segment_ids += ([pad_token_segment_id] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)
        if self.is_train:
            masked_pos = torch.zeros(self.max_seq_len, dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS], [PAD]
            # it is important to mask [SEP] for image captioning as it means [EOS].
            if self.mask_b:
                # can mask both text_a and text_b
                candidate_masked_idx = list(range(1, seq_a_len)) + \
                        list(range(self.max_seq_a_len, seq_len))
                num_masked = min(max(round(self.mask_prob * seq_len), 1), self.max_masked_tokens)
            else:
                # only mask text_a
                candidate_masked_idx = list(range(1, seq_a_len))
                num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            random.shuffle(candidate_masked_idx)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= 0.8:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= 0.5:
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    tokens[pos] = self.tokenizer.get_random_token()
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            masked_pos[masked_idx] = 1 
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                        (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(self.max_seq_len, dtype=torch.int)

        # pad on the right for image captioning
        seq_padding_len = self.max_seq_len - seq_len
        tokens = tokens + ([self.tokenizer.pad_token] * seq_padding_len)
        segment_ids += ([pad_token_segment_id] * seq_padding_len)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            # this case should not happen since we have already filtered before
            # calling this function
            step = img_len // self.max_img_seq_len
            img_feat = img_feat[0:(self.max_img_seq_len * step):step, ]
            #img_feat = img_feat[0 : self.max_img_seq_len, ]
            img_len = img_feat.shape[0]
            assert img_len == self.max_img_seq_len
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        max_len = self.max_seq_len + self.max_img_seq_len
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len
        if self.is_train and self.mask_type == 'bidirectional':
            attention_mask = torch.zeros(max_len, dtype=torch.long)
            attention_mask[c_start : c_end] = 1 # for text_a
            attention_mask[l_start : l_end] = 1 # for text_b if any
            attention_mask[r_start : r_end] = 1 # for image
        elif self.is_train and self.mask_type in ('cap_s2s', 'cap_bidir'):
            # caption is a single modality, and without attention on others
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # no attention between [CLS] and caption
            attention_mask[0, 0] = 1
            if self.mask_type == 'cap_s2s':
                attention_mask[c_start + 1 : c_end, c_start + 1 : c_end].copy_(
                    self._triangle_mask[0 : seq_a_len - 1, 0 : seq_a_len - 1]
                )
            else:
                attention_mask[c_start + 1 : c_end, c_start + 1: c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            # cross attention for L-R, R-L
            attention_mask[l_start : l_end, r_start : r_end] = 1
            attention_mask[r_start : r_end, l_start : l_end] = 1
            # cross attention between [CLS] and L/R
            attention_mask[0, l_start : l_end] = 1
            attention_mask[l_start : l_end, 0] = 1
            attention_mask[0, r_start : r_end] = 1
            attention_mask[r_start : r_end, 0] = 1
        else:
            # prepare attention mask:
            # note that there is no attention from caption to image
            # because otherwise it will violate the triangle attention 
            # for caption as caption will have full attention on image. 
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # triangle mask for caption to caption
            attention_mask[c_start : c_end, c_start : c_end].copy_(
                    self._triangle_mask[0 : seq_a_len, 0 : seq_a_len]
            )
            # full attention for L-L, R-R
            attention_mask[l_start : l_end, l_start : l_end] = 1
            attention_mask[r_start : r_end, r_start : r_end] = 1
            # full attention for C-L, C-R
            attention_mask[c_start : c_end, l_start : l_end] = 1
            attention_mask[c_start : c_end, r_start : r_end] = 1
            # full attention for L-R:
            attention_mask[l_start : l_end, r_start : r_end] = 1
            attention_mask[r_start : r_end, l_start : l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            if return_dict:
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'segment_ids': segment_ids,
                    'img_feats': img_feat,
                    'masked_pos': masked_pos,
                    'masked_ids': masked_ids,
                }
            else:
                return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids)
        if return_dict:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'segment_ids': segment_ids,
                'img_feats': img_feat,
                'masked_pos': masked_pos,
            }
        return input_ids, attention_mask, segment_ids, img_feat, masked_pos

    def prod_tensorize_example(self, text_a, img_feat, text_b=None,
            cls_token_segment_id=0, pad_token_segment_id=0,
            sequence_a_segment_id=0, sequence_b_segment_id=1):
        ''' Tensorize for inference in PROD, batch size is 1, no padding is
            needed
        '''
        tokens = []
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            max_seq_b_len = self.max_seq_len - self.max_seq_a_len - 1
            if len(tokens_b) > max_seq_b_len:
                tokens_b = tokens_b[: max_seq_b_len]
            tokens += tokens_b
        tokens += [self.tokenizer.sep_token]
        od_label_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # image features
        img_len = img_feat.shape[0]
        if img_len > self.max_img_seq_len:
            img_feat = img_feat[0 : self.max_img_seq_len, ]
            img_len = img_feat.shape[0]

        od_label_ids = torch.tensor(od_label_ids, dtype=torch.long)

        return od_label_ids, img_feat


def build_tensorizer(args, tokenizer, is_train=True):
    if hasattr(args, 'mask_od_labels'):
        mask_b = args.mask_od_labels
    else:
        mask_b = False
    if is_train:
        return CaptionTensorizer(
            tokenizer,
            max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length,
            max_seq_a_length=args.max_seq_a_length,
            mask_prob=args.mask_prob,
            max_masked_tokens=args.max_masked_tokens,
            mask_type=args.mask_type,
            is_train=True,
            mask_b=mask_b,
        )
    return CaptionTensorizer(
            tokenizer,
            max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length if args.add_od_labels else args.max_gen_length,
            max_seq_a_length=args.max_gen_length,
            is_train=False
    )

