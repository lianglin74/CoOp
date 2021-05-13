import torch
import random
import os.path as op
import numpy as np


class CaptionTensorizer(object):
    def __init__(self, tokenizer, max_img_seq_length=50, max_seq_length=70, 
                 max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3,
                 mask_type='seq2seq', is_train=True, mask_b=False,
                 replace_by_mask_prob=0.8,
                 replace_by_rand_prob=0.1,
                 output_isvalid=False,
                 mask_token_by_word_in_train=False,
                 ):
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
        self.replace_by_mask_prob = replace_by_mask_prob
        self.replace_by_rand_prob = replace_by_rand_prob
        # in train, it is possible to be bidirectional as in CLIP
        #assert mask_type in ('seq2seq', 'seq2seq_off', 'bidirectional', 'cap_s2s', 'cap_bidir')
        self._triangle_mask = torch.tril(torch.ones((self.max_seq_len, 
            self.max_seq_len), dtype=torch.long))

        self._triangle_mask_off = torch.tril(torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.long))
        self._triangle_mask_off[
            list(range(1, self.max_seq_len)),
            list(range(1, self.max_seq_len)),
        ] = 0
        self.output_isvalid = output_isvalid
        self.mask_token_by_word_in_train = mask_token_by_word_in_train

    def tensorize_ab(self, text_a, text_b=None,
                     cls_token_segment_id=0, pad_token_segment_id=0,
                     sequence_a_segment_id=0, sequence_b_segment_id=1,
                     pad_to_max=True,
                     real_text_a_in_test=True,
                     ):
        # this function is used also in clip, where in test mode, we need the
        # real captions as the captions are also input.
        # in captining task, it should be ok either this one or the masked
        # tokens
        #if self.is_train:
        mask_token_by_word = self.is_train and self.mask_token_by_word_in_train
        if not real_text_a_in_test and not self.is_train:
            tokens_a = [self.tokenizer.mask_token] * (self.max_seq_a_len - 2)
        else:
            if mask_token_by_word:
                tokens_a, word_start_idx = self.tokenizer.rich_tokenize(text_a)
            else:
                tokens_a = self.tokenizer.tokenize(text_a)
        #else:
            ## fake tokens to generate masks
        #if not self.is_train:
            ## in captioning, so far, it has to be padded
            #pad_to_max = True
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]
            if mask_token_by_word:
                word_start_idx = [i for i in word_start_idx if i < len(tokens_a)]
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        if mask_token_by_word:
            word_start_idx = [0] + [i + 1 for i in word_start_idx] + [len(tokens) - 1]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        seq_a_len = len(tokens)
        seq_a_padded_len = len(tokens)
        if text_b:
            # pad text_a to keep it in fixed length for better inference.
            if pad_to_max:
                padding_a_len = self.max_seq_a_len - seq_a_len
                tokens += [self.tokenizer.pad_token] * padding_a_len
                segment_ids += ([pad_token_segment_id] * padding_a_len)
                seq_a_padded_len = self.max_seq_a_len
            assert not mask_token_by_word, 'not supported'
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        seq_len = len(tokens)

        if pad_to_max:
            seq_padding_len = self.max_seq_len - seq_len
            tokens = tokens + ([self.tokenizer.pad_token] * seq_padding_len)
            segment_ids += ([pad_token_segment_id] * seq_padding_len)

        origin_input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        origin_input_ids = torch.tensor(origin_input_ids, dtype=torch.long)

        if self.is_train:
            masked_pos = torch.zeros(len(tokens), dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS], [PAD]
            # it is important to mask [SEP] for image captioning as it means [EOS].
            if not mask_token_by_word:
                if self.mask_b:
                    # can mask both text_a and text_b
                    candidate_masked_idx = list(range(1, seq_a_len)) + \
                            list(range(seq_a_padded_len, seq_len))
                    num_masked = min(max(round(self.mask_prob * seq_len), 1), self.max_masked_tokens)
                else:
                    # only mask text_a
                    candidate_masked_idx = list(range(1, seq_a_len))
                    num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
                if self.mask_prob == 0:
                    num_masked = 0
                random.shuffle(candidate_masked_idx)
                num_masked = int(num_masked)
                masked_idx = candidate_masked_idx[:num_masked]
            else:
                assert not self.mask_b, 'b should not be available'
                candidate_masked_idx = list(range(1, len(word_start_idx)))
                num_masked = max(round(self.mask_prob * len(word_start_idx)), 1)
                random.shuffle(candidate_masked_idx)
                masked_word_idx = candidate_masked_idx[:num_masked]
                masked_idx = []
                for i in masked_word_idx:
                    end_idx = seq_a_len if i == len(word_start_idx) - 1 else word_start_idx[i + 1]
                    masked_idx.extend(range(word_start_idx[i], end_idx))
                if len(masked_idx) > self.max_masked_tokens:
                    # the masked_idx here is still the shuffle one, and thus
                    # removing the last one is kind of random. This removal
                    # from the last is not the removal from the right-side of
                    # the description
                    masked_idx = masked_idx[:self.max_masked_tokens]
                num_masked = len(masked_idx)

            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= self.replace_by_mask_prob:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= self.replace_by_rand_prob / (1 - self.replace_by_mask_prob):
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    tokens[pos] = self.tokenizer.get_random_token()
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass
            num_masked = len(masked_idx)
            masked_pos[masked_idx] = 1
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens and pad_to_max:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                        (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(len(tokens), dtype=torch.int)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = seq_a_padded_len, seq_len

        #max_len = self.max_seq_len
        max_len = len(tokens)
        mask_type = self.mask_type
        if mask_type == 'seqbid':
            if random.random() < 0.5:
                mask_type = 'seq2seq'
            else:
                mask_type = 'bidirectional'
        if mask_type == 'bidirectional':
            if self.mask_type == 'seqbid':
                # sometimes it is seq2seq, sometimes it is bid, to make the
                # collate easy, we use a matrix rather than a vector as seq2seq
                # uses a matrix
                attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
                # it does not matter how padding token relies on the valid
                # tokens.
                attention_mask[:, c_start : c_end] = 1 # for text_a
                attention_mask[:, l_start : l_end] = 1 # for text_b if any
            else:
                attention_mask = torch.zeros(max_len, dtype=torch.long)
                attention_mask[c_start : c_end] = 1 # for text_a
                attention_mask[l_start : l_end] = 1 # for text_b if any
        elif self.is_train and mask_type in ('cap_s2s', 'cap_bidir'):
            # caption is a single modality, and without attention on others
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # no attention between [CLS] and caption
            attention_mask[0, 0] = 1
            if mask_type == 'cap_s2s':
                attention_mask[c_start + 1 : c_end, c_start + 1 : c_end].copy_(
                    self._triangle_mask[0 : seq_a_len - 1, 0 : seq_a_len - 1]
                )
            else:
                attention_mask[c_start + 1 : c_end, c_start + 1: c_end] = 1
            attention_mask[l_start : l_end, l_start : l_end] = 1
            # cross attention between [CLS] and L/R
            attention_mask[0, l_start : l_end] = 1
            attention_mask[l_start : l_end, 0] = 1
        else:
            assert mask_type in ('seq2seq', 'seq2seq_off')
            # prepare attention mask:
            # note that there is no attention from caption to image
            # because otherwise it will violate the triangle attention
            # for caption as caption will have full attention on image.
            attention_mask = torch.zeros((max_len, max_len), dtype=torch.long)
            # triangle mask for caption to caption
            if mask_type == 'seq2seq':
                attention_mask[c_start : c_end, c_start : c_end].copy_(
                        self._triangle_mask[0 : seq_a_len, 0 : seq_a_len]
                )
            else:
                attention_mask[c_start : c_end, c_start : c_end].copy_(
                        self._triangle_mask_off[0 : seq_a_len, 0 : seq_a_len]
                )
            # full attention for L-L, R-R
            attention_mask[l_start : l_end, l_start : l_end] = 1
            # full attention for C-L, C-R
            attention_mask[c_start : c_end, l_start : l_end] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        if not pad_to_max:
            assert self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token) == 0

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            info = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'segment_ids': segment_ids,
                'masked_pos': masked_pos,
                'masked_ids': masked_ids,
                'origin_input_ids': origin_input_ids,
            }
            if self.mask_type == 'seqbid':
                info['mask_type'] = mask_type == 'seq2seq'
            return info
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'segment_ids': segment_ids,
            'masked_pos': masked_pos,
        }

    def tensorize_student_example_by_teacher_mask(
        self, text_a, img_feat, text_b=None,
        cls_token_segment_id=0, pad_token_segment_id=0,
        sequence_a_segment_id=0, sequence_b_segment_id=1,
        teacher_masked_ids=None,
        teacher_masked_pos=None,
        teacher_input_ids=None,
        teacher_segment_ids=None
    ):
        # teacher has already done the random mask. We will not do the random
        # mask again. This is only used for training. this is only used for
        # captioning task
        assert self.is_train
        assert teacher_masked_ids is not None
        assert teacher_masked_pos is not None
        assert teacher_segment_ids is not None
        assert not self.mask_b
        assert self.mask_type == 'seq2seq'
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_a_len - 2:
            tokens_a = tokens_a[:(self.max_seq_a_len - 2)]
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        seq_a_len = len(tokens)
        valid = []
        valid.extend([True] * seq_a_len)
        valid.extend([False] * (self.max_seq_a_len - seq_a_len))
        #segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens) - 1)
        #seq_a_padded_len = len(tokens)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - self.max_seq_a_len - 1:
                tokens_b = tokens_b[: (self.max_seq_len - self.max_seq_a_len - 1)]
            tokens_b += [self.tokenizer.sep_token]
            valid.extend([True] * len(tokens_b))
            seq_len = self.max_seq_a_len + len(tokens_b)
            if len(tokens_b) < self.max_seq_len - self.max_seq_a_len:
                padding_b_len = self.max_seq_len - self.max_seq_a_len - len(tokens_b)
                tokens_b += [self.tokenizer.pad_token] * padding_b_len
                valid.extend([False] * padding_b_len)
            input_ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
            input_ids_b = torch.tensor(input_ids_b)
            input_ids = torch.cat((teacher_input_ids[:self.max_seq_a_len], input_ids_b))
        else:
            input_ids = teacher_input_ids
            seq_len = seq_a_len
            valid.extend([False] * (self.max_seq_len - self.max_seq_a_len))
        assert len(input_ids) == self.max_seq_len
        assert len(valid) == self.max_seq_len
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
            valid.extend([True] * self.max_img_seq_len)
        else:
            valid.extend([True] * img_len)
            img_padding_len = self.max_img_seq_len - img_len
            valid.extend([False] * img_padding_len)
            padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
            img_feat = torch.cat((img_feat, padding_matrix), 0)

        max_len = self.max_seq_len + self.max_img_seq_len
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = self.max_seq_a_len, seq_len
        r_start, r_end = self.max_seq_len, self.max_seq_len + img_len

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

        assert len(input_ids) + len(img_feat) == len(attention_mask)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'segment_ids': teacher_segment_ids,
            'img_feats': img_feat,
            'masked_pos': teacher_masked_pos,
            'masked_ids': teacher_masked_ids,
            'seq_a_padded_len': self.max_seq_a_len,
        }
        if self.output_isvalid:
            result['valid'] = torch.tensor(valid)
        return result

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

    def tensorize_example(self, text_a, img_feat, text_b=None,
                          cls_token_segment_id=0, pad_token_segment_id=0,
                          sequence_a_segment_id=0, sequence_b_segment_id=1,
                          return_dict=False,
                          pad_to_max=True,
                          pad_image_to_max=True,
                          img_feats_holder=False,
                          ):
        if not self.is_train:
            # in captioning, so far, it has to be padded
            pad_to_max = True
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
        seq_a_padded_len = len(tokens)
        valid = [True] * seq_a_len
        if text_b:
            if pad_to_max:
                # pad text_a to keep it in fixed length for better inference.
                padding_a_len = self.max_seq_a_len - seq_a_len
                tokens += [self.tokenizer.pad_token] * padding_a_len
                segment_ids += ([pad_token_segment_id] * padding_a_len)
                seq_a_padded_len = self.max_seq_a_len
                valid.extend([False] * padding_a_len)

            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) - 1)]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
            valid.extend([True] * (len(tokens_b) + 1))

        seq_len = len(tokens)
        padded_len = seq_len
        if pad_to_max:
            # pad on the right for image captioning
            seq_padding_len = self.max_seq_len - seq_len
            tokens = tokens + ([self.tokenizer.pad_token] * seq_padding_len)
            segment_ids += ([pad_token_segment_id] * seq_padding_len)
            padded_len = len(tokens)
            valid.extend([False] * seq_padding_len)

        if self.is_train:
            masked_pos = torch.zeros(len(tokens), dtype=torch.int)
            # randomly mask words for prediction, ignore [CLS], [PAD]
            # it is important to mask [SEP] for image captioning as it means [EOS].
            if self.mask_b:
                # can mask both text_a and text_b
                candidate_masked_idx = list(range(1, seq_a_len)) + \
                        list(range(seq_a_padded_len, seq_len))
                num_masked = min(max(round(self.mask_prob * seq_len), 1), self.max_masked_tokens)
            else:
                # only mask text_a
                candidate_masked_idx = list(range(1, seq_a_len))
                num_masked = min(max(round(self.mask_prob * seq_a_len), 1), self.max_masked_tokens)
            if self.mask_prob == 0:
                num_masked = 0
            random.shuffle(candidate_masked_idx)
            num_masked = int(num_masked)
            masked_idx = candidate_masked_idx[:num_masked]
            masked_idx = sorted(masked_idx)
            masked_token = [tokens[i] for i in masked_idx]
            for pos in masked_idx:
                if random.random() <= self.replace_by_mask_prob:
                    # 80% chance to be a ['MASK'] token
                    tokens[pos] = self.tokenizer.mask_token
                elif random.random() <= self.replace_by_rand_prob / (1 - self.replace_by_mask_prob):
                    # 10% chance to be a random word ((1-0.8)*0.5)
                    tokens[pos] = self.tokenizer.get_random_token()
                else:
                    # 10% chance to remain the same (1-0.8-0.1)
                    pass

            assert num_masked == len(masked_idx)
            masked_pos[masked_idx] = 1 
            # pad masked tokens to the same length
            if num_masked < self.max_masked_tokens and pad_to_max:
                masked_token = masked_token + ([self.tokenizer.pad_token] *
                        (self.max_masked_tokens - num_masked))
            masked_ids = self.tokenizer.convert_tokens_to_ids(masked_token)
        else:
            masked_pos = torch.ones(len(tokens), dtype=torch.int)

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
            valid.extend([True] * self.max_img_seq_len)
        else:
            if pad_to_max or pad_image_to_max:
                valid.extend([True] * img_len)
                img_padding_len = self.max_img_seq_len - img_len
                padding_matrix = torch.zeros((img_padding_len, img_feat.shape[1]))
                img_feat = torch.cat((img_feat, padding_matrix), 0)
                valid.extend([False] * img_padding_len)
        padded_img_len = len(img_feat)

        max_len = padded_len + padded_img_len
        assert max_len == len(valid)
        # C: caption, L: label, R: image region
        c_start, c_end = 0, seq_a_len
        l_start, l_end = seq_a_padded_len, seq_len
        r_start, r_end = padded_len, padded_len + img_len
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

        assert len(input_ids) + len(img_feat) == len(attention_mask)

        if self.is_train:
            masked_ids = torch.tensor(masked_ids, dtype=torch.long)
            if return_dict:
                result = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'segment_ids': segment_ids,
                    'img_feats': img_feat,
                    'masked_pos': masked_pos,
                    'masked_ids': masked_ids,
                    'seq_a_padded_len': seq_a_padded_len,
                }
                if self.output_isvalid:
                    result['valid'] = torch.tensor(valid)
                return result
            else:
                return (input_ids, attention_mask, segment_ids, img_feat, masked_pos, masked_ids)
        if return_dict:
            assert not self.output_isvalid, 'not implemented'
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'segment_ids': segment_ids,
                'img_feats': img_feat,
                'masked_pos': masked_pos,
            }
        return input_ids, attention_mask, segment_ids, img_feat, masked_pos

def build_tensorizer(args, tokenizer, is_train=True):
    if hasattr(args, 'mask_od_labels'):
        mask_b = args.mask_od_labels
    else:
        mask_b = False
    if is_train:
        extra_keys = ['replace_by_mask_prob', 'replace_by_rand_prob']
        extra_param = {}
        for k in extra_keys:
            if hasattr(args, k):
                extra_param[k] = getattr(args, k)
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
            **extra_param,
        )
    return CaptionTensorizer(
            tokenizer,
            max_img_seq_length=args.max_img_seq_length,
            max_seq_length=args.max_seq_length if args.add_od_labels else args.max_gen_length,
            max_seq_a_length=args.max_gen_length,
            is_train=False
    )

