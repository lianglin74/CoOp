import torch
from torchvision.transforms import transforms


def collate_fn(batch):
    # this function is designed to support any customized type and to be compatible
    # with the default collate function
    ele = batch[0]
    from qd.torch_common import SparseTensor, IgnoreLastDimSparseTensor
    from torch.utils.data.dataloader import default_collate
    if isinstance(ele, dict):
        return {key: collate_fn([d[key] for d in batch]) for key in ele}
    elif isinstance(ele, SparseTensor):
        return SparseTensor.stack(batch)
    elif isinstance(ele, IgnoreLastDimSparseTensor):
        return IgnoreLastDimSparseTensor.stack(batch)
    else:
        return default_collate(batch)

def build_vlp_dataset(data,
                      split,
                      label_version,
                      feature_version,
                      add_od_labels,
                      od_label_conf,
                      label_sort_by_conf,
                      unique_labels_on,
                      qa2caption,
                      sep_token,
                      pert_caption_prob,
                      pert_labels_prob,
                      tensorizer,
                      img_feature_dim,
                      max_img_seq_len,
                      feat_sort_by_conf,
                      label_type=None, # or BCE
                      img_feat_label_type=None,
                      region_loss_for_unmatched=True,
                      ):
    from .dataset import CaptionIdxTSVDataset, DatasetPlusTransform
    dataset = CaptionIdxTSVDataset(
        data=data,
        split=split,
        caption_version=None,
    )
    from .transform import (
        LoadLabel,
        LoadHW,
        LoadFeature,
        LoadCaption,
        IdentifyTextAB,
        RandomPairNegative,
        CaptionTensorizer,
        TokenizeTransform,
        NoChange,
        PrepareLabel,
        RemoveUselessKeys,
        RenameKey,
    )
    hw_loader = LoadHW(
        data=data, split=split,
    )
    feature_loader = LoadFeature(
        data=data, split=split,
        version=feature_version,
        img_feature_dim=img_feature_dim,
        max_len=max_img_seq_len,
        sort_by_conf=feat_sort_by_conf,
    )
    caption_loader = LoadCaption(
        data=data, split=split, version=None,
    )
    text_ab = IdentifyTextAB(
        add_od_labels, od_label_conf, label_sort_by_conf,
        unique_labels_on, qa2caption, sep_token,
    )
    if add_od_labels:
        label_loader = LoadLabel(
            data=data, split=split,
            version=label_version)
        load_negative_transform = [
            label_loader,
            caption_loader,
            text_ab,
        ]
    else:
        load_negative_transform = [
            caption_loader,
            text_ab,
        ]
    neg_pair = RandomPairNegative(
        pert_caption_prob, pert_labels_prob,
        transforms.Compose(load_negative_transform),
    )
    trans_tensorizer = CaptionTensorizer(tensorizer)

    token_img_label = (
        NoChange() if img_feat_label_type is None else
        TokenizeTransform(tensorizer.tokenizer, ['feats_class'])
    )

    prepare_label = PrepareLabel(
        label_type, img_feat_label_type,
        region_loss_for_unmatched,
    )
    all_trans = []
    if add_od_labels:
        all_trans.append(label_loader)

    all_trans.extend([
        caption_loader,
        text_ab,
        hw_loader,
        feature_loader,
        neg_pair,
        trans_tensorizer,
        token_img_label,
        prepare_label,
        RemoveUselessKeys([
            'idx',
            'idx_img',
            'idx_cap',
            'key',
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
            'masked_pos',
            'masked_ids',
            'max_seq_a_len',
            'feats_conf',
            'feats_class',
            'vocab_size',
            'feats_class_token_ids',
            'feats_class_tokens',
        ]),
        RenameKey({'segment_ids': 'token_type_ids'}),
    ])
    all_trans = transforms.Compose(all_trans)
    dataset = DatasetPlusTransform(dataset, all_trans)
    return dataset

