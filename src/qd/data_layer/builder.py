from torchvision.transforms import transforms
from .dataset import (
    CaptionIdxTSVDataset,
    ImageIdxTSVDataset,
    DatasetPlusTransform,
)
from .transform import (
    LoadLabel,
    LoadHW,
    LoadFeature,
    LoadImage,
    LoadCaption,
    IdentifyTextAB,
    RandomPairNegative,
    CaptionTensorizer,
    TokenizeTransform,
    NoChange,
    PrepareLabel,
    RemoveUselessKeys,
    RenameKey,
    AppendDummyFeature,
    LogSystemInfo,
)

def collate_fn(batch):
    # this function is designed to support any customized type and to be compatible
    # with the default collate function
    ele = batch[0]
    from qd.torch_common import SparseTensor, IgnoreLastDimSparseTensor
    from qd.torch_common import BoxList
    from torch.utils.data.dataloader import default_collate
    import torch
    if isinstance(ele, dict):
        return {key: collate_fn([d[key] for d in batch]) for key in ele}
    elif isinstance(ele, SparseTensor):
        return SparseTensor.stack(batch)
    elif isinstance(ele, IgnoreLastDimSparseTensor):
        return IgnoreLastDimSparseTensor.stack(batch)
    elif isinstance(ele, BoxList):
        return batch
    else:
        if all(isinstance(b, torch.Tensor) for b in batch) and len(batch) > 0:
            if not all(b.shape == batch[0].shape for b in batch[1:]):
                assert all(len(b.shape) == len(batch[0].shape) for b in batch[1:])
                shape = torch.tensor([b.shape for b in batch])
                max_shape = tuple(shape.max(dim=0)[0].tolist())
                batch2 = []
                for b in batch:
                    if any(c < m for c, m in zip(b.shape, max_shape)):
                        b2 = torch.zeros(max_shape, dtype=b.dtype, device=b.device)
                        if b.dim() == 1:
                            b2[:b.shape[0]] = b
                        elif b.dim() == 2:
                            b2[:b.shape[0], :b.shape[1]] = b
                        elif b.dim() == 3:
                            b2[:b.shape[0], :b.shape[1], :b.shape[2]] = b
                        else:
                            raise NotImplementedError
                        b = b2
                    batch2.append(b)
                batch = batch2
        return default_collate(batch)

def build_caption_dataset(
    data, split, is_train, on_memory,
    feature_version,
    img_feature_dim,
    max_img_seq_len,
    feat_sort_by_conf,
    add_od_labels,
    label_version,
    od_label_conf,
    label_sort_by_conf,
    unique_labels_on,
    sep_token,
    tensorizer,
    qa2caption=None,
    # the following parameters are only used in the case of e2e where feature
    # is not pre-calculated
    input_crop_size=224,
    test_resize_size=None,
    input_small_scale=None,
    pad_to_max=True,
):
    if is_train:
            #'idx': idx,
            #'idx_img': idx_img,
            #'idx_cap': idx_cap,
            #'key': key,
            #'dataset': self,
        dataset = CaptionIdxTSVDataset(
            data=data,
            split=split,
            caption_version=None,
        )
    else:
        dataset = ImageIdxTSVDataset(
            data=data,
            split=split,
        )
    all_trans = []
    cache_policy = 'memory' if on_memory else None
    hw_loader = LoadHW(
        data=data, split=split,
        cache_policy=cache_policy,
    )
    all_trans.append(hw_loader)

    load_feature = max_img_seq_len > 0
    if load_feature:
        feature_loader = LoadFeature(
            data=data,
            split=split,
            version=feature_version,
            img_feature_dim=img_feature_dim,
            max_len=max_img_seq_len,
            sort_by_conf=feat_sort_by_conf,
        )
    else:
        # load image and we will extract the features online. This is mainly
        # used for end-to-end training or inference.
        image_loader = LoadImage(data, split)
        if is_train:
            from qd.data_layer.transform import get_inception_train_transform
            image_transform = get_inception_train_transform(
                bgr2rgb=True,
                crop_size=input_crop_size,
                small_scale=input_small_scale,
            )
        else:
            if test_resize_size is None:
                resize_size = 256 * input_crop_size // 224
            else:
                resize_size = test_resize_size
            from qd.data_layer.transform import get_inception_test_transform
            image_transform = get_inception_test_transform(
                bgr2rgb=True,
                resize_size=resize_size,
                crop_size=input_crop_size,
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
            data=data, split=split, version=None,
            cache_policy=cache_policy,
        )
        all_trans.append(caption_loader)

    if add_od_labels:
        label_loader = LoadLabel(
            data=data, split=split,
            version=label_version)
        all_trans.append(label_loader)

    text_ab = IdentifyTextAB(
        add_od_labels, od_label_conf, label_sort_by_conf,
        unique_labels_on, qa2caption, sep_token,
    )
    all_trans.append(text_ab)

    trans_tensorizer = CaptionTensorizer(
        tensorizer,
        with_img_feats=load_feature,
        pad_to_max=pad_to_max,
        real_text_a_in_test=False,
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
            'vocab_size',
            'feats_class_token_ids',
            'feats_class_tokens',
    ]
    all_trans.extend([
        RemoveUselessKeys(useless_keys),
        RenameKey({'segment_ids': 'token_type_ids'}),
    ])
    all_trans = transforms.Compose(all_trans)
    dataset = DatasetPlusTransform(dataset, all_trans)
    return dataset

# used in mmask_pretrain pipeline
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
                      on_memory=False,
                      caption_version=None
                      ):
    dataset = CaptionIdxTSVDataset(
        data=data,
        split=split,
        caption_version=caption_version,
    )
    cache_policy = 'memory' if on_memory else None
    hw_loader = LoadHW(
        data=data, split=split,
        cache_policy=cache_policy,
    )
    load_feature = max_img_seq_len > 0
    if load_feature:
        feature_loader = LoadFeature(
            data=data, split=split,
            version=feature_version,
            img_feature_dim=img_feature_dim,
            max_len=max_img_seq_len,
            sort_by_conf=feat_sort_by_conf,
        )
    else:
        # load image and we will extract the features online. This is mainly
        # used for end-to-end training or inference.
        image_loader = LoadImage(data, split)
        from qd.data_layer.transform import get_inception_transform
        image_transform = get_inception_transform(bgr2rgb=True, crop_size=224)
        from qd.data_layer.transform import ImageTransform2Dict
        image_transform = ImageTransform2Dict(image_transform)
        feature_loader = transforms.Compose([
            image_loader,
            image_transform,
        ])
    caption_loader = LoadCaption(
        data=data, split=split, version=caption_version,
        cache_policy=cache_policy,
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
    trans_tensorizer = CaptionTensorizer(
        tensorizer,
        with_img_feats=load_feature,
    )

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
            'seq_a_padded_len',
            'feats_conf',
            'feats_class',
            'vocab_size',
            'feats_class_token_ids',
            'feats_class_tokens',
        ]),
        RenameKey({'segment_ids': 'token_type_ids'}),
        LogSystemInfo(),
    ])
    all_trans = transforms.Compose(all_trans)
    dataset = DatasetPlusTransform(dataset, all_trans)
    return dataset

# in pythia-m4c repo, we use this function to replace the original non-tsv
# format
def build_tap_imdb(
    data,
    split,
    caption_version,
    feature_version,
    ocr_feature_version,
):
    from qd.data_layer.dataset import CaptionIdxTSVDataset
    dataset = CaptionIdxTSVDataset(data, split, caption_version)
    ts = []

    # hw
    hw_loader = LoadHW(data=data, split=split,)
    ts.append(hw_loader)

    feature_loader = LoadFeature(
        data=data,
        split=split,
        version=feature_version,
        img_feature_dim=2048,
        max_len=100,
        sort_by_conf=False,
        append_box=False,
        attach_box=True,
    )
    ts.append(feature_loader)
    appender = AppendDummyFeature(append_to=100)
    ts.append(appender)
    rn = RenameKey({
        'img_feats': 'image_feature_0',
        'img_feats_bbox': 'image_info_0$bbox',
        'feats_conf': 'image_info_0$conf',
        'feats_class': 'image_info_0$object_tokens',
        'img_feats_valid': 'image_info_0$max_features',
    })
    ts.append(rn)

    feature_loader = LoadFeature(
        data=data,
        split=split,
        version=ocr_feature_version,
        img_feature_dim=2048,
        max_len=100,
        sort_by_conf=False,
        append_box=False,
        attach_box=True,
    )
    ts.append(feature_loader)
    appender = AppendDummyFeature(append_to=100)
    ts.append(appender)
    rn = RenameKey({
        'img_feats': 'image_feature_1',
        'img_feats_bbox': 'image_info_1$ocr_boxes',
        'feats_conf': 'image_info_1$ocr_conf',
        'feats_class': 'image_info_1$ocr_tokens',
    })
    ts.append(rn)

    caption_loader = LoadCaption(data=data, split=split, version=caption_version,)
    ts.append(caption_loader)
    rn = RenameKey({
        'caption$question': 'question',
        'caption$question_id': 'question_id',
        'caption$answers': 'answers',
        # in m4c_textcaps/dataset.py, caption_str will be renamed as answers
        # for prediction in fine-tuning
        'caption$caption': 'caption_str',
        'caption$caption_id': 'caption_id',
        #'img_feats_normalized_box': 'image_height',
    })
    ts.append(rn)

    rn = RenameKey({
        'width': 'image_width',
        'height': 'image_height',
        'key': 'image_id',
    })
    ts.append(rn)

    rm = RemoveUselessKeys([
        'image_info_1$ocr_conf',
    ])
    ts.append(rm)

    all_trans = transforms.Compose(ts)
    dataset = DatasetPlusTransform(dataset, all_trans)

    return dataset

