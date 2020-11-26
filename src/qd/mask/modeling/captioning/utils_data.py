import os.path as op
import torch
import logging

from qd.mask.data import samplers
from qd.mask.utils.comm import get_world_size
from qd.mask.data.build import make_data_sampler
from qd.mask.data.datasets import (CaptionTSVDataset,
        PretrainCaptionTSVDataset, CaptionTSVDatasetWithConstraints)
from qd.mask.data.datasets.caption_tensorizer import build_tensorizer


def build_dataset(yaml_file, tokenizer, args, is_train=True, is_pretrain=False):
    if not op.isfile(yaml_file):
        yaml_file = op.join(args.data_dir, yaml_file)
        assert op.isfile(yaml_file)
    tensorizer = build_tensorizer(args, tokenizer, is_train=is_train)
    if is_pretrain:
        if args.qd_format:
            from qd.data_layer.builder import build_vlp_dataset
            return build_vlp_dataset(
                data=args.data,
                split='train',
                label_version=args.train_label_version,
                feature_version=args.train_feature_version,
                add_od_labels=args.add_od_labels,
                od_label_conf=args.od_label_conf,
                label_sort_by_conf=not args.no_sort_by_conf,
                unique_labels_on=args.unique_labels_on,
                qa2caption=args.qa2caption,
                sep_token=tensorizer.tokenizer.sep_token,
                pert_caption_prob=args.pert_caption_prob,
                pert_labels_prob=args.pert_labels_prob,
                tensorizer=tensorizer,
                img_feature_dim=args.img_feature_dim,
                max_img_seq_len=tensorizer.max_img_seq_len,
                feat_sort_by_conf=not args.no_sort_by_conf,
            )
        else:
            return PretrainCaptionTSVDataset(
                yaml_file, 
                tensorizer,
                tokenizer,
                add_od_labels=args.add_od_labels,
                od_label_conf=args.od_label_conf,
                is_train=is_train,
                sort_by_conf=not args.no_sort_by_conf,
                unique_labels_on=args.unique_labels_on,
                pert_caption_prob=args.pert_caption_prob,
                pert_labels_prob=args.pert_labels_prob,
                mask_loss_for_unmatched=args.mask_loss_for_unmatched,
                on_memory=args.on_memory,
                img_feature_dim=args.img_feature_dim,
                qa2caption=args.qa2caption,
            )
    else:
        if args.use_cbs:
            dataset_class = CaptionTSVDatasetWithConstraints
        else:
            dataset_class = CaptionTSVDataset
        return dataset_class(
            yaml_file,
            tensorizer,
            tokenizer=tokenizer,
            add_od_labels=args.add_od_labels,
            od_label_conf = args.od_label_conf if hasattr(args, 'od_label_conf') else 0.0,
            is_train=is_train,
            sort_by_conf=not args.no_sort_by_conf if hasattr(args, 'no_sort_by_conf') else None,
            unique_labels_on=args.unique_labels_on if hasattr(args, 'unique_labels_on') else None,
            on_memory=args.on_memory if hasattr(args, 'on_memory') else False,
            img_feature_dim=args.img_feature_dim,
        )

def make_batch_data_sampler(sampler, images_per_gpu, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_gpu, drop_last=False
    )
    if num_iters is not None and num_iters >= 0:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(args, yaml_file, tokenizer, is_distributed=True, 
        is_train=True, start_iter=0, is_pretrain=False):
    if is_pretrain:
        assert is_train
        dataset = build_dataset(yaml_file, tokenizer, args, is_train, is_pretrain)
    else:
        dataset = build_dataset(yaml_file, tokenizer, args,
            is_train=(is_train and not args.scst), is_pretrain=is_pretrain)
    logger = logging.getLogger(__name__)
    if is_train:
        shuffle = args.train_shuffle
        images_per_gpu = args.per_gpu_train_batch_size
        images_per_batch = images_per_gpu * get_world_size()
        iters_per_batch = len(dataset) // images_per_batch
        num_iters = iters_per_batch * args.num_train_epochs
        logger.info("Train with {} images per GPU.".format(images_per_gpu))
        logger.info("Total batch size {}".format(images_per_batch))
        logger.info("Total training steps {}".format(num_iters))
        logging.info('shuffle = {}'.format(shuffle))
    else:
        shuffle = False
        images_per_gpu = args.per_gpu_eval_batch_size
        num_iters = None
        start_iter = 0

    sampler = make_data_sampler(
        dataset, shuffle, is_distributed, images_per_gpu)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True,
    )
    return data_loader

