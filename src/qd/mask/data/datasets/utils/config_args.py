import os.path as op
import pdb

def config_tsv_dataset_args(cfg, dataset_file, factory_name=None, is_train=True):
    full_yaml_file = op.join(cfg.DATA_DIR, dataset_file)

    assert op.isfile(full_yaml_file)

    if cfg.DATASETS.BOX_EXTRA_FIELDS:
        extra_fields = cfg.DATASETS.BOX_EXTRA_FIELDS
    else:
        extra_fields = ["class"]
        if cfg.MODEL.MASK_ON:
            extra_fields.append("mask")
        if cfg.MODEL.RPN.USE_SAMPLE_CONFIDENCE or \
               cfg.MODEL.ROI_BOX_HEAD.USE_SAMPLE_CONFIDENCE:
            extra_fields.append("confidence")
        if cfg.MODEL.ATTRIBUTE_ON:
            extra_fields.append("attributes_encode")

    skip_performance_eval = False if is_train else cfg.TEST.SKIP_PERFORMANCE_EVAL
    if cfg.MODEL.RPN.FORCE_BOXES or cfg.MODEL.ROI_BOX_HEAD.FORCE_BOXES:
        is_load_label = True
    else:
        is_load_label = not skip_performance_eval

    args = dict(
        yaml_file=full_yaml_file,
        extra_fields=extra_fields,
        is_load_label=is_load_label,
        mask_mode=cfg.MODEL.ROI_MASK_HEAD.MASK_MODE,
        relation_on=cfg.MODEL.RELATION_ON,
        cv2_output=cfg.INPUT.CV2_OUTPUT
    )

    if factory_name is not None:
        tsv_dataset_name = factory_name
        # adding this check to support all old configs
        # that rely on this dataset name
        if tsv_dataset_name == "COCOCaptionsTSVDataset":
            tsv_dataset_name = "ImageCaptionsTSVDataset"
    else:
        if "openimages" in dataset_file:
            tsv_dataset_name = "OpenImagesTSVDataset"
        elif "visualgenome" in dataset_file or "gqa" in dataset_file:
            tsv_dataset_name = "VGTSVDataset"
        else:
            tsv_dataset_name = "ODTSVDataset"

    if tsv_dataset_name == "ImageCaptionsTSVDataset":
        args['features_as_input'] = cfg.MODEL.SCAN.FEATURES_AS_INPUT
        args['bbox_as_input'] = cfg.MODEL.SCAN.BBOX_AS_INPUT
        args['text_features_as_input'] = cfg.MODEL.SCAN.TEXT_FEATURES_AS_INPUT
        args['num_captions_per_image'] = cfg.MODEL.SCAN.NUM_CAPTIONS_PER_IMG
    elif tsv_dataset_name in ("TSVBoxDataset", "TSVBoxYamlDataset"):
        args['add_box_channel'] = cfg.MODEL.CLASSIFIER.ADD_BOX_CHANNEL
        args['pad_ratio'] = cfg.MODEL.CLASSIFIER.PAD_RATIO
        args['has_negative_label'] = cfg.MODEL.CLASSIFIER.HAS_NEGATIVE_LABEL
    elif tsv_dataset_name in ["VQADataset", "VQAImgDataset"]:
        args['features_as_input'] = (cfg.bbox_as_input==0)
        args['bbox_as_input'] = (cfg.bbox_as_input==1)
        args['label_data_dir'] = cfg.label_data_dir
        args['max_img_seq_length'] = cfg.max_img_seq_length
        args['num_qas_per_image'] = cfg.num_qas_per_image
        args['args'] = cfg
    elif tsv_dataset_name in ["RetrievalDataset"]:
        args['args'] = cfg
    elif tsv_dataset_name in ["OscarTSVDataset", "OscarGroundedTSVDataset", "OscarImgTSVDataset"]:
        args['args'] = cfg
        args['seq_len'] = cfg.max_seq_length
        args['on_memory'] = cfg.on_memory
        args['num_qas_per_image'] = cfg.num_qas_per_image

    return args, tsv_dataset_name
