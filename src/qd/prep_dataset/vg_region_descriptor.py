from qd.tsv_io import (TSVDataset, csv_reader, is_verified_rect, tsv_reader,
                       tsv_writer, tsv_writers)
from qd.qd_common import (calc_mean, calculate_image_ap2, calculate_iou,
                          check_best_iou, cmd_run, concat_files, copy_file,
                          dict_ensure_path_key_converted, dict_get_path_value,
                          dict_has_path, dict_set_path_if_not_exist,
                          dict_to_list, dict_update_nested_dict,
                          dict_update_path_value, encode_expid,
                          encoded_from_img, ensure_copy_file, ensure_directory,
                          ensure_remove_dir, find_float_tolorance_unequal,
                          float_tolorance_equal, get_current_time_as_str,
                          get_mpi_rank, get_mpi_size, init_logging,
                          iter_swap_param, json_dump, list_to_dict,
                          load_from_yaml_file, load_list_file, parse_yolo_log,
                          parse_yolo_log_acc, parse_yolo_log_st,
                          print_offensive_folder, print_table, read_to_buffer,
                          remove_dir, run_if_not_cached, set_if_not_exist,
                          softnms, softnms_c, try_delete, url_to_str,
                          write_to_file, write_to_yaml_file, zip_qd)
import base64
from qd.process_image import (draw_bb, draw_rects, load_image,
                              network_input_to_image, put_text, save_image,
                              show_image, show_images, show_net_input)
from qd.qd_common import qd_tqdm as tqdm
from qd.process_tsv import generate_key_idximage_idxcaption, generate_key_idximage_idxcaption_from_num
import os.path as op
from qd.qd_common import read_to_buffer
import json
from qd.process_tsv import (CogAPI, TSVFile, build_tax_dataset_from_db,
                            build_taxonomy_from_single_source,
                            build_taxonomy_impl, concat_tsv_files,
                            convert_one_label, ensure_inject_dataset,
                            ensure_inject_decorate, ensure_inject_expid_pred,
                            ensure_inject_gt, ensure_inject_image,
                            ensure_inject_pred, ensure_upload_image_to_blob,
                            find_best_matched_rect_idx,
                            find_same_location_rects, find_same_rects,
                            get_data_sources, get_taxonomy_path, hash_sha1,
                            img_from_base64, inc_one_dic_dic, load_key_rects,
                            load_labels, normalize_to_str,
                            parallel_multi_tsv_process, parallel_tsv_process,
                            parse_combine, populate_dataset_details,
                            populate_dataset_hw, rect_in_rects,
                            regularize_data_sources, softnms_row_process,
                            update_confusion_matrix, visualize_tsv2)


class PrepareVGRegionDescriptionDataset():
    def __init__(self):
        self.data = 'VisualGenomeRegionCaption'

    def debug(self):
        f = op.join('data', 'VisualGenomeRegionCaption', 'raw', 'region_descriptions.json')
        info = read_to_buffer(f)
        info = json.loads(info)
        #ipdb> pp len(info)
        #108077

    def populate_dataset_hw(self):
        populate_dataset_details('VisualGenomeRegionCaption')
        populate_dataset_hw('VisualGenomeRegionCaption')

    def generate_lineidx(self):
        generate_key_idximage_idxcaption(self.data, 'train', version=None)

    def generate_tsv(self):
        f = op.join('data', 'VisualGenomeRegionCaption', 'raw', 'region_descriptions.json')
        info = read_to_buffer(f)
        info = json.loads(info)
        data_root = op.join('data', 'VisualGenomeRegionCaption')
        image_folder = op.join(
            'data', 'VisualGenomeRegionCaption', 'raw', 'VG_100K')
        #debug = True
        debug = False
        def gen_rows():
            for i in tqdm(info):
                img_f = op.join(image_folder, str(i['id']) + '.jpg')
                im = load_image(img_f)
                assert im.shape[0] > 0 and im.shape[1] > 0

                img_row = (i['id'], base64.b64encode(read_to_buffer(img_f)))
                label_row = (i['id'], json.dumps([]))

                for r in i['regions']:
                    r['caption'] = r['phrase']
                    del r['phrase']
                    r['rect'] = [r['x'], r['y'], r['x'] + r['width'], r['y'] +
                                 r['height']]
                    if debug:
                        curr_im = im.copy()
                        r['class'] = ''
                        draw_rects([r], curr_im)
                        save_image(curr_im, op.join(data_root, 'debug', r['caption'] +'.jpg'))
                caps = i['regions']
                caption_row = (i['id'], json_dump(caps))
                yield img_row, label_row, caption_row
        dataset = TSVDataset('VisualGenomeRegionCaption')
        img_f = dataset.get_data('train')
        label_f = dataset.get_data('train', 'label')
        caption_f = dataset.get_data('train', 'caption')
        tsv_writers(gen_rows(),
                    (img_f, label_f, caption_f))

    def download_raw_image():
        #https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
        #https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
        pass

    def downoad_json_meta():
        # https://visualgenome.org/static/data/dataset/region_descriptions.json.zip
        pass

