from qd.tsv_io import TSVDataset, tsv_reader
from qd.qd_common import ensure_copy_file
from qd_classifier.utils.prep_data import ensure_populate_dataset_crop_index
from qd.pipelines.crop_classification import combine_tag_region_pred
from qd.pipeline import load_pipeline

def predict_on_region(tag_model_expid, region_file, outfile, test_data,
        test_split='test'):
    # create label file for crop and tagging
    regions_dict = {k: v for k, v in tsv_reader(region_file)}
    dataset = TSVDataset(test_data)
    def gen_rows():
        for key, _, _ in dataset.iter_data(test_split):
            yield key, regions_dict.get(key, '[]')

    dataset.update_data(gen_rows(), test_split, t='label',
            generate_info=[['region from:', region_file]])
    ensure_populate_dataset_crop_index(test_data, test_split, version=-1)

    param = {'full_expid': tag_model_expid}
    param.update({
            'test_data': test_data,
            'test_split': test_split,
            'test_version': -1,
            })
    pip = load_pipeline(**param)
    pred_file = pip.ensure_predict()
    combine_tag_region_pred(pred_file, outfile)

