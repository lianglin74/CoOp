from qd.qd_common import ensure_remove_dir
from qd.tsv_io import TSVDataset
from qd.process_tsv import populate_dataset_details
from qd.process_tsv import build_taxonomy_impl
from qd.process_tsv import get_taxonomy_path
import unittest

class TestProcessTSV(unittest.TestCase):
    def test_build_taxonomy_details_min_image_num(self):
        data = 'TaxPersonDogV1_99999'
        datas = [{'name': 'voc20',
                  'split_infos': [{'split': 'train', 'version': 0}],
                  'use_all': True}]
        ensure_remove_dir('./data/{}'.format(data))
        ensure_remove_dir('./data/{}_with_bb'.format(data))
        ensure_remove_dir('./data/{}_no_bb'.format(data))
        build_taxonomy_impl(
                get_taxonomy_path(data),
                num_test=0,
                data=data,
                datas=datas,
                min_image_per_label=50000,
                max_image_per_label=50000)
        populate_dataset_details(data + '_with_bb')

        dataset = TSVDataset(data + '_with_bb')
        for label, str_count in dataset.iter_data('train',
                'inverted.label.count'):
            count = int(str_count)
            self.assertGreater(count, 50000)

if __name__ == '__main__':
    unittest.main()
