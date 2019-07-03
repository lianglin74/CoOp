import unittest


class TestQDCommon(unittest.TestCase):
    def test_parse_test_data_with_version(self):
        pred_data_split_versions = [
                ('model_iter_368408.caffemodel.Tax1300V14.1_OpenImageV4_448Test_with_bb.train.maintainRatio.OutTreePath.TreeThreshold0.1.ClsIndependentNMS.predict',
                    'Tax1300V14.1_OpenImageV4_448Test_with_bb', 'train', 0),
                ('model_iter_0090000.pt.coco2017Full.test.predict.coco_box.report',
                    'coco2017Full', 'test', 0),
                ('model_iter_10000.caffemodel.voc20.maintainRatio.report',
                    'voc20', 'test', 0),
                ('model_iter_2.caffemodel.voc20.test.report',
                    'voc20', 'test', 0),
                ('model_iter_271598.caffemodel.Top100Instagram_with_bb.test.maintainRatio.OutTreePath.TreeThreshold0.1.ClsIndependentNMS.v5.report',
                    'Top100Instagram_with_bb', 'test', 5),
                ('model_iter_0090000.pt.OpenImageV5C.trainval.predict.tsv',
                    'OpenImageV5C', 'trainval', 0),
                ]
        for f, d, s, v in pred_data_split_versions:
            from qd.qd_common import parse_test_data_with_version
            od, os, ov = parse_test_data_with_version(f)
            self.assertEqual(od, d)
            self.assertEqual(s, os)
            self.assertEqual(ov, v)

    def test_run_if_not_cached(self):
        from qd.qd_common import run_if_not_cached

        y = run_if_not_cached(lambda x: x, 2)
        self.assertEqual(y, 2)

        y = run_if_not_cached(lambda x: x, 2)
        self.assertEqual(y, 2)


if __name__ == '__main__':
    unittest.main()
