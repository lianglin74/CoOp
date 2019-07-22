import unittest
import logging
from pprint import pformat


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
                ('model_iter_0090000.pt.coco2017Full.test.testInputSize640.predict.coco_box.report',
                    'coco2017Full', 'test', 0),
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

    def test_extra_param_decode(self):
        from qd.qd_common import decode_general_cmd
        param = 'abc'
        import base64
        base64_param = base64.b64encode(param.encode()).decode()

        p = 'python src/abc.py -bp {}'.format(base64_param)
        result = decode_general_cmd(p)

        self.assertEqual(result, param)

    def test_attach_log_parsing_result_philly(self):
        log = '2019-05-26T22:36:42.365Z: [1,0]<stdout>:2019-05-26 22:36:42,364.364 trainer.py:116   do_train(): eta: 5 days, 2:12:43  iter: 277800  speed: 63.3 images/sec  loss: 0.4990 (0.5110)  loss_box_reg: 0.1158 (0.1131)  loss_classifier: 0.2901 (0.3015)  loss_objectness: 0.0528 (0.0543)  loss_rpn_box_reg: 0.0336 (0.0420)  time: 0.2521 (0.2739)  data: 0.0035 (0.0119)  lr: 0.020000  max mem: 2107'
        info = {'latest_log': log}
        from qd.qd_common import attach_log_parsing_result
        attach_log_parsing_result(info)
        logging.info(pformat(info))
        self.assertTrue(info['left'].startswith('5-2.2h'))
        self.assertTrue(info['speed'], 63.3)

    def test_attach_log_parsing_result_aml(self):
        log = '2019-07-05 05:29:15,703.703 trainer.py:138   do_train(): eta: 12:03:24  iter: 2600  speed: 34.0 images/sec  loss: 0.7286 (0.7346)  loss_box_reg: 0.1492 (0.1147)  loss_classifier: 0.4042 (0.3980)  loss_objectness: 0.1057 (0.1466)  loss_rpn_box_reg: 0.0659 (0.0753)  time: 0.4061 (0.4966)  data: 0.0081 (0.0944)  lr: 0.050000  max mem: 6684'
        info = {'latest_log': log}
        from qd.qd_common import attach_log_parsing_result
        attach_log_parsing_result(info)
        logging.info(pformat(info))
        import datetime
        self.assertEqual(info['log_time'], datetime.datetime(2019, 7, 5, 5, 29, 15))
        self.assertEqual(info['speed'], 63.3)

    def test_attach_gpu_utility_from_log(self):
        log = r"07-05 22:32:53.330 67bc8cd0f14741d2b64331f9356eb105000001 905 aml_server.py:138    monitor(): [{'mem_used': 4178, 'mem_total': 16130, 'gpu_util': 97}, {'mem_used': 3820, 'mem_total': 16130, 'gpu_util': 91}, {'mem_used': 3806, 'mem_total': 16130, 'gpu_util': 94}, {'mem_used': 3784, 'mem_total': 16130, 'gpu_util': 90}]"
        from qd.qd_common import attach_gpu_utility_from_log
        info = {'latest_log': log}
        attach_gpu_utility_from_log([log], info)
        logging.info(info)
        self.assertEqual(info['mem_used'], '3.7-4.1')
        self.assertEqual(info['gpu_util'], '90-97')

    def test_dict_remove_path(self):
        from qd.qd_common import dict_remove_path
        d = {'a': {'b': {'c': 'd'}}}
        p = 'a$b$c'
        dict_remove_path(d, p)

        self.assertEqual(len(d), 0)

    def test_dict_remove_path2(self):
        from qd.qd_common import dict_remove_path
        d = {'a': {'b': {'c': 'd',
                         'e': 'f'}}}
        p = 'a$b$c'
        dict_remove_path(d, p)

        self.assertEqual(len(d), 1)
        self.assertEqual(len(d['a']), 1)
        self.assertEqual(d['a']['b']['e'], 'f')

    def test_dict_remove_path3(self):
        from qd.qd_common import dict_remove_path
        d = {'a': {'b': {'c': 'd'},
                   'e': 'f'}}
        p = 'a$b$c'
        dict_remove_path(d, p)

        self.assertEqual(len(d), 1)
        self.assertEqual(len(d['a']), 1)
        self.assertEqual(d['a']['e'], 'f')

    def test_dict_ensure_path_key_converted(self):
        from qd.qd_common import dict_ensure_path_key_converted
        x = {'a$b': 'c'}
        dict_ensure_path_key_converted(x)
        self.assertDictEqual(x, {'a': {'b': 'c'}})

if __name__ == '__main__':
    unittest.main()
