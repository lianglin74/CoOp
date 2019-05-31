from qd.philly import create_philly_client
import unittest

class TestPhilly(unittest.TestCase):
    def setUp(self):
        from qd.qd_common import init_logging
        init_logging()

    def test_qd_data_exists(self):
        c = create_philly_client()
        self.assertTrue(c.qd_data_exists('voc20/labelmap.txt'))
        self.assertFalse(c.qd_data_exists('voc20/labelmap.dummy.txt'))

    def test_qd_data_upload(self):
        c = create_philly_client()
        c.upload_qd_data('voc20')

if __name__ == '__main__':
    unittest.main()
