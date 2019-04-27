from qd.qd_common import init_logging
import logging
from pprint import pformat


def test_uhrs_verify_db_merge_to_tsv():
    from qd.db import uhrs_verify_db_merge_to_tsv
    uhrs_verify_db_merge_to_tsv()

def test_create_new_image_tsv_if_exif_rotated():
    from qd.process_tsv import create_new_image_tsv_if_exif_rotated
    create_new_image_tsv_if_exif_rotated('voc20', 'train')

if __name__ == '__main__':
    from qd.qd_common import parse_general_args
    init_logging()
    kwargs = parse_general_args()
    logging.info('param:\n{}'.format(pformat(kwargs)))
    function_name = kwargs['type']
    del kwargs['type']
    locals()[function_name](**kwargs)
