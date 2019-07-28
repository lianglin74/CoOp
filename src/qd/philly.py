import logging
logging.warn('use import qd.gpucluster.philly_client')
from qd.gpucluster.philly_client import *

if __name__ == '__main__':
    os.environ['LD_LIBRARY_PATH'] = '/opt/intel/mkl/lib/intel64'
    init_logging()
    args = parse_args()
    param = vars(args)
    execute(**param)
