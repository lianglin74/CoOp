from __future__ import print_function
import os
import logging
import time

from qd.qd_common import ensure_directory

class Logger(object):
    def __init__(self, root_output_path, logger_name):
        # set up logger
        ensure_directory(root_output_path)
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)

        log_file = '{}_{}.log'.format(logger_name, time.strftime('%Y-%m-%d-%H-%M'))
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=os.path.join(root_output_path, log_file), format=head)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.rank = 0

    def info(self, msg):
        print(msg)
        if self.rank == 0:
            self.logger.info(msg)

class DistributedLogger(Logger):
    def __init__(self, root_output_path, logger_name, rank):
        ensure_directory(root_output_path)
        try:
            for handler in logging.root.handlers:
                logging.root.removeHandler(handler)
        except:
            pass

        log_file = '{}.log'.format(logger_name)
        head = '%(asctime)-15s %(message)s'
        logging.basicConfig(filename=os.path.join(root_output_path, log_file), format=head)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.rank = rank
