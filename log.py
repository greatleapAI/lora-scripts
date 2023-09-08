# coding: utf-8

import logging
import sys

logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)
logger.setLevel(logging.DEBUG)


def get_logger():
    return logger
