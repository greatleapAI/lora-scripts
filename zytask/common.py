# coding:utf-8

import os
import errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    # Python >2.5 (except OSError, exc: for Python <2.5)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
