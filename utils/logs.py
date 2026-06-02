#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from datetime import datetime
from functools import partial
from utils.utils import get_project_root
from loguru import logger as _logger

def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """Adjust the log level to above level"""
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d")
    log_name = f"{name}_{formatted_date}" if name else formatted_date  # name a log with prefix name

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    logs_path = os.path.join(get_project_root(), f"logs/{log_name}.txt")
    _logger.add(logs_path, level=logfile_level)
    return _logger

logger = define_log_level()

logger.info("Logging initialized")

def log_llm_stream(msg):
    _llm_stream_log(msg)


def set_llm_stream_logfunc(func):
    global _llm_stream_log
    _llm_stream_log = func


_llm_stream_log = partial(print, end="")
