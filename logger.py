import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from numbers import Number
from typing import Dict, Optional, TextIO

import torch

logger = logging.getLogger(__name__)


def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.today().strftime(fmt)


def format_stat(stat):
    if isinstance(stat, Number):
        stat = "{:g}".format(stat)
    elif torch.is_tensor(stat):
        stat = stat.tolist()
    elif isinstance(stat, str):
        pass
    else:
        raise TypeError("stat only support number, str and tensor")
    return stat


def _format_stats(stats, epoch=None, step=None):
    poststats = OrderedDict()
    if epoch is not None:
        poststats['epoch'] = epoch
    if step is not None:
        poststats['step'] = step
    for key in stats.keys():
        poststats[key] = format_stat(stats[key])
    return poststats


class BaseLogger(object):
    def __init__(
        self,
        logger_name: str,
        log_dir: Optional[str] = None,
        stream: Optional[TextIO] = None,
        log_format: Optional[str] = None,
        log_level: Optional[int] = None,
        args=None
    ):
        super(BaseLogger, self).__init__()

        assert (
            log_dir is not None or stream is not None
        ), "at least one of log_dir and stream is not None"

        self.logger_name = logger_name
        self.log_dir = log_dir
        self.log_level = log_level
        self.stream = stream

        if log_format is None:
            self.log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        if log_level is None:
            self.log_level = logging.INFO

        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self._init_logger()
        if args is not None:
            self.logger.info(json.dumps(vars(args), indent=4))

    def _init_logger(self):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.log_level)
        
        if self.log_dir is not None:
            file_handler = logging.FileHandler(
                os.path.join(self.log_dir, f'{self.logger_name}_{time_str()}.log'),
                mode='w', encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter(self.log_format))
            logger.addHandler(file_handler)
        if self.stream is not None:
            channel_handler = logging.StreamHandler(stream=self.stream)
            channel_handler.setFormatter(logging.Formatter(self.log_format))
            logger.addHandler(channel_handler)
        return logger

    def log(self, msg: str):
        """Log intermediate Results"""
        pass

    def info(self, msg: str):
        """Print end-of-epoch stats"""
        pass


class WandbLogger(BaseLogger):
    def __init__(
        self,
        logger_name: str,
        log_dir: Optional[str] = None,
        stream: Optional[TextIO] = None,
        log_format: Optional[str] = None,
        log_level: Optional[int] = None,
        args=None,
        wandb_project: Optional[str] = None,
        wandb_logdir: Optional[str] = None
    ):
        
        super(WandbLogger, self).__init__(
            logger_name=logger_name,
            log_dir=log_dir,
            stream=stream,
            log_format=log_format,
            log_level=log_level,
            args=args,
        )
        self.wandb_dir = wandb_logdir
        if self.wandb_dir is None:
            self.wandb_dir = self.log_dir

        if wandb_project is None:
            wandb_project = self.logger_name

        # wandb.init(project=wandb_project, reinit=False, name=logger_name)
        # if args is not None:
        #     wandb.config.update(args)

    def log(self, stats: Dict, epoch=None, step=None):
        stats = _format_stats(stats, epoch=epoch, step=step)
        self.logger.info(json.dumps(stats))

    def info(self, msg: str):
        self.logger.info(msg)

    def _log_to_wandb(self, stats: Dict, step=None, epoch=None):
        pass