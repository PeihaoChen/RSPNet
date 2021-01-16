import torch
from torch.backends import cudnn
import logging
import warnings

logger = logging.getLogger(__name__)


def lock_random_seed(seed: int):
    import random
    random.seed(seed)


def lock_numpy_seed(seed: int):
    import numpy as np
    np.random.seed(seed)


def cudnn_benchmark():
    cudnn.benchmark = True
    logger.info(f'cudnn.benchmark = {cudnn.benchmark}')


def lock_torch_seed(seed: int):
    import torch
    torch.manual_seed(seed)


def initialize_seed(seed: int):
    lock_random_seed(seed)
    lock_torch_seed(seed)
    lock_numpy_seed(seed)
    cudnn.deterministic = True

    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')
