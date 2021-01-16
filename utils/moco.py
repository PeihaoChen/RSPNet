import torch
from pyhocon import ConfigTree
import logging

logger = logging.getLogger(__name__)


def trim_moco_k(k: int, batch_size: int, world_size: int) -> int:
    total_batch_size = batch_size * world_size
    return k // total_batch_size * total_batch_size


def replace_moco_k_in_config(cfg: ConfigTree, moco_k_key='moco.k', batch_size_key='batch_size'):
    ''' Ensure k is a multiple of batch_size * world_size
    '''
    moco_k = cfg.get_int(moco_k_key)
    batch_size = cfg.get_int(batch_size_key)
    world_size = torch.cuda.device_count()
    trimed_moco_k = trim_moco_k(moco_k, batch_size, world_size)
    cfg.put(moco_k_key, trimed_moco_k)
    logger.warning(f'Changing MoCo K: {moco_k} => {trimed_moco_k}')
