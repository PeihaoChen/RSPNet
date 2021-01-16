import logging
import resource

logger = logging.getLogger(__name__)


def ulimit_n_max():
    _soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))
    logger.warning('Setting ulimit -n %d', hard_limit)


def scale_learning_rate(lr: float, world_size: int, batch_size: int, base_batch_size: int = 64) -> float:
    new_lr = lr * world_size * batch_size / base_batch_size
    logger.warning(f'adjust lr according to the number of GPU and batch size：{lr} -> {new_lr}')
    return new_lr
