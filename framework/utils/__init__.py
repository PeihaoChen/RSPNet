from . import (
    reproduction,
    distributed,
    environment,
)
from .code_pack import pack_code
from .checkpoint import CheckpointManager
from torch.optim.optimizer import Optimizer


def get_lr(optimizer: Optimizer) -> float:
    return optimizer.param_groups[0]['lr']
