from typing import Optional
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
import logging

logger = logging.getLogger(__name__)


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup_epoch: int, after_scheduler: _LRScheduler, last_epoch=-1):
        if warmup_epoch < 0:
            raise ValueError(
                'warmup_epoch should be greater than or equal to zero')
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler

        after_last_epoch = last_epoch - warmup_epoch
        if after_last_epoch < 0:
            after_last_epoch = -1
        super().__init__(optimizer=optimizer, last_epoch=after_last_epoch)

    @property
    def warmed_up(self):
        return self.last_epoch >= self.warmup_epoch

    def state_dict(self) -> dict:
        states = {key: value for key, value in vars(
            self).items() if key not in ('optimizer', 'after_scheduler')}
        states['after_scheduler_states'] = self.after_scheduler.state_dict()
        return states

    def load_state_dict(self, state_dict: dict):
        if 'after_scheduler_states' in state_dict:
            self.after_scheduler.load_state_dict(
                state_dict['after_scheduler_states'])
            del state_dict['after_scheduler_states']
        else:
            logger.warning('no after_scheduler_states')
        super().load_state_dict(state_dict)

    def step(self, epoch: Optional[int] = None):
        if self.warmed_up:
            after_epoch = epoch - self.warmup_epoch if epoch is not None else None
            self.after_scheduler.step(after_epoch)
        else:
            super().step(epoch)

    def get_lr(self):
        if self.warmup_epoch == 0:
            return self.base_lrs
        lr_rate = self.last_epoch / self.warmup_epoch
        return [lr_rate * base_lr for base_lr in self.base_lrs]

    def get_last_lr(self):
        if self.warmed_up:
            return self.after_scheduler.get_last_lr()
        else:
            return super().get_last_lr()
