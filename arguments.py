import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import *

import torch
from typed_args import add_argument

from framework.arguments import Args as BaseArgs

logger = logging.getLogger(__name__)


def get_world_size() -> int:
    """
    It has to be larger than 2. Otherwise, the shuffle bn cannot work.
    :return:
    """
    num_gpus = torch.cuda.device_count()
    return max(2, num_gpus)
    # return num_gpus


@dataclass
class Args(BaseArgs):
    load_checkpoint: Optional[Path] = add_argument(
        '--load-checkpoint', required=False,
        help='path to the checkpoint file to be loaded'
    )
    load_model: Optional[Path] = add_argument(
        '--load-model', required=False,
        help='path to the checkpoint file to be loaded, but only load model.'
    )
    validate: bool = add_argument(
        '--validate', action='store_true',
        help='Only run final validate then exit'
    )
    moco_checkpoint: Optional[str] = add_argument(
        '--mc', '--moco-checkpoint',
        help='load moco checkpoint'
    )
    seed: Optional[int] = add_argument(
        '--seed', help='random seed'
    )
    world_size: int = add_argument(
        '--ws', '--world-size', default=torch.cuda.device_count(),
        help='total processes'
    )
    _continue: bool = add_argument(
        '--continue', action='store_true',
        help='Use previous config and checkpoint',
    )
    no_scale_lr: bool = add_argument(
        '--no-scale-lr', action='store_true',
        help='Do not change lr according to batch size'
    )

    def resolve_continue(self):
        if not self._continue:
            return
        if not self.experiment_dir.exists():
            raise EnvironmentError(f'Experiment directory "{self.experiment_dir}" does not exists.')

        if self.config is None:
            run_id = -1
            for run in self.experiment_dir.iterdir():
                match = self.RUN_DIR_NAME_REGEX.match(run.name)
                if match is not None:
                    this_run_id = int(match.group(1))
                    if this_run_id > run_id and run.is_dir():
                        this_config_path = run / 'config.json'
                        if this_config_path.exists():
                            run_id = this_run_id
                            self.config = this_config_path
            if self.config is None:
                raise EnvironmentError(f'No previous run config found')
            logger.info('Continue using previous config: "%s"', self.config)
        if self.load_checkpoint is None:
            checkpoint_path = self.experiment_dir / 'checkpoint.pth.tar'
            if checkpoint_path.exists():
                self.load_checkpoint = checkpoint_path
                logger.info('Continue using previous checkpoint: "%s"', self.load_checkpoint)
            else:
                logger.warning('No previous checkpoint found')
