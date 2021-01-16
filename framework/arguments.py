import argparse
import os
import re
import shutil
import sys
import time
from shlex import quote
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import *

from typed_args import TypedArgs, add_argument


def get_timestamp(fmt: str = '%Y%m%d_%H%M%S') -> str:
    timestamp = time.strftime(fmt, time.localtime())
    return timestamp


@dataclass
class Args(TypedArgs):
    config: Optional[str] = add_argument(
        '-c', '--config',
        help='path to config'
    )
    ext_config: List[str] = add_argument(
        '-x', '--ext-config',
        nargs='*', default=[],
        help='Extra jsonnet config',
    )
    debug: bool = add_argument(
        '-d', '--debug', action='store_true',
        help='debug flag'
    )
    experiment_dir: Optional[Path] = add_argument(
        '-e', '--experiment-dir',
        const=Path('temp') / get_timestamp(), nargs=argparse.OPTIONAL,
        help='experiment dir'
    )
    _run_dir: Optional[Path] = add_argument(
        '--run-dir'
    )

    def __repr__(self):
        d = self.__dict__.copy()
        d.pop('parser')
        return pformat(d)

    def save(self):
        with open(self.run_dir / 'run.sh', 'w') as f:
            f.write(f'cd {quote(os.getcwd())}\n')
            envs = ['CUDA_VISIBLE_DEVICES']
            for env in envs:
                value = os.environ.get(env, None)
                if value is not None:
                    f.write(f'export {env}={quote(value)}\n')
            f.write(sys.executable + ' ' + ' '.join(quote(arg) for arg in sys.argv) + '\n')

    RUN_DIR_NAME_REGEX = re.compile('^run_(\d+)_')

    @property
    def run_dir(self):
        if self.experiment_dir is not None and self._run_dir is None:
            run_id = -1
            if self.experiment_dir.exists():
                for previous_runs in self.experiment_dir.iterdir():
                    match = self.RUN_DIR_NAME_REGEX.match(previous_runs.name)
                    if match is not None:
                        run_id = max(int(match.group(1)), run_id)

            run_id += 1
            self._run_dir = self.experiment_dir / f'run_{run_id}_{get_timestamp()}'
        return self._run_dir

    def make_run_dir(self):
        if self.experiment_dir is not None:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            if not self.ask_for_replacing_older_dir(self.run_dir):
                raise EnvironmentError(f'Run dir "{self.run_dir}" exists')
            self.run_dir.mkdir(parents=True, exist_ok=False)

    def make_experiment_dir(self):
        if not self.ask_for_replacing_older_dir(self.experiment_dir):
            raise EnvironmentError(f'Experiment dir "{self.experiment_dir}" exists')
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def ask_for_replacing_older_dir(self, dir_to_be_replaced: Path) -> bool:
        if not dir_to_be_replaced.exists():
            return True

        print(
            f'File exists: {dir_to_be_replaced}\nDo you want to remove it and create a new one?'
        )
        choice = input('Remove older directory? [y]es/[n]o: ')

        if choice in ['y', 'yes']:
            shutil.rmtree(dir_to_be_replaced)
            return True
        return False
