import subprocess
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


def pack_code(run_dir: Path):
    if os.path.isdir("./.git"):
        subprocess.run(
            ['git', 'archive', '-o', str(run_dir/'code.tar.gz'), 'HEAD'],
            check=True,
        )
        diff_process = subprocess.run(
            ['git', 'diff', 'HEAD'],
            check=True, stdout=subprocess.PIPE, text=True,
        )
        if diff_process.stdout:
            logger.warning('Working tree is dirty. Patch:\n%s', diff_process.stdout)
            with (run_dir / 'dirty.patch').open('w') as f:
                f.write(diff_process.stdout)
    else:
        logger.warning('.git does not exist in current dir')
