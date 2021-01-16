from typing import Sequence
import dataclasses
import os
from pathlib import Path
import argparse
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

def build_subset(fullset_path: Path, dest_path: Path, categories: Sequence[str]):
    dest_path.mkdir(parents=True)
    fullset_path = fullset_path.absolute()
    fullset_path = Path(os.path.relpath(fullset_path, dest_path))
    if logger.isEnabledFor(logging.INFO):
        categories = tqdm(categories)
    for c in categories:
        (dest_path/c).symlink_to(fullset_path/c, target_is_directory=True)

@dataclasses.dataclass
class Category:
    name: str
    video_size: int

def find_smallest_categories(path: Path, num_category):
    all_categories = []
    for category_path in path.iterdir():
        video_size = sum(video.stat().st_size for video in category_path.iterdir())
        all_categories.append(Category(category_path.name, video_size))
        logger.info('Discovering category "%s", total video size: %d', category_path.name, video_size)

    for c in sorted(all_categories, key=lambda c: c.video_size)[:num_category]:
        yield c.name

def main(args):
    categories = list(find_smallest_categories(args.train_full, args.num_category))
    logger.info('Building train set')
    build_subset(args.train_full, args.train_dest, categories)
    logger.info('Building val set')
    build_subset(args.val_full, args.val_dest, categories)
    os.chdir('data')
    os.system('ln -s {} {}'.format('kinetics100_links', 'kinetics100'))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_category', type=int, default=100)
    parser.add_argument('--train_full', type=Path, default=Path('data/kinetics400/train_video'))
    parser.add_argument('--train_dest', type=Path, default=Path('data/kinetics100_links/train_video'))
    parser.add_argument('--val_full', type=Path, default=Path('data/kinetics400/val_video'))
    parser.add_argument('--val_dest', type=Path, default=Path('data/kinetics100_links/val_video'))
    args = parser.parse_args()

    main(args)
