import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import asyncio

logger = logging.getLogger(__name__)
MAX_TASK = 16

async def transcode(raw_video: Path, input_dir: Path, output_dir: Path):
    output = output_dir / raw_video.relative_to(input_dir)
    output = output.with_suffix('.mp4')
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists()
    proc = await asyncio.create_subprocess_exec(
        'ffmpeg', '-loglevel', 'warning', '-i', str(raw_video), '-c:v', 'libx264', '-filter:v', 'scale=w=-2:h=256', '-g', '16', '-tune', 'fastdecode', '-an', str(output)
        # 'cp', str(raw_video), str(output)
    )
    returncode = await proc.wait()
    if returncode != 0:
        logger.error('Transcode %s failed with return code %d', raw_video, returncode)

async def main(args):
    input_dir: Path = args.input
    output_dir: Path = args.output

    tasks = set()
    def search_files():
        yield from input_dir.glob('**/*.mp4')
        yield from input_dir.glob('**/*.avi')
    pending_videos = sorted(search_files())
    with tqdm(total=len(pending_videos)) as progress:
        while True:
            while len(tasks) < MAX_TASK and pending_videos:
                raw_video = pending_videos.pop()
                t = asyncio.create_task(transcode(raw_video, input_dir, output_dir))
                tasks.add(t)

            if not tasks:
                break

            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                await t
                progress.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)

    args = parser.parse_args()

    asyncio.run(main(args))
