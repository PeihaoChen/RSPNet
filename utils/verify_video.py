import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import pickle
import asyncio
import json

logger = logging.getLogger(__name__)

async def verify(video_path: Path, failed: list):
    proc = await asyncio.create_subprocess_exec(
        'ffprobe', '-loglevel', 'warning', '-show_streams', '-select_streams', 'v', '-print_format', 'json', str(video_path),
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error('ffprobe %s failed with return code %d', video_path, proc.returncode)
        failed.append(video_path)
        return

    probe_result = json.loads(stdout)
    if not probe_result['streams']:
        logger.error('No Video found for "%s"', video_path)
        failed.append(video_path)
        return


async def main(args):
    input_dir: Path = args.input

    tasks = set()
    def search_files():
        yield from input_dir.glob('**/*.mp4')
        yield from input_dir.glob('**/*.avi')
    pending_videos = sorted(search_files())

    failed = list()

    with tqdm(total=len(pending_videos), smoothing=0.1) as progress:
        while True:
            while len(tasks) < args.jobs and pending_videos:
                raw_video = pending_videos.pop()
                t = asyncio.create_task(verify(raw_video, failed))
                tasks.add(t)

            if not tasks:
                break

            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                await t
                progress.update()

    print('The following video failed the test: ')
    for p in failed:
        print(p.relative_to(input_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--jobs', default=32, type=int)
    parser.add_argument('input', type=Path)

    args = parser.parse_args()

    asyncio.run(main(args))
