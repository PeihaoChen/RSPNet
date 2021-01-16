import logging
from .arguments import Args
import sys
import os
from tqdm import tqdm

class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

def set_logging_basic_config(args: Args, tqdm=False):
    format = '%(asctime)s|%(levelname)-8s|%(message)s'
    formatter = logging.Formatter(fmt=format)

    handlers = []
    if tqdm:
        console_handler = TqdmHandler()
    else:
        console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    if args.run_dir is not None:
        filename = args.run_dir / 'experiment.log'
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    level = logging.DEBUG if args.debug else logging.INFO

    logging.basicConfig(
        handlers=handlers,
        level=level
    )
