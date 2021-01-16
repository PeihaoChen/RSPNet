from .arguments import Args
from _jsonnet import evaluate_snippet
from pyhocon import ConfigFactory, ConfigTree, HOCONConverter
import json
import logging
import re
import os
from pathlib import Path
import difflib

logger = logging.getLogger(__name__)


def config_snippet(ext_config_count: int):
    snippet = 'local base = import "__base_config__";\n'
    for i in range(ext_config_count):
        snippet += f'local arg{i} = import "__arg_{i}__";\n'

    snippet += 'base'
    for i in range(ext_config_count):
        snippet += f' + arg{i}'
    return snippet


def ext_config_template(ext_config: str):
    snippet = 'local add = import "__addition_config__";\n'
    snippet += ext_config
    return snippet


def try_path(dir, rel):
    if rel[0] == '/':
        full_path = rel
    else:
        full_path = dir + rel

    with open(full_path) as f:
        return full_path, f.read()


arg_regex = re.compile('^__arg_(\d+)__$')


def get_config(args: Args) -> ConfigTree:
    def import_callback(dir, rel):
        arg_match = arg_regex.match(rel)
        if arg_match is not None:
            full_path = rel
            index = int(arg_match.group(1))
            content = ext_config_template(args.ext_config[index])
        else:
            if rel == '__base_config__':
                rel = Path(args.config)
            elif rel == '__addition_config__':
                rel = Path(args.config).with_name('addition.libsonnet')
            else:
                rel = Path(rel)
            full_path = dir / rel
            full_path = str(full_path)
            with open(full_path) as f:
                content = f.read()
        return full_path, content

    json_str = evaluate_snippet(
        '__composed_config__',
        config_snippet(len(args.ext_config)),
        import_callback=import_callback,
    )

    json_obj = json.loads(json_str)
    cfg = ConfigFactory.from_dict(json_obj)

    logger.info(f'Config = \n{HOCONConverter.to_hocon(cfg)}')

    return cfg


def save_config(args: Args, cfg: ConfigTree):
    config_path = args.run_dir / 'config.json'
    with open(config_path, 'w') as f:
        f.write(HOCONConverter.to_json(cfg))
