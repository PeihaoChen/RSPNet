from fvcore.common.config import CfgNode
from pyhocon import ConfigTree, ConfigFactory
from .config.defaults import get_cfg
from typing import *
from pathlib import Path
from .video_model_builder import SlowFast

CONFIG_DIR = Path('config/slowfast-configs')


def load_slowfast_cfg(cfg: ConfigTree, num_classes=None) -> CfgNode:
    base_cfg = get_cfg()
    cfg_file = cfg.get_string('model.cfg_file')
    base_cfg.merge_from_file(cfg_file)

    if num_classes is None:
        base_cfg.MODEL.NUM_CLASSES = cfg.get_int('dataset.num_classes')
    else:
        base_cfg.MODEL.NUM_CLASSES = num_classes
    return base_cfg


def get_model_from_cfg(cfg: ConfigTree, num_classes: Optional[int] = None):
    slowfast_cfg = load_slowfast_cfg(cfg, num_classes=num_classes)

    name = slowfast_cfg.MODEL.MODEL_NAME

    if name == 'SlowFast':
        from .video_model_builder import SlowFast
        model = SlowFast(slowfast_cfg)

    else:
        raise Exception

    return model


def load_slowfast_cfg_from_yaml(yaml_path: str, num_classes: Optional[int] = None) -> CfgNode:
    base_cfg = get_cfg()
    base_cfg.merge_from_file(yaml_path)

    if num_classes is not None:
        base_cfg.MODEL.NUM_CLASSES = num_classes
    return base_cfg


def get_model_from_yaml(yaml_path: str, num_classes: Optional[int] = None):
    cfg = load_slowfast_cfg_from_yaml(yaml_path)

    name = cfg.MODEL.MODEL_NAME
    if name == 'SlowFast':
        from .video_model_builder import SlowFast
        model = SlowFast(cfg)

    else:
        raise Exception

    return model


def get_kineitcs_model_class_by_name(name: str):
    kinetics_dir = CONFIG_DIR / 'Kinetics'

    def model_class(num_classes=128):
        if name.startswith('SLOWFAST'):
            cfg = load_slowfast_cfg_from_yaml(
                str(kinetics_dir / name + '.yaml'),
                num_classes=num_classes,
            )
            model = SlowFast(cfg)
        else:
            raise Exception('No rules for {}'.format(name))

        return model

    return model_class


if __name__ == '__main__':
    from .video_model_builder import SlowFast
    import torch

    torch.set_grad_enabled(False)

    cfg: CfgNode = get_cfg()
    cfg.merge_from_file('config/slowfast-configs/Kinetics/SLOWFAST_4x16_R50.yaml')
    print(cfg)
    model = SlowFast(cfg)
    print(model)
    x = torch.rand(2, 3, 32, 224, 224)
    y = model(x)
    print(y.shape)

    x = torch.rand(2, 3, 16, 112, 112)
    y = model(x)
    print(y.shape)
