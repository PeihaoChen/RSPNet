"""
2020-05-09
Abstract the process of creating a model into get_model_class, model_class(num_classes=N)
"""

from pyhocon import ConfigTree, ConfigFactory
from torch import nn
import torch
from torch import nn
from typing import *
import logging

logger = logging.getLogger(__name__)


def get_model_class(**kwargs) -> Callable[[int], nn.Module]:
    """
    Pass the model config as parameters. For convinence, we change the cfg to dict, and then reverse it
    :param kwargs:
    :return:
    """
    logger.info(f'Using global get_model_class({kwargs})')

    cfg = ConfigFactory.from_dict(kwargs)

    arch: str = cfg.get_string('arch')

    if arch == 'resnet18':
        from .resnet import resnet18
        model_class = resnet18
    elif arch == 'resnet34':
        from .resnet import resnet34
        model_class = resnet34
    elif arch == 'resnet50':
        from .resnet import resnet50
        model_class = resnet50
    elif arch == 'torchvision-resnet18':
        from torchvision.models.video import r3d_18
        def model_class(num_classes):
            model = r3d_18(
                pretrained=cfg.get_bool('pretrained', default=False),
            )
            model.fc = nn.Linear(model.fc.in_features, num_classes, model.fc.bias is not None)
            return model
    elif arch == 'c3d':
        from .c3d import C3D
        model_class = C3D
    elif arch == 's3dg':
        from .s3dg import S3D_G
        model_class = S3D_G
    elif arch == 'mfnet':
        from .mfnet.mfnet_3d import MFNET_3D
        model_class = MFNET_3D
    elif arch == 'tsm':
        from models.tsm import TSM
        model_class = lambda num_classes=128: TSM(
            num_classes=num_classes,
            num_segments=cfg.get_int('num_segments'),
            base_model=cfg.get_string('base_model'),
            pretrain=cfg.get_string('pretrain', default=None),
        )
    elif arch.startswith('SLOWFAST'):
        from .slowfast import get_kineitcs_model_class_by_name
        model_class = get_kineitcs_model_class_by_name(arch)
    elif arch == 'r2plus1d-vcop':
        from .r2plus1d_vcop import R2Plus1DNet
        model_class = lambda num_classes=128: R2Plus1DNet(
            (1, 1, 1, 1),
            with_classifier=True,
            num_classes=num_classes
        )
    else:
        raise ValueError(f'Unknown model architecture "{arch}"')

    return model_class


class ModelFactory:

    def __init__(self, cfg: ConfigTree):
        self.cfg = cfg

    def _post_process_model(self, model: nn.Module):
        if self.cfg.get_bool('only_train_fc', False):
            for param in model.parameters():
                param.requires_grad = False

            fc_names = ['fc', 'new_fc']
            fc_module = next(getattr(model, n) for n in fc_names if hasattr(model, n))
            if fc_module is None:
                raise Exception('"only_train_fc" specified, but no fc layer found')

            for param in fc_module.parameters():
                param.requires_grad = True

            orig_train = model.train

            def override_train(mode=True):
                orig_train(mode=False)
                fc_module.train(mode)

            model.train = override_train

            logger.info('Only last fc layer will have grad and enter train mode')

        return model

    def build(self, local_rank: int) -> nn.Module:
        # arch = self.cfg.get_string('model.arch')
        num_classes = self.cfg.get_int('dataset.num_classes')

        model_class = get_model_class(**self.cfg.get_config('model'))

        model = model_class(num_classes=num_classes)
        model = self._post_process_model(model)

        model = model.cuda(local_rank)

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
        )
        return model

    def build_multitask_wrapper(self, local_rank: int) -> nn.Module:
        # arch = self.cfg.get_string('model.arch')

        from moco.split_wrapper import MultiTaskWrapper
        num_classes = self.cfg.get_int('dataset.num_classes')

        model_class = get_model_class(**self.cfg.get_config('model'))

        model = MultiTaskWrapper(model_class, num_classes=num_classes, finetune=True)
        model = self._post_process_model(model)

        model = model.cuda(local_rank)

        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,  # some of forward output are not involved in calculation
        )
        return model
