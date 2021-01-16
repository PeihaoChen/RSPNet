''' TSM model
This module is based on https://github.com/mit-han-lab/temporal-shift-module/tree/f5606ac4556bb5ec3a9ab63d0888fe3c11a7a7a7
'''

from .models import TSN as _TSN
from .models_half import TSN as _TSNHalf
from .models_group import TSN as _TSNGroup
import logging

logger = logging.getLogger(__name__)


class TSM(_TSN):
    ''' Same as model from TSM repository.

    Except default parameters updated from script `train_tsm_kinetics_rgb_8f.sh`.
    - Except not use ImageNet pretrain
    '''

    def __init__(self,
                 num_classes, modality='RGB', base_model='resnet50', pretrain=None,
                 num_segments=8, dropout=0.5, consensus_type='avg',
                 is_shift=True, shift_div=8, shift_place='blockres',
                 partial_bn=False,
                 img_feature_dim=256,
                 temporal_pool=False,
                 non_local=False,
                 fc_lr5=True,
                 **kwargs):
        additional_args = {
            k: v for k, v in locals().items()
            if k not in ['num_classes', 'self', 'kwargs'] and not k.startswith('__')
        }
        kwargs.update(additional_args)
        super().__init__(num_class=num_classes, **kwargs)


class TSMHalf(_TSNHalf):
    ''' Same as model from TSM repository.

    Except default parameters updated from script `train_tsm_kinetics_rgb_8f.sh`.
    - Except not use ImageNet pretrain

    '''

    def __init__(self,
                 num_classes, modality='RGB', base_model='resnet50', pretrain=None,
                 num_segments=8, dropout=0.5, consensus_type='avg',
                 is_shift=True, shift_div=8, shift_place='blockres',
                 partial_bn=False,
                 img_feature_dim=256,
                 temporal_pool=False,
                 non_local=False,
                 fc_lr5=True,
                 inplanes=64,
                 **kwargs):
        additional_args = {
            k: v for k, v in locals().items()
            if k not in ['num_classes', 'self', 'kwargs'] and not k.startswith('__')
        }
        kwargs.update(additional_args)
        super().__init__(num_class=num_classes, **kwargs)


class TSMGroup(_TSNGroup):
    ''' Same as model from TSM repository.

    Except default parameters updated from script `train_tsm_kinetics_rgb_8f.sh`.
    - Except not use ImageNet pretrain
    '''

    def __init__(self,
                 num_classes, modality='RGB', base_model='resnet50', pretrain=None,
                 num_segments=8, dropout=0.5, consensus_type='avg',
                 is_shift=True, shift_div=8, shift_place='blockres',
                 partial_bn=False,
                 img_feature_dim=256,
                 temporal_pool=False,
                 non_local=False,
                 fc_lr5=True,
                 **kwargs):
        additional_args = {
            k: v for k, v in locals().items()
            if k not in ['num_classes', 'self', 'kwargs'] and not k.startswith('__')
        }
        kwargs.update(additional_args)
        super().__init__(num_class=num_classes, **kwargs)
        groups = kwargs.get('groups')
        conv1_stage = kwargs.get('conv1_stage')
        logger.warning(f'You are using TSM Group model (just for experimental usage), groups: {groups}, conv1_stage: {conv1_stage}')


if __name__ == '__main__':
    import torch
    import time

    x = torch.rand(2, 3, 8, 112, 112)
    m = TSM(10)
    y = m(x)
    start = time.perf_counter()
    y = m(x)
    end = time.perf_counter()
    print(y.shape)
    print(end - start)
