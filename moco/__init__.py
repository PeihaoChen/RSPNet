import logging

from pyhocon import ConfigTree
from torch import distributed as dist
from torch import nn

from models import get_model_class
from .builder_diffspeed_diffloss import MoCoDiffLossTwoFc
from .split_wrapper import MultiTaskWrapper

logger = logging.getLogger(__name__)


class ModelFactory:

    def __init__(self, cfg: ConfigTree):
        self.cfg = cfg

    def build_moco_diffloss(self):
        moco_dim = self.cfg.get_int('moco.dim')
        moco_t = self.cfg.get_float('moco.t')
        moco_k = self.cfg.get_int('moco.k')
        moco_m = self.cfg.get_float('moco.m')
        moco_fc_type = self.cfg.get_string('moco.fc_type')
        moco_diff_speed = self.cfg.get_list('moco.diff_speed')

        base_model_class = get_model_class(**self.cfg.get_config('model'))

        def model_class(num_classes=128):
            model = MultiTaskWrapper(
                base_model_class,
                num_classes=num_classes,
                fc_type=moco_fc_type,
                finetune=False,
                groups=1,
            )
            return model

        model = MoCoDiffLossTwoFc(
            model_class,
            dim=moco_dim,
            K=moco_k,
            m=moco_m,
            T=moco_t,
            diff_speed=moco_diff_speed,
        )

        model.cuda()
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_rank()],
            find_unused_parameters=True,
        )

        return model
