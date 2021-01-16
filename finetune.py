import time
import logging
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pyhocon import ConfigTree
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from arguments import Args
from datasets.classification import DataLoaderFactoryV3
from framework import utils
from framework.config import get_config, save_config
from framework.logging import set_logging_basic_config
from framework.meters.average import AverageMeter
from framework.metrics.classification import accuracy
from framework.utils import CheckpointManager, pack_code
from models import ModelFactory

logger = logging.getLogger(__name__)


class EpochContext:
    def __init__(self, engine: 'Engine', name: str, n_crop: int, dataloader, tensorboard_prefix: str):
        self.engine = engine
        self.log_interval = engine.cfg.get_int('log_interval')

        self.n_crop = n_crop
        self.name = name
        self.dataloader = dataloader
        self.tensorboard_prefix = tensorboard_prefix

        self.dataloader.set_epoch(self.engine.current_epoch)
        # start dataloader early for better performance
        self.data_iter = iter(dataloader)

        device = self.engine.device
        self.loss_meter = AverageMeter('Loss', device=device)  # This place displays decimals directly because the loss is relatively large
        self.top1_meter = AverageMeter('Acc@1', fmt=':6.2f', device=device)
        self.top5_meter = AverageMeter('Acc@5', fmt=':6.2f', device=device)

    def reshape_clip(self, clip: torch.FloatTensor):
        if self.n_crop == 1:
            return clip
        clip = clip.refine_names('batch', 'channel', 'time', 'height', 'width')
        crop_len = clip.size(2) // self.n_crop
        clip = clip.unflatten('time', [('crop', self.n_crop), ('time', crop_len)])
        clip = clip.align_to('batch', 'crop', ...)
        clip = clip.flatten(['batch', 'crop'], 'batch')
        return clip.rename(None)

    def average_logits(self, logits: torch.FloatTensor):
        if self.n_crop == 1:
            return logits
        logits = logits.refine_names('batch', 'class')
        num_sample = logits.size(0) // self.n_crop
        logits = logits.unflatten('batch', [('batch', num_sample), ('crop', self.n_crop)])
        logits = logits.mean(dim='crop')
        return logits.rename(None)

    def meters(self):
        yield self.loss_meter
        yield self.top1_meter
        yield self.top5_meter

    def sync_meters(self):
        for m in self.meters():
            m.sync_distributed()

    def write_tensorboard(self):
        epoch = self.engine.current_epoch
        prefix = self.tensorboard_prefix
        tb = self.engine.summary_writer
        if tb is None:
            return

        tb.add_scalar(
            f'{prefix}/loss', self.loss_meter.avg, epoch
        )
        tb.add_scalar(
            f'{prefix}/acc1', self.top1_meter.avg, epoch
        )
        tb.add_scalar(
            f'{prefix}/acc5', self.top5_meter.avg, epoch
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.write_tensorboard()

    def forward(self):
        logger.info('%s epoch begin.', self.name)
        begin_time = time.perf_counter()
        num_iters = len(self.dataloader)
        remaining_valid_samples = self.dataloader.num_valid_samples()

        for i, ((clip,), target, *others) in enumerate(self.data_iter):
            clip = self.reshape_clip(clip)
            output = self.engine.model(clip)
            output = self.average_logits(output)
            loss = self.engine.criterion(output, target)

            # This will make tensorboard load very slow. enable if needed
            # if self.engine.summary_writer is not None:
            #     self.engine.summary_writer.add_scalar(f'step/{self.tensorboard_prefix}/loss', loss,
            #         self.engine.current_epoch * num_iters + i)

            batch_size = target.size(0)
            if batch_size > remaining_valid_samples:
                # Distributed sampler will add some repeated samples. cut them off.
                output = output[:remaining_valid_samples]
                target = target[:remaining_valid_samples]
                others = [o[:remaining_valid_samples] for o in others]
                batch_size = remaining_valid_samples
            remaining_valid_samples -= batch_size

            if batch_size == 0:
                continue

            if i > 0 and i % self.log_interval == 0:
                # Do logging as late as possible. this will force CUDA sync.
                # Log numbers from last iteration, just before update
                logger.info(
                    f'{self.name} [{self.engine.current_epoch}/{self.engine.num_epochs}][{i - 1}/{num_iters}]\t'
                    f'{self.loss_meter}\t{self.top1_meter}\t{self.top5_meter}'
                )

            num_classes = output.size(1)
            if num_classes >= 5:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                self.top1_meter.update(acc1, batch_size)
                self.top5_meter.update(acc5, batch_size)
            else:
                acc1, = accuracy(output, target, topk=(1,))
                self.top1_meter.update(acc1, batch_size)

            self.loss_meter.update(loss, batch_size)

            yield loss, output, others

        end_time = time.perf_counter()
        logger.info('%s epoch finished. Time: %.2f sec.\t%s\t%s\t%s', self.name, end_time - begin_time, *self.meters())


class Engine:

    def __init__(self, args: Args, cfg: ConfigTree, local_rank: int, final_validate=False):
        self.args = args
        self.cfg = cfg
        self.local_rank = local_rank

        self.model_factory = ModelFactory(cfg)
        self.data_loader_factory = DataLoaderFactoryV3(cfg, final_validate)
        self.final_validate = final_validate

        self.device = torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        model_type = cfg.get_string('model_type')
        if model_type == '1stream':
            self.model = self.model_factory.build(local_rank)  # basic model
        elif model_type == 'multitask':
            self.model = self.model_factory.build_multitask_wrapper(local_rank)
        else:
            raise ValueError(f'Unrecognized model_type "{model_type}"')
        if not final_validate:
            self.train_loader = self.data_loader_factory.build(
                vid=False,  # need label to gpu
                split='train',
                device=self.device
            )
        self.validate_loader = self.data_loader_factory.build(
            vid=False,
            split='val',
            device=self.device
        )

        if final_validate:
            self.n_crop = cfg.get_int('temporal_transforms.validate.final_n_crop')
        else:
            self.n_crop = cfg.get_int('temporal_transforms.validate.n_crop')

        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = self.cfg.get_float('optimizer.lr')
        optimizer_type = self.cfg.get_string('optimizer.type', default='sgd')
        if optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=self.cfg.get_float('optimizer.momentum'),
                dampening=self.cfg.get_float('optimizer.dampening'),
                weight_decay=self.cfg.get_float('optimizer.weight_decay'),
                nesterov=self.cfg.get_bool('optimizer.nesterov'),
            )
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                eps=self.cfg.get_float('optimizer.eps'),
            )
        else:
            raise ValueError(f'Unknown optimizer {optimizer_type})')

        self.num_epochs = cfg.get_int('num_epochs')
        self.schedule_type = self.cfg.get_string('optimizer.schedule')
        if self.schedule_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer,
                mode='min',
                patience=self.cfg.get_int('optimizer.patience'),
                verbose=True
            )
        elif self.schedule_type == "multi_step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.cfg.get("optimizer.milestones"),
            )
        elif self.schedule_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.learning_rate / 1000
            )
        elif self.schedule_type == 'none':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda epoch: 1,
            )
        else:
            raise ValueError("Unknow schedule type")

        self.arch = cfg.get_string('model.arch')

        if local_rank == 0:
            self.summary_writer = SummaryWriter(
                log_dir=str(args.experiment_dir)
            )
        else:
            self.summary_writer = None

        self.best_acc1 = 0.
        self.current_epoch = 0
        self.next_epoch = None
        logger.info('Engine: n_crop=%d', self.n_crop)

        self.checkpoint_manager = CheckpointManager(
            self.args.experiment_dir, keep_interval=None
        )
        self.loss_meter = None

    def has_next_epoch(self):
        return not self.final_validate and self.current_epoch < self.num_epochs - 1

    def load_checkpoint(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=self.device)
        if states['arch'] != self.arch:
            raise ValueError(f'Loading checkpoint arch {states["arch"]} does not match current arch {self.arch}')

        logger.info('Loading checkpoint from %s', checkpoint_path)
        self.model.module.load_state_dict(states['model'])
        logger.info('Checkpoint loaded')

        self.optimizer.load_state_dict(states['optimizer'])
        self.scheduler.load_state_dict(states['scheduler'])
        self.current_epoch = states['epoch']
        self.best_acc1 = states['best_acc1']

    def load_moco_checkpoint(self, checkpoint_path: str):
        cp = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in cp and 'arch' in cp:
            logger.info('Loading MoCo checkpoint from %s (epoch %d)', checkpoint_path, cp['epoch'])
            moco_state = cp['model']
            prefix = 'encoder_q.'
        else:
            # This checkpoint is from third-party
            logger.info('Loading third-party model from %s', checkpoint_path)
            if 'state_dict' in cp:
                moco_state = cp['state_dict']
            else:
                # For c3d
                moco_state = cp
                logger.warning('if you are not using c3d sport1m, maybe you use wrong checkpoint')
            if next(iter(moco_state.keys())).startswith('module'):
                prefix = 'module.'
            else:
                prefix = ''

        """
        fc -> fc. for c3d sport1m. Beacuse fc6 and fc7 is in use.
        """
        blacklist = ['fc.', 'linear', 'head', 'new_fc', 'fc8']
        blacklist += ['encoder_fuse']

        def filter(k):
            return k.startswith(prefix) and not any(k.startswith(f'{prefix}{fc}') for fc in blacklist)

        model_state = {k[len(prefix):]: v for k, v in moco_state.items() if filter(k)}
        msg = self.model.module.load_state_dict(model_state, strict=False)
        # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} or \
        #        set(msg.missing_keys) == {"linear.weight", "linear.bias"} or \
        #        set(msg.missing_keys) == {'head.projection.weight', 'head.projection.bias'} or \
        #        set(msg.missing_keys) == {'new_fc.weight', 'new_fc.bias'},\
        #     msg

        logger.warning(f'Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}')

    def train_context(self):
        return EpochContext(
            self, name='Train',
            n_crop=1,
            dataloader=self.train_loader,
            tensorboard_prefix='train')

    def validate_context(self):
        return EpochContext(
            self, name='Validate',
            n_crop=self.n_crop,
            dataloader=self.validate_loader,
            tensorboard_prefix='val')

    def train_epoch(self):
        epoch = self.next_epoch
        if epoch is None:
            epoch = self.train_context()
        self.next_epoch = self.validate_context()

        self.model.train()
        with epoch:
            for loss, *_ in epoch.forward():
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.loss_meter = epoch.loss_meter

    def validate_epoch(self):
        epoch = self.next_epoch
        if epoch is None:
            epoch = self.validate_context()
        if self.has_next_epoch():
            self.next_epoch = self.train_context()
        else:
            self.next_epoch = None

        self.model.eval()
        all_logits = torch.empty(0, device=next(self.model.parameters()).device)
        indices = []
        with epoch:
            with torch.no_grad():
                for _, logits, others in epoch.forward():
                    all_logits = torch.cat((all_logits, logits), dim=0)
                    if others:
                        assert len(others[0]) == logits.size(0), \
                            f'Length of indices and logits not match. {others[0]} vs {logits.size(0)}'
                        indices.extend(others[0])

            epoch.sync_meters()
            logger.info('Validation finished.\n\tLoss = %f\n\tAcc@1 = %.2f%% (%d/%d)\n\tAcc@5 = %.2f%% (%d/%d)',
                        epoch.loss_meter.avg.item(),
                        epoch.top1_meter.avg.item(), epoch.top1_meter.sum.item() / 100, epoch.top1_meter.count.item(),
                        epoch.top5_meter.avg.item(), epoch.top5_meter.sum.item() / 100, epoch.top5_meter.count.item(),
                        )

        if self.final_validate:
            ds = self.validate_loader.dataset
            if hasattr(ds, 'save_results'):
                assert indices, 'Dataset should return indices to sort logits'
                assert len(indices) == all_logits.size(0), \
                    f'Length of indices and logits not match. {len(indices)} vs {all_logits.size(0)}'
                with (self.args.experiment_dir / f'results_{self.local_rank}.json').open('w') as f:
                    ds.save_results(f, indices, all_logits)
        return epoch.top1_meter.avg.item()

    def run(self):

        num_epochs = 1 if self.args.debug else self.num_epochs

        self.model.train()

        while self.current_epoch < num_epochs:
            logger.info("Current LR:{}".format(self.scheduler._last_lr))
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('train/lr', utils.get_lr(self.optimizer), self.current_epoch)
            self.train_epoch()
            acc1 = self.validate_epoch()
            if self.schedule_type == "plateau":
                self.scheduler.step(self.loss_meter.val.item())
            else:
                self.scheduler.step()

            self.current_epoch += 1

            if self.local_rank == 0:
                is_best = acc1 > self.best_acc1
                self.best_acc1 = max(acc1, self.best_acc1)

                # save_checkpoint({
                #     'epoch': self.current_epoch,
                #     'arch': self.arch,
                #     'model': self.model.module.state_dict(),
                #     'best_acc1': self.best_acc1,
                #     'optimizer': self.optimizer.state_dict(),
                #     'scheduler': self.scheduler.state_dict(),
                # }, is_best, self.args.experiment_dir)
                self.checkpoint_manager.save(
                    {
                        'epoch': self.current_epoch,
                        'arch': self.arch,
                        'model': self.model.module.state_dict(),
                        'best_acc1': self.best_acc1,
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    },
                    is_best,
                    self.current_epoch
                )

        if self.summary_writer is not None:
            self.summary_writer.flush()


def main_worker(local_rank: int, args: Args, dist_url: str):
    print('Local Rank:', local_rank)

    # log in main process only
    if local_rank == 0:
        set_logging_basic_config(args)

    logger.info(f'Args = \n{args}')

    if args.config is not None and args.experiment_dir is not None:
        # Open multi-process. We only have one group, which is on the current node.
        dist.init_process_group(
            backend='nccl',
            init_method=dist_url,
            world_size=args.world_size,
            rank=local_rank,
        )
        utils.reproduction.cudnn_benchmark()

        cfg = get_config(args)
        if local_rank == 0:
            save_config(args, cfg)
            args.save()

        with torch.cuda.device(local_rank):
            if not args.validate:
                engine = Engine(args, cfg, local_rank=local_rank)
                if args.load_checkpoint is not None:
                    engine.load_checkpoint(args.load_checkpoint)
                elif args.moco_checkpoint is not None:
                    engine.load_moco_checkpoint(args.moco_checkpoint)
                engine.run()
                validate_checkpoint = args.experiment_dir / 'model_best.pth.tar'
            else:
                validate_checkpoint = args.load_checkpoint
                if not validate_checkpoint:
                    raise ValueError('With "--validate" specified, you should also specify "--load-checkpoint"')

            logger.info('Doing final validate.')
            engine = Engine(args, cfg, local_rank=local_rank, final_validate=True)
            engine.load_checkpoint(validate_checkpoint)
            engine.validate_epoch()
            if engine.summary_writer is not None:
                engine.summary_writer.flush()

    else:
        logger.warning('No config. Do nothing.')


def main():
    args = Args.from_args()

    if args.seed is not None:
        utils.reproduction.initialize_seed(args.seed)

    # run in main process for preventing concurrency conflict
    args.resolve_continue()
    args.make_run_dir()
    args.save()
    pack_code(args.run_dir)

    utils.environment.ulimit_n_max()

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'

    print(f'world_size={args.world_size} Using dist_url={dist_url}')

    """
    We only consider single node here. 'world_size' is the number of processes.
    """
    args.parser = None
    mp.spawn(main_worker, args=(args, dist_url,), nprocs=args.world_size)


if __name__ == '__main__':
    main()
