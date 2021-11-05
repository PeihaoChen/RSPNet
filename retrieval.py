from pyhocon.config_tree import ConfigTree
import sklearn.metrics as metrics
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from arguments import Args
from datasets.classification import DataLoaderFactoryV3
from framework import utils
from framework.config import get_config, save_config
from framework.logging import set_logging_basic_config
from framework.meters.average import AverageMeter
from framework.metrics.classification import accuracy
from framework.utils import CheckpointManager, pack_code
from models import ModelFactory

import argparse
import logging
import os
import os.path as P
logger = logging.getLogger(__name__)
from tqdm import tqdm
import json
import numpy as np
import random


class Engine:

    def __init__(self, args: Args, cfg: ConfigTree, local_rank: int, final_validate=True):
        self.args = args
        self.cfg = cfg

        self.model_factory = ModelFactory(cfg)
        self.data_loader_factory = DataLoaderFactoryV3(cfg, final_validate)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.model_factory.build(local_rank)

        self.train_loader = self.data_loader_factory.build(
            vid=False,
            split='train',
            device=self.device)
        self.val_loader = self.data_loader_factory.build(
            vid=False,
            split='val',
            device=self.device)

        self.n_crop = cfg.get_int('temporal_transforms.validate.final_n_crop')
        self.arch = cfg.get_string('model.arch')
        
        self.train_feat_list = []
        self.train_label_list = []
        self.test_feat_list = []
        self.test_label_list = []

    def reshape_clip(self, clip: torch.FloatTensor):
        if self.n_crop == 1:
            return clip
        clip = clip.refine_names('batch', 'channel', 'time', 'height', 'width')
        crop_len = clip.size(2) // self.n_crop
        clip = clip.unflatten('time', [('crop', self.n_crop), ('time', crop_len)])
        clip = clip.align_to('batch', 'crop', ...)
        clip = clip.flatten(['batch', 'crop'], 'batch')
        return clip.rename(None)

    def average_clips(self, logits: torch.FloatTensor):
        if self.n_crop == 1:
            return logits
        logits = logits.refine_names('batch', 'feat_dim')
        num_sample = logits.size(0) // self.n_crop
        logits = logits.unflatten('batch', [('batch', num_sample), ('crop', self.n_crop)])
        logits = logits.mean(dim='crop')
        return logits.rename(None)

    def load_moco_checkpoint(self, checkpoint_path: str):
        cp = torch.load(checkpoint_path, map_location=self.device)
        logger.info('Loading MoCo checkpoint from %s (epoch %d)', checkpoint_path, cp['epoch'])
        moco_state = cp['model']
        prefix = 'encoder_q.encoder.'

        blacklist = ['fc', 'linear', 'head', 'new_fc']

        def filter(k):
            return k.startswith(prefix) and not any(k.startswith(f'{prefix}{fc}') for fc in blacklist)

        model_state = {k[len(prefix):]: v for k, v in moco_state.items() if filter(k)}
        msg = self.model.module.load_state_dict(model_state, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} or \
               set(msg.missing_keys) == {"linear.weight", "linear.bias"} or \
               set(msg.missing_keys) == {'head.projection.weight', 'head.projection.bias'} or \
               set(msg.missing_keys) == {'new_fc.weight', 'new_fc.bias'}, \
            msg

    @torch.no_grad()
    def extract_features_train(self):
        self.model.eval()
        
        data_iter = iter(self.train_loader)

        for i, ((clip,), target) in tqdm(enumerate(data_iter)):
            # torch.Size([2, 3, 160, 112, 112])
            clip = self.reshape_clip(clip) # torch.Size([20, 3, 16, 112, 112])
            output = self.model.module.get_feature(clip) # torch.Size([20, 512, 1, 4, 4])
            output = nn.AdaptiveAvgPool3d((1, 1, 1))(output).squeeze() # torch.Size([20, 512])
            output = self.average_clips(output) # torch.Size([2, 512])
            self.train_feat_list.append(output.cpu().numpy().tolist())
            self.train_label_list.append(target.cpu().numpy().tolist())

    @torch.no_grad()
    def extract_features_test(self):
        self.model.eval()

        data_iter = iter(self.val_loader)

        for i, ((clip,), target) in tqdm(enumerate(data_iter)):
            clip = self.reshape_clip(clip)
            output = self.model.module.get_feature(clip)
            output = nn.AdaptiveAvgPool3d((1, 1, 1))(output).squeeze()
            output = self.average_clips(output)
            self.test_feat_list.append(output.cpu().numpy().tolist())
            self.test_label_list.append(target.cpu().numpy().tolist())


    def save_features(self, save_dir:str):
        os.makedirs(save_dir, exist_ok=True)
        fold = self.cfg.get_int('dataset.fold')
        logger.info(f'Saving features for train and test splits in {save_dir}...')

        np.save(P.join(save_dir, f'train_fold{fold}_feats.npy'), np.concatenate(self.train_feat_list))
        np.save(P.join(save_dir, f'train_fold{fold}_labels.npy'), np.concatenate(self.train_label_list))
        np.save(P.join(save_dir, f'test_fold{fold}_feats.npy'), np.concatenate(self.test_feat_list))
        np.save(P.join(save_dir, f'test_fold{fold}_labels.npy'), np.concatenate(self.test_label_list))

        logger.info(f'Saving features done.')

    def run(self, feat_dir):
        self.extract_features_train()
        self.extract_features_test()
        # optional
        self.save_features(feat_dir)


def topk_retrieval(feature_dir, cfg):
    """Extract features from test split and search on train split features."""
    logger.info('Loading local .npy files...')
    fold = cfg.get_int('dataset.fold')

    X_train = np.load(os.path.join(feature_dir, f'train_fold{fold}_feats.npy'))
    y_train = np.load(os.path.join(feature_dir, f'train_fold{fold}_labels.npy'))

    X_test = np.load(os.path.join(feature_dir, f'test_fold{fold}_feats.npy'))
    y_test = np.load(os.path.join(feature_dir, f'test_fold{fold}_labels.npy'))

    ks = [1, 5, 10, 20, 50]
    topk_correct = {k:0 for k in ks}

    distances = cosine_distances(X_test, X_train)
    indices = np.argsort(distances)

    for k in ks:
        # print(k)
        top_k_indices = indices[:, :k]
        # print(top_k_indices.shape, y_test.shape)
        for ind, test_label in zip(top_k_indices, y_test):
            labels = y_train[ind]
            if test_label in labels:
                # print(test_label, labels)
                topk_correct[k] += 1

    for k in ks:
        correct = topk_correct[k]
        total = len(X_test)
        logger.info('Top-{}, correct = {:.2f}, total = {}, acc = {:.3f}'.format(k, correct, total, correct/total))

    with open(os.path.join(feature_dir, f'topk_correct_fold{fold}.json'), 'w') as fp:
        json.dump(topk_correct, fp)


def main_worker(local_rank: int, args: Args, dist_url: str):
    FEATS_SAVE_DIR = P.join(args.run_dir, '../feature')
    # args = Args.from_args()
    print('Local Rank:', local_rank)

    # loggin in main process
    if local_rank == 0:
        set_logging_basic_config(args)

    logger.info(f'Args = \n{args}')

    if args.config is not None and args.experiment_dir is not None:
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
            engine = Engine(args, cfg, local_rank, True)
            engine.load_moco_checkpoint(args.moco_checkpoint)
            # extract and save features for train and test splits
            engine.run(FEATS_SAVE_DIR)
            # conduct retrieval
            topk_retrieval(FEATS_SAVE_DIR, cfg)
    else:
        logger.warning('No config. Do nothing.')


def main():
    args = Args.from_args()

    if args.seed is not None:
        utils.reproduction.initialize_seed(args.seed)

    args.resolve_continue()
    args.make_run_dir()
    args.save()
    # pack_code(args.run_dir)

    utils.environment.ulimit_n_max()

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'

    print(f'world_size={args.world_size} Using dist_url={dist_url}')

    args.parser = None
    mp.spawn(main_worker, args=(args, dist_url,), nprocs=args.world_size)


if __name__ == '__main__':
    main()