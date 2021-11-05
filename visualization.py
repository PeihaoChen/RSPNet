import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed
from arguments import Args
from pyhocon import ConfigTree
from moco import ModelFactory
from datasets.classification import DataLoaderFactoryV3
from tqdm import tqdm
import cv2
import numpy as np
import os
import os.path as P
import copy

from framework import utils
from framework.config import get_config
from utils.moco import replace_moco_k_in_config


class Engine:

    def __init__(self, args: Args, cfg: ConfigTree, local_rank: int):
        self.args = args
        self.cfg = cfg
        self.local_rank = local_rank

        self.opt_level = self.cfg.get_string('opt_level')

        self.model_factory = ModelFactory(cfg)
        self.data_loader_factory = DataLoaderFactoryV3(cfg)

        self.device = torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        self.model = self.model_factory.build_moco_diffloss()
        self.val_loader = self.data_loader_factory.build(vid=True, split='train', device=self.device, visualization=True)
        self.arch = cfg.get_string('arch')

    def _load_ckpt_file(self, checkpoint_path):
        states = torch.load(checkpoint_path, map_location=self.device)
        if states['arch'] != self.arch:
            raise ValueError(f'Loading checkpoint arch {states["arch"]} does not match current arch {self.arch}')
        return states

    def load_model(self, checkpoint_path):
        states = self._load_ckpt_file(checkpoint_path)
        msg = self.model.module.load_state_dict(states['model'])
        print(f'Missing keys: {msg.missing_keys}, Unexpected keys: {msg.unexpected_keys}')

    def cam_rgbmask(self, cam_mat):
        max = cam_mat.max()
        min = cam_mat.min()
        cam_mat = (cam_mat - min) / (max - min) 
        cam_mat *= 255
        cam_mat = cam_mat.astype(np.uint8)
        cam_mask = cv2.applyColorMap(cam_mat, cv2.COLORMAP_JET)
        cam_mask = cv2.resize(cam_mask, (224,224))
        return cam_mask

    def mask_clip(self, clip : torch.Tensor, cam_mask, rnd_idx=0):
        clip = clip[0].permute(1,2,3,0)
        
        frame : np.ndarray = clip[rnd_idx].cpu().numpy()
        frame = copy.deepcopy(frame)
        frame = np.multiply(frame, [0.485, 0.456, 0.406])
        frame += [0.229, 0.224, 0.22]
        frame *= 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masked_img = cv2.addWeighted(frame, 0.6, cam_mask, 0.4, 0)
        #cv2.imwrite(f'results/iter-{iter}-frame-{rnd_idx}-{discrip}.png', masked_img)
        return masked_img

    def save_fig(self, mskimgs, iter, prefix='RSP', local_rank=0):
        w,h,c = mskimgs[0].shape

        background = np.ones(((w+40), (h*2+30), c), dtype=np.uint8) * 255
        background[10:10+h, 10:10+w, :] = mskimgs[0]
        background[10:10+h, 20+w:20+w+w, :] = mskimgs[1]
        cv2.putText(background, f'query for {prefix}', (10+w//4, 10+h+15), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0,0,0))
        cv2.putText(background, f'key for {prefix}', (20+w+w//4, 10+h+15), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0,0,0))
        cv2.imwrite(P.join(self.args.experiment_dir, f'iter-{iter}-{prefix}-{local_rank}.png'), background)

    def visual_epoch(self):
        iter_data = tqdm(self.val_loader, desc='Current Epoch', disable=self.local_rank != 0, dynamic_ncols=True)
        for i, ((clip_q, clip_k), *_) in enumerate(iter_data):
            # if self.local_rank == 0:
            #     torch.save((clip_q, clip_k), self.args.experiment_dir / 'input.pth')
            Ms_qA, Ms_qM, Ms_kA, Ms_kM = self.model.module.cam_visualize(clip_q, clip_k)
            cam_mat_qA : np.ndarray = Ms_qA[0].mean(0).cpu().numpy()
            cam_mat_qM : np.ndarray = Ms_qM[0].mean(0).cpu().numpy()
            cam_mat_kA : np.ndarray = Ms_kA[0].mean(0).cpu().numpy()
            cam_mat_kM : np.ndarray = Ms_kM[0].mean(0).cpu().numpy()

            cam_mask_qA = self.cam_rgbmask(cam_mat_qA)
            cam_mask_qM = self.cam_rgbmask(cam_mat_qM)
            cam_mask_kA = self.cam_rgbmask(cam_mat_kA)
            cam_mask_kM = self.cam_rgbmask(cam_mat_kM)

            #rnd_idx = np.random.randint(0, clip_q.shape[1])
            rnd_idx = clip_q.shape[1] // 2

            masked_img_qA = self.mask_clip(clip_q, cam_mask_qA, rnd_idx=rnd_idx)
            masked_img_qM = self.mask_clip(clip_q, cam_mask_qM, rnd_idx=rnd_idx)
            masked_img_kA = self.mask_clip(clip_k, cam_mask_kA, rnd_idx=rnd_idx)
            masked_img_kM = self.mask_clip(clip_k, cam_mask_kM, rnd_idx=rnd_idx)

            self.save_fig((masked_img_qA, masked_img_kA), iter=i, prefix='RSP', local_rank=self.local_rank)
            self.save_fig((masked_img_qM, masked_img_kM), iter=i, prefix='AVID', local_rank=self.local_rank)

    def run(self):
        self.model.eval()
        self.visual_epoch()


def main_worker(local_rank: int, args: Args, dist_url: str):
    print('Local Rank:', local_rank)

    if args.seed is not None:
        utils.reproduction.initialize_seed(args.seed + local_rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method=dist_url,
        world_size=args.world_size,
        rank=local_rank,
    )

    utils.reproduction.cudnn_benchmark()

    cfg = get_config(args)

    replace_moco_k_in_config(cfg)

    engine = Engine(args, cfg, local_rank=local_rank)
    if args.load_model is not None:
        engine.load_model(args.load_model)

    engine.run()


def main():
    args = Args.from_args()
    if args.seed is not None:
        utils.reproduction.initialize_seed(args.seed)

    utils.environment.ulimit_n_max()

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    # Run on main process to avoid conflict
    args.resolve_continue()

    free_port = utils.distributed.find_free_port()
    dist_url = f'tcp://127.0.0.1:{free_port}'

    print(f'world_size={args.world_size} Using dist_url={dist_url}')

    args.parser = None
    # Only single node distributed training is supported
    mp.spawn(main_worker, args=(args, dist_url,), nprocs=args.world_size)


if __name__ == '__main__':
    main()
