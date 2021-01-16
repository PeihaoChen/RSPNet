import torch
import torch.nn as nn
from torch import Tensor
import logging
from typing import *
import random

logger = logging.getLogger(__name__)


class MoCoDiffLoss(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
            self,
            base_encoder,
            dim=128,
            K=65536,
            m=0.999,
            T=0.07,
            mlp=False,
            diff_speed: Optional[List[int]] = None,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        self.diff_speed = diff_speed
        logger.warning('Using diffspeed: %s', self.diff_speed)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.alpha = 0.5

        assert self.diff_speed is not None, "This branch is for diff speed"

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _diff_speed(self, im_q: Tensor, im_k: Tensor):
        B, C, T, H, W = im_q.shape
        random_indices = torch.randperm(B, device=im_q.device)
        selected_s1 = random_indices[:int(B * self.alpha)]
        selected_s2 = random_indices[int(B * self.alpha):]

        diff_speed = random.choice(self.diff_speed)
        T_real = T // diff_speed
        speed1 = torch.arange(0, T, 1, device=im_q.device)[: T_real]
        speed2 = torch.arange(0, T, diff_speed, device=im_q.device)[: T_real]
        im_q_real = torch.empty(B, C, T_real, H, W, device=im_q.device)
        im_k_real = torch.empty_like(im_q_real)
        im_k_negative = torch.empty_like(im_q_real)

        im_q_real[selected_s1] = im_q.index_select(0, selected_s1).index_select(2, speed1)
        im_q_real[selected_s2] = im_q.index_select(0, selected_s2).index_select(2, speed2)

        im_k_real[selected_s1] = im_k.index_select(0, selected_s1).index_select(2, speed1)
        im_k_real[selected_s2] = im_k.index_select(0, selected_s2).index_select(2, speed2)

        im_k_negative[selected_s1] = im_k.index_select(0, selected_s1).index_select(2, speed2)
        im_k_negative[selected_s2] = im_k.index_select(0, selected_s2).index_select(2, speed1)

        k_negative = self._forward_encoder_k(im_k_negative)

        return im_q_real, im_k_real, k_negative

    @staticmethod
    def _shuffle_im_k(im_k: Tensor):
        batch_size = len(im_k)
        shuffle_idx = torch.randperm(batch_size, dtype=torch.long, device=im_k.device)
        im_k = im_k[shuffle_idx]
        return im_k

    @torch.no_grad()
    def _forward_encoder_k(self, im_k: Tensor):
        # shuffle for making use of BN
        im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

        k = self.encoder_k(im_k)  # keys: NxC
        k = nn.functional.normalize(k, dim=1)

        # undo shuffle
        k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        return k

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        reversed_k, speed_k = None, None
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.diff_speed is not None:
                """
                DO NOT try to use shuffle k in training
                """
                # im_k = self._shuffle_im_k(im_k)
                im_q, im_k, speed_k = self._diff_speed(im_q, im_k)  # Update im_q, im_k
            k = self._forward_encoder_k(im_k)

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        if self.diff_speed is not None:
            l_neg_speed = torch.einsum('nc,nc->n', [q, speed_k]).unsqueeze(-1)
            # l_neg = torch.cat([l_neg_speed, l_neg], dim=1)

        l_pos /= self.T
        l_neg /= self.T

        l_neg_speed /= self.T

        # logits: Nx(1+K)
        logits1 = torch.cat([l_pos, l_neg], dim=1)
        logits2 = torch.cat([l_neg_speed, l_neg], dim=1)
        ranking_logits = (l_pos, l_neg_speed)  # l_pos > l_neg_speed

        # apply temperature
        # logits1 /= self.T
        # logits2 /= self.T

        # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        labels = torch.zeros(logits1.shape[0], dtype=torch.long, device=logits1.device)
        ranking_target = torch.ones_like(labels)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        logits = (logits1, logits2)

        return logits, labels, ranking_logits, ranking_target
        # return (logits, ranking_logits), (labels, ranking_target)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class Loss(nn.Module):

    def __init__(self, margin=1.0, A: float = 1.0, M: float = 1.0):
        super().__init__()
        self.A = A
        self.M = M
        self._cross_entropy_loss = nn.CrossEntropyLoss()
        self._margin_ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(
            self,
            output: Tuple[Tensor, Tensor],
            target: Tensor,
            ranking_logits: Tuple[Tensor, Tensor],
            ranking_target: Tensor
    ):
        ce1 = self._cross_entropy_loss(output[0], target)
        ce2 = self._cross_entropy_loss(output[1], target)
        ranking = self._margin_ranking_loss(ranking_logits[0], ranking_logits[1], ranking_target)
        loss = self.A * (ce1 + ce2) + self.M * ranking
        return loss, ce1 + ce2, ranking


class MoCoDiffLossTwoFc(nn.Module):
    def __init__(
            self,
            base_encoder,
            dim=128,
            K=65536,
            m=0.999,
            T=0.07,
            mlp=False,
            diff_speed: Optional[List[int]] = None,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()

        self.K = K
        self.m = m
        self.T = T

        self.diff_speed = diff_speed
        logger.warning('Using diffspeed: %s', self.diff_speed)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.alpha = 0.5

        assert self.diff_speed is not None, "This branch is for diff speed"

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _forward_encoder_k(self, im_k: Tensor):
        # shuffle for making use of BN
        im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

        k_A, k_M = self.encoder_k(im_k)  # keys: NxC

        # undo shuffle
        k_A = self._batch_unshuffle_ddp(k_A, idx_unshuffle)
        k_M = self._batch_unshuffle_ddp(k_M, idx_unshuffle)

        return k_A, k_M

    @torch.no_grad()
    def _diff_speed(self, im_q: Tensor, im_k: Tensor):
        B, C, T, H, W = im_q.shape
        random_indices = torch.randperm(B, device=im_q.device)
        selected_s1 = random_indices[:int(B * self.alpha)]
        selected_s2 = random_indices[int(B * self.alpha):]

        diff_speed = random.choice(self.diff_speed)
        T_real = T // diff_speed
        speed1 = torch.arange(0, T, 1, device=im_q.device)[: T_real]            # speed1 is normal speed
        speed2 = torch.arange(0, T, diff_speed, device=im_q.device)[: T_real]   # speed2 is randomly selected from self.diff_speed
        im_q_real = torch.empty(B, C, T_real, H, W, device=im_q.device)
        im_k_real = torch.empty_like(im_q_real)
        im_k_negative = torch.empty_like(im_q_real)

        im_q_real[selected_s1] = im_q.index_select(0, selected_s1).index_select(2, speed1)
        im_q_real[selected_s2] = im_q.index_select(0, selected_s2).index_select(2, speed2)

        im_k_real[selected_s1] = im_k.index_select(0, selected_s1).index_select(2, speed1)
        im_k_real[selected_s2] = im_k.index_select(0, selected_s2).index_select(2, speed2)

        im_k_negative[selected_s1] = im_k.index_select(0, selected_s1).index_select(2, speed2)
        im_k_negative[selected_s2] = im_k.index_select(0, selected_s2).index_select(2, speed1)

        k_negative_A, k_negative_M = self._forward_encoder_k(im_k_negative)

        return im_q_real, im_k_real, k_negative_A, k_negative_M

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets

        Variable:
            q_A, q_M: query for A-VID and RSP tasks
            k_A, k_M: positive key for A-VID and RSP tasks
            k_neg_A, k_neg_M: negative key for A-VID and RSP tasks
        """
        reversed_k, speed_k = None, None
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            if self.diff_speed is not None:
                im_q, im_k, k_neg_A, k_neg_M = self._diff_speed(im_q, im_k)  # Update im_q, im_k
            k_A, k_M = self._forward_encoder_k(im_k)

        # compute query features
        q_A, q_M = self.encoder_q(im_q)  # queries: NxC

        # compute logits
        # Einstein sum is more intuitive
        # For A-VID task, we consider the clips in the same videos as positive key even they are sampled in different speeds
        # positive logits: Nx1
        l_pos_A1 = torch.einsum('nc,nc->n', [q_A, k_A]).unsqueeze(-1)
        l_pos_A2 = torch.einsum('nc,nc->n', [q_A, k_neg_A]).unsqueeze(-1)
        l_pos_M = torch.einsum('nc,nc->n', [q_M, k_M]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_A = torch.einsum('nc,ck->nk', [q_A, self.queue.clone().detach()])
        l_neg_M = torch.einsum('nc,nc->n', [q_M, k_neg_M]).unsqueeze(-1)

        l_pos_A1 /= self.T
        l_pos_A2 /= self.T
        l_neg_A /= self.T
        l_pos_M /= self.T
        l_neg_M /= self.T

        # logits: Nx(1+K)
        logits1 = torch.cat([l_pos_A1, l_neg_A], dim=1)
        logits2 = torch.cat([l_pos_A2, l_neg_A], dim=1)
        logits_A = (logits1, logits2)
        logits_M = (l_pos_M, l_neg_M)  # l_pos > l_neg_speed

        # labels: positive key indicators
        labels_A = torch.zeros(logits1.shape[0], dtype=torch.long, device=logits1.device)
        labels_M = torch.ones_like(labels_A)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_neg_A)

        return logits_A, labels_A, logits_M, labels_M