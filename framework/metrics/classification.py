import torch
from torch import Tensor
from typing import *


def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> List[Tensor]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float)
            res.append(correct_k * (100.0 / batch_size))
        return res




@torch.jit.script
def top5_accuracy(output: Tensor, target: Tensor) -> List[Tensor]:
    topk = (1, 5)
    maxk = 5
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target[None])

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float)
        res.append(correct_k * (100.0 / batch_size))
    return res


def binary_accuracy(output: Tensor, target: Tensor) -> Tensor:
    batch_size = target.shape[0]
    pred = output > 0.5
    correct = pred.eq(target).sum()
    return correct * (100.0 / batch_size)
