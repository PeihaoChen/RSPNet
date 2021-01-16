import torch
import torch.distributed

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', device=torch.device('cpu')):
        self.name = name
        self.fmt = fmt

        self.val = torch.tensor(0, dtype=torch.float, device=device)
        self.sum = torch.tensor(0, dtype=torch.float, device=device)
        self.count = torch.tensor(0, dtype=torch.int, device=device)

        self.reset()

    def reset(self):
        self.val = torch.tensor(0, dtype=torch.float, device=self.val.device)
        self.sum = torch.tensor(0, dtype=torch.float, device=self.sum.device)
        self.count = torch.tensor(0, dtype=torch.int, device=self.count.device)

    @torch.no_grad()
    def update(self, val: torch.Tensor, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(
            name=self.name,
            val=self.val.item(),
            avg=self.avg.item(),
        )

    def sync_distributed(self):
        r_count = torch.distributed.all_reduce(self.count, op=torch.distributed.ReduceOp.SUM, async_op=True)
        r_sum = torch.distributed.all_reduce(self.sum, op=torch.distributed.ReduceOp.SUM, async_op=True)
        r_count.wait()
        r_sum.wait()
