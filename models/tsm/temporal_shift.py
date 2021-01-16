# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False, groups=1):
        """
        Use inplace by default, maybe buggy
        :param net:
        :param n_segment:
        :param n_div:
        :param inplace:
        """
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.groups = groups
        self.inplace = inplace
        if inplace:
            logger.info('=> Using in-place shift...')
        logger.info('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        if self.groups == 1:
            x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        else:
            x = self.shift_group(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace, groups=self.groups)

        # x = self.fast_shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)

    @staticmethod
    def shift_group(x, n_segment, fold_div=3, inplace=False, groups=2):
        assert groups != 1, "shift_group method is not for groups == {}!".format(groups)
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)
        assert c % groups == 0, "{} % {} != 0".format(c, groups)
        groups_c = c // groups
        out_list = []
        # out = torch.zeros_like(x)
        for ic in range(groups):
            fold = groups_c // fold_div
            if inplace:
                # Due to some out of order error when performing parallel computing.
                # May need to write a CUDA kernel.
                raise NotImplementedError
                # out = InplaceShift.apply(x, fold)
            else:
                # way 2
                # base_fold = groups_c * ic
                # # print(base_fold)
                # # shift left
                # out[:, :-1, base_fold:base_fold + fold] = x[:, 1:, base_fold:base_fold + fold]
                # # shift right=
                # out[:, 1:, base_fold + fold: base_fold + (2 * fold)] = x[:, :-1,
                #                                                        base_fold + fold: base_fold + (2 * fold)]
                # # not shift
                # out[:, :, base_fold + (2 * fold):(ic + 1) * groups_c] = x[:, :,
                #                                                         base_fold + (2 * fold):(ic + 1) * groups_c]
                # # print(base_fold + fold, base_fold + (2 * fold), (ic + 1) * groups_c, out.shape)

                # way 1
                out = torch.zeros_like(x[:, :, :groups_c])
                base_fold = groups_c * ic
                out[:, :-1, :fold] = x[:, 1:, base_fold:base_fold + fold]  # shift left
                out[:, 1:, fold: (2 * fold)] = x[:, :-1, base_fold + fold: base_fold + (2 * fold)]  # shift right

                out[:, :, (2 * fold):] = x[:, :, base_fold + (2 * fold):(ic + 1) * groups_c]  # not shift
                out_list.append(out)
        out = torch.cat(out_list, dim=2)

        return out.view(nt, c, h, w)

    @staticmethod
    @torch.jit.script
    def fast_shift(x: torch.Tensor, n_segment: int, fold_div: int = 3):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div

        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None


class TemporalPool(nn.Module):
    def __init__(self, net, n_segment):
        super(TemporalPool, self).__init__()
        self.net = net
        self.n_segment = n_segment

    def forward(self, x):
        x = self.temporal_pool(x, n_segment=self.n_segment)
        return self.net(x)

    @staticmethod
    def temporal_pool(x, n_segment):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
        x = F.max_pool3d(x, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        x = x.transpose(1, 2).contiguous().view(nt // 2, c, h, w)
        return x


def make_temporal_shift(net, n_segment, n_div=8, place='blockres', temporal_pool=False):
    if temporal_pool:
        n_segment_list = [n_segment, n_segment // 2, n_segment // 2, n_segment // 2]
    else:
        n_segment_list = [n_segment] * 4
    assert n_segment_list[-1] > 0
    logger.info('=> n_segment per stage: {}'.format(n_segment_list))

    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        if place == 'block':
            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                logger.info('=> Processing stage with {} blocks'.format(len(blocks)))
                if blocks[0].conv1.groups != 1:
                    logger.info("=> init all block in this stage with groups!!!!!")
                for i, b in enumerate(blocks):
                    blocks[i] = TemporalShift(b, n_segment=this_segment, n_div=n_div, groups=b.conv1.groups)
                return nn.Sequential(*(blocks))

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])

        elif 'blockres' in place:
            n_round = 1
            if len(list(net.layer3.children())) >= 23:
                n_round = 2
                logger.info('=> Using n_round {} to insert temporal shift'.format(n_round))

            def make_block_temporal(stage, this_segment):
                blocks = list(stage.children())
                logger.info('=> Processing stage with {} blocks residual'.format(len(blocks)))
                for i, b in enumerate(blocks):
                    if i % n_round == 0:
                        if not isinstance(blocks[i], nn.Conv2d):  # FIXME
                            blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div,
                                                            groups=b.conv1.groups)
                return nn.Sequential(*blocks)

            net.layer1 = make_block_temporal(net.layer1, n_segment_list[0])
            net.layer2 = make_block_temporal(net.layer2, n_segment_list[1])
            net.layer3 = make_block_temporal(net.layer3, n_segment_list[2])
            net.layer4 = make_block_temporal(net.layer4, n_segment_list[3])
    else:
        raise NotImplementedError(place)


def make_temporal_pool(net, n_segment):
    import torchvision
    if isinstance(net, torchvision.models.ResNet):
        logger.info('=> Injecting nonlocal pooling')
        net.layer2 = TemporalPool(net.layer2, n_segment)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # test inplace shift v.s. vanilla shift
    import time

    start = time.time()
    tsm1 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=False, groups=2)
    # tsm2 = TemporalShift(nn.Sequential(), n_segment=8, n_div=8, inplace=True)

    print('=> Testing CPU...')
    # test forward
    with torch.no_grad():
        for i in range(10):
            x = torch.rand(2 * 8, 56, 224, 224)
            y1 = tsm1(x)
            # y2 = tsm2(x)
            # assert torch.norm(y1 - y2).item() < 1e-5

    # test backward
    with torch.enable_grad():
        for i in range(10):
            x1 = torch.rand(2 * 8, 56, 224, 224)
            x1.requires_grad_()
            # x2 = x1.clone()
            y1 = tsm1(x1)
            # y2 = tsm2(x2)
            grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
            # grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
            # assert torch.norm(grad1 - grad2).item() < 1e-5
    print(time.time() - start)
    # print('=> Testing GPU...')
    # tsm1.cuda()
    # tsm2.cuda()
    # # test forward
    # with torch.no_grad():
    #     for i in range(10):
    #         x = torch.rand(2 * 8, 3, 224, 224).cuda()
    #         y1 = tsm1(x)
    #         y2 = tsm2(x)
    #         assert torch.norm(y1 - y2).item() < 1e-5
    #
    # # test backward
    # with torch.enable_grad():
    #     for i in range(10):
    #         x1 = torch.rand(2 * 8, 3, 224, 224).cuda()
    #         x1.requires_grad_()
    #         x2 = x1.clone()
    #         y1 = tsm1(x1)
    #         y2 = tsm2(x2)
    #         grad1 = torch.autograd.grad((y1 ** 2).mean(), [x1])[0]
    #         grad2 = torch.autograd.grad((y2 ** 2).mean(), [x2])[0]
    #         assert torch.norm(grad1 - grad2).item() < 1e-5
    # print('Test passed.')
