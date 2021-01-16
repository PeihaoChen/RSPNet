import math
import random
import warnings

import torch
from torchvision.transforms import Compose, RandomApply
from torchvision.transforms._transforms_video import \
    NormalizeVideo as Normalize
from torchvision.transforms._transforms_video import \
    RandomHorizontalFlipVideo as RandomHorizontalFlip
from torchvision.transforms._transforms_video import ToTensorVideo as ToTensor

from .transforms_tensor import ColorJitter, RandomGrayScale


class Resize:
    def __init__(self, size, interpolation_mode='bilinear'):
        self.size = size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        return torch.nn.functional.interpolate(
            clip, size=self.size, mode=self.interpolation_mode,
            align_corners=False,  # Suppress warning
        )


class RawVideoCrop:
    def get_params(self, clip):
        raise NotImplementedError

    def get_size(self, clip):
        height, width, _ = clip.size()[-3:]
        return height, width

    def __call__(self, clip: torch.Tensor):
        i, j, h, w = self.get_params(clip)
        region = clip[..., i: i + h, j: j + w, :]
        return region.contiguous()


class RawVideoRandomCrop(RawVideoCrop):
    def __init__(self, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio

    def get_params(self, clip):
        # Mostly copied from torchvision.transforms_video.RandomResizedCrop
        height, width = self.get_size(clip)
        area = height * width
        ratio = self.ratio
        scale = self.scale

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


class RawVideoCenterMaxCrop(RawVideoCrop):
    def __init__(self, ratio=1.):
        self.ratio = ratio

    def get_params(self, clip):
        height, width = self.get_size(clip)
        if width / height > self.ratio:
            h = height
            w = int(round(h * self.ratio))
        else:
            w = width
            h = int(round(w / self.ratio))
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
