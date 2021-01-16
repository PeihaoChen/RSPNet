import logging
import random
from bisect import bisect_left
from dataclasses import dataclass
from . import functional_temporal as F

import numpy as np

logger = logging.getLogger(__name__)


def calc_needed_frames(size, stride):
    return (size - 1) * stride + 1


def fallback_select(size, stride, num_frames):
    assert num_frames > 0, 'No frames in video'
    if num_frames <= size:
        return np.arange(size) % num_frames
    if num_frames < calc_needed_frames(size, stride):
        return np.linspace(0, num_frames - 1, num=size).round().astype(int)
    return None


class RandomStrideCrop:
    def __init__(self, size: int, strides=({'stride': 1, 'weight': 1},)):
        self.size = size
        self.strides = strides

        weight_sum = sum(s['weight'] for s in strides)
        prefix_sum = 0
        self.prefix_weight_sum = []
        for s in strides:
            s['weight'] /= weight_sum
            prefix_sum += s['weight']
            self.prefix_weight_sum.append(prefix_sum)

    def __call__(self, frame_indices: np.ndarray):
        num_frames = len(frame_indices)
        rand = random.random()
        idx = bisect_left(self.prefix_weight_sum, rand)
        stride = self.strides[idx]['stride']

        selected = fallback_select(self.size, stride, num_frames)
        if selected is None:
            needed_frames = calc_needed_frames(self.size, stride)
            start_index = random.randint(0, num_frames - needed_frames)
            selected = np.arange(start_index, start_index + needed_frames, stride)

        return frame_indices[selected]

    def set_strides(self, strides=({'stride': 1, 'weight': 1},)):
        self.strides = strides

        weight_sum = sum(s['weight'] for s in strides)
        prefix_sum = 0
        self.prefix_weight_sum = []
        for s in strides:
            s['weight'] /= weight_sum
            prefix_sum += s['weight']
            self.prefix_weight_sum.append(prefix_sum)

    def set_size(self, size: int):
        self.size = size


class EvenNCrop:

    def __init__(self, size: int, stride=1, n=1):
        self.size = size
        self.stride = stride
        self.n = n

    def __call__(self, frame_indices: np.ndarray):
        num_frames = len(frame_indices)
        selected = fallback_select(self.size, self.stride, num_frames)
        if selected is not None:
            selected = np.tile(selected, self.n)
        else:
            needed_frames = calc_needed_frames(self.size, self.stride)
            if self.n == 1:
                start_index = (num_frames - needed_frames) // 2
                selected = np.arange(start_index, start_index + needed_frames, self.stride)
            else:
                start_index = np.linspace(0, num_frames - needed_frames, num=self.n).round().astype(int)
                offset = np.arange(0, 0 + needed_frames, self.stride)
                selected = start_index[:, np.newaxis] + offset
                selected = selected.flat

        return frame_indices[selected]


class Resample:

    def __init__(self, target_fps: float = 30.):
        self.target_fps = target_fps
        if target_fps is not None:
            logger.info('Resample to %f FPS', target_fps)

    def __call__(self, frame_indices, source_fps: float):
        # return F.resample(frame_indices, source_fps, self.target_fps)
        return F.resample_video_idx(frame_indices, source_fps, self.target_fps)


class LimitRange:
    def __init__(self, min_frames: int, limit_rate=1):
        self.min_frames = min_frames
        self.limit_rate = limit_rate
        if limit_rate < 1:
            logger.info('Limit clips in %.1f%% of video length', limit_rate * 100)

    def __call__(self, frame_indices):
        if len(frame_indices) <= self.min_frames:
            return frame_indices

        target_length = (len(frame_indices) - self.min_frames) * self.limit_rate + self.min_frames
        target_length = int(round(target_length))
        start = random.randint(0, len(frame_indices) - target_length)
        selected = np.arange(start, start + target_length)
        return frame_indices[selected]


class RandomStrideTwoCrop:
    def __init__(self, size: int, strides=({'stride': 1, 'weight': 1},)):
        self.size = size
        self.strides = strides
        self.total_size = size * 2

        weight_sum = sum(s['weight'] for s in strides)
        prefix_sum = 0
        self.prefix_weight_sum = []
        for s in strides:
            s['weight'] /= weight_sum
            prefix_sum += s['weight']
            self.prefix_weight_sum.append(prefix_sum)

    def __call__(self, frame_indices: np.ndarray):
        num_frames = len(frame_indices)
        rand = random.random()
        idx = bisect_left(self.prefix_weight_sum, rand)
        stride = self.strides[idx]['stride']

        selected = fallback_select(self.total_size, stride, num_frames)
        if selected is None:
            needed_frames = calc_needed_frames(self.total_size, stride)
            start_index = random.randint(0, num_frames - needed_frames)
            selected = np.arange(start_index, start_index + needed_frames, stride)

        return frame_indices[selected]


class Cover:
    def __init__(self, size: int, n_crop=None):
        '''
        n_crop: `None` means random offset. used in train
        '''
        self.size = size
        self.n_crop = n_crop

    def __call__(self, frame_indices: np.ndarray):
        selected = fallback_select(self.size, 1, len(frame_indices))
        if selected is not None:
            if self.n_crop is not None:
                selected = np.tile(selected, self.n_crop)
        else:
            stride = len(frame_indices) / self.size

            def select(offset):
                selected = np.arange(self.size) * stride + offset
                selected = np.floor(selected).astype(int)
                selected = np.minimum(selected, len(frame_indices) - 1)  # protect against overflow
                return selected

            if self.n_crop is None:
                offset = [random.uniform(0, stride)]
            elif self.n_crop == 1:
                offset = [0.5 * stride]
            else:
                offset = np.linspace(0, stride, num=self.n_crop, endpoint=False)

            selected = np.concatenate([select(o) for o in offset])

        return frame_indices[selected]
