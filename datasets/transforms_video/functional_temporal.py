import math
from typing import Optional

import numpy as np


def resample_video_idx(
        frame_indices: np.ndarray,
        original_fps: Optional[float],
        new_fps: Optional[float]
) -> np.ndarray:
    """
    https://github.com/pytorch/vision/blob/70ed29d00188a1b29c172a1859e6296fbe62005c/torchvision/datasets/video_utils.py#L273
    """
    if original_fps is None or new_fps is None:
        return frame_indices

    step: float = original_fps / new_fps
    if step.is_integer():
        step = int(step)
        # optimization: if step is integer, don't need to perform
        # advanced indexing
        return frame_indices[::step]

    new_num_frames = int(math.floor(len(frame_indices) / step))
    idxs = np.arange(new_num_frames) * step
    idxs = np.floor(idxs).astype(np.int)
    return frame_indices[idxs]
