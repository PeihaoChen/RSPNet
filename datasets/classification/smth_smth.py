import json
from pathlib import Path
from typing import Dict

from torch.utils.data import Dataset

from .video import Sample, VideoDataset


class Smth_Smth(Dataset):
    '''somthing-something samples

    Sample 1 clip per video
    '''

    def __init__(
            self,
            video_dir,
            annotation_dir,
            split='train',
    ):
        self.annotation_dir = Path(annotation_dir)
        self.class_idx_dict = self.read_class_idx(self.annotation_dir)

        list_name = {
            'train': 'train',
            'val': 'validation',
            'test': 'validation',
        }[split]
        video_list_path = self.annotation_dir / f'something-something-v2-{list_name}.json'
        self._samples = []
        with video_list_path.open('r') as f:
            video_infos = json.load(f)
            for video_info in video_infos:
                video = int(video_info['id'])
                video_path = Path(video_dir) / f'{video}.mp4'
                class_name = video_info['template'].replace('[', '').replace(']', '')
                class_index = int(self.class_idx_dict[class_name])

                s = Sample(
                    video_path=video_path,
                    class_index=class_index,
                )
                self._samples.append(s)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index: int):
        return self._samples[index]

    @staticmethod
    def read_class_idx(annotation_dir: Path) -> Dict[str, str]:
        class_ind_path = annotation_dir / 'something-something-v2-labels.json'
        with class_ind_path.open('r') as f:
            class_dict = json.load(f)
        return class_dict
