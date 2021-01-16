import glob
import os

from torchvision.datasets.folder import make_dataset

from .video import Sample


class HMDB51:
    '''UCF101 samples

    Sample 1 clip per video
    '''
    data_url = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
    splits = {
        "url": "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar",
        "md5": "15e67781e70dcfbdce2d7dbb9b3344b5"
    }

    # TRAIN_TAG = 1
    # TEST_TAG = 2

    def __init__(
            self,
            video_dir,
            annotation_dir,
            fold=1,
            split='train',
            extensions=('.avi', '.mp4'),
    ):
        classes = sorted(list(filter(lambda p: os.path.isdir(os.path.join(video_dir, p)), os.listdir(video_dir))))
        self.class_to_index = {class_: i for (i, class_) in enumerate(classes)}

        list_name = {
            'train': 1,
            'val': 2,
            'test': 2,
        }[split]

        video_list = [path for (path, _) in make_dataset(video_dir, self.class_to_index, extensions, )]
        video_list_path = self._select_fold(video_list, annotation_dir, fold, list_name)

        self._samples = []
        for video_path in video_list_path:
            # Not using class index in list file, because test split don't have it.
            class_name = video_path.split(os.sep)[-2]
            class_index = self.class_to_index[class_name]
            s = Sample(
                video_path=video_path,
                class_index=class_index,
            )
            self._samples.append(s)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index: int):
        return self._samples[index]

    def _select_fold(self, video_list, annotations_dir, fold, list_name):
        target_tag = list_name
        split_pattern_name = "*test_split{}.txt".format(fold)
        split_pattern_path = os.path.join(annotations_dir, split_pattern_name)
        annotation_paths = glob.glob(split_pattern_path)
        selected_files = []
        for filepath in annotation_paths:
            with open(filepath) as fid:
                lines = fid.readlines()
            for line in lines:
                video_filename, tag_string = line.split()
                tag = int(tag_string)
                if tag == target_tag:
                    selected_files.append(video_filename)
        selected_files = set(selected_files)

        video_path_list = []
        for video_index, video_path in enumerate(video_list):
            if os.path.basename(video_path) in selected_files:
                video_path_list.append(video_path)

        return video_path_list
