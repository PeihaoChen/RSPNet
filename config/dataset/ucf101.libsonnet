local normalization = import "normalization.libsonnet";

{
    name: 'ucf101',
    root: 'data/UCF101/UCF-101',
    annotation_path: 'data/UCF101/ucfTrainTestlist',
    fold: 1,
    num_classes: 101,

    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}