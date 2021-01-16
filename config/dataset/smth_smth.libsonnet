local normalization = import "normalization.libsonnet";

{
    name: 'smth_smth',
    root: 'data/smth-smth-v2/20bn-something-something-v2',
    annotation_path: 'data/smth-smth-v2/annotations',
    fold: 1,
    num_classes: 174,

    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}