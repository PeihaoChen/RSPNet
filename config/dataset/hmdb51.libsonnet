local normalization = import "normalization.libsonnet";

{
    name: 'hmdb51',
    root: 'data/hmdb51/videos',
    annotation_path: 'data/hmdb51/metafile',
    fold: 1,
    num_classes: 51,

    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}