local normalization = import "normalization.libsonnet";

{
    name: 'kinetics400',
    root: 'data/kinetics400',
    num_classes: 400,
    blacklist: [
        'train_video/playing_monopoly/NLL667uPWVA.mp4',
    ],
    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}