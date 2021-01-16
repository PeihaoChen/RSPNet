local normalization = import "normalization.libsonnet";

{
    name: 'oops_moco',
    root: 'data/oops/oops_video_256/train',
    blacklist: [
        'FailArmy Presents - People are Awesome _ Epic Wins Compilation46.mp4',
    ],

    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}
