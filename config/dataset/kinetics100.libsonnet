local normalization = import "normalization.libsonnet";

{
    name: 'kinetics100',
    root: 'data/kinetics100',
    num_classes: 100,
    blacklist: [
        'train_video/eating_carrots/eiZ8Hzc7FPU_000080_000090.mp4',
        'train_video/playing_flute/co50KUHacYw_000005_000015.mp4',
        'train_video/sweeping_floor/EuGXJiVQwCg_000005_000015.mp4',
        'train_video/making_tea/mtYFNsRcxY4_000063_000073.mp4',
        'train_video/building_cabinet/jQPSzhKkk-g_000028_000038.mp4',
        'val_video/skipping_rope/sAA809R_u1E_000077_000087.mp4',
    ],
    normalization:: normalization.imagenet,
    mean: self.normalization.mean,
    std: self.normalization.std,
}