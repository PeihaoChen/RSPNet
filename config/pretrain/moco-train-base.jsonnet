local sgd = import "../optimizer/sgd.libsonnet";
local kinetics400 = import "../dataset/kinetics400.libsonnet";
local kinetics100 = import "../dataset/kinetics100.libsonnet";
local loss_lambda = import "../optimizer/loss_lambda.libsonnet";

{
    arch: 'resnet18',

    model: {
        arch: $.arch,
    },

    dataset: kinetics400, // or kinetics100

    batch_size: 64,
    num_workers: 4,

    num_epochs: '200',

    optimizer: sgd,
    loss_lambda: loss_lambda,
    log_interval: 10,
    opt_level: 'O0',

    checkpoint_interval: 50,

    moco: {
        dim: 128,
        k: 16384,
        m: 0.999,
        t: 0.07,
        mlp: false,
        diff_speed: [2], // Avalable choices: [2] (2x speed)，[4] (4x speed), [4,2,1] (randomly choose a speed)，[] (not enabled)
        aug_plus: false,
        fc_type: 'linear', // Avalable choices: linear, mlp, conv
    },

    spatial_transforms: {
        size: 112,
    },
    temporal_transforms: {
        _size:: 16,
        size: if std.length($.moco.diff_speed) == 0 then self._size else $.moco.diff_speed[0] * self._size,
        strides: [
            {stride: 1, weight: 1},
        ], 
        frame_rate: null, // Follow the origin video fps if not set. Use fixed fps if set.
        random_crop: true,
    },
}
