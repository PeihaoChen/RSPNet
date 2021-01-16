local hmdb51 = import '../dataset/hmdb51.libsonnet';
local c3d = import '../model/c3d.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: hmdb51,
    model: c3d,
    model_type: 'multitask',
    temporal_transforms+: {
        size: 16,
        validate: {
                    stride: 1,
                    n_crop: 1,
                    final_n_crop: 10,
                },
    },
    batch_size: 6,
    validate: {
        batch_size: 6,
    },
    final_validate: {
        batch_size: 6,
    },
    optimizer+: {
        lr: 0.005,
        milestones: [50, 70, 90],
        schedule: "multi_step",
        },
    num_epochs: 100,
}
