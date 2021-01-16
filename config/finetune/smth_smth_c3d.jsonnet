local smth_smth = import '../dataset/smth_smth.libsonnet';
local c3d = import '../model/c3d.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: smth_smth,
    model: c3d,
    model_type: 'multitask',
    temporal_transforms+: {
        size: 16,
        type: 'cover',
        validate+: {
            final_n_crop: 2,
        },
    },
    spatial_transforms+: {
        h_flip: 0,
    },
    batch_size: 16,
    validate: {
        batch_size: 32,
    },
    final_validate: {
        batch_size: 16,
        milestones: [20, 40],
        schedule: "multi_step",
    },
    optimizer+: {lr: 0.1},
    num_epochs: 50,
}
