local smth_smth = import '../dataset/smth_smth.libsonnet';
local s3dg = import '../model/s3dg.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: smth_smth,
    model: s3dg,
    model_type: 'multitask',
    temporal_transforms+: {
        size: 16,
        type: 'cover',
        validate+: {
            final_n_crop: 2,
        },
    },
    spatial_transforms+: {
        size: 224,
        h_flip: 0,
    },
    batch_size: 16,
    validate: {
        batch_size: 32,
    },
    final_validate: {
        batch_size: 16,
    },
    optimizer+: {
        lr: 0.01,
        milestones: [20, 40],
        schedule: "multi_step",
        },
    num_epochs: 50,
}
