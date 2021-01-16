local smth_smth = import '../dataset/smth_smth.libsonnet';
local resnet = import '../model/resnet.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: smth_smth,
    model: resnet.resnet18,
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
    batch_size: 32,
    validate: {
        batch_size: 32,
    },
    final_validate: {
        batch_size: 32,
    },
    optimizer+: {
        lr: 0.01,
        milestones: [20, 40],
        schedule: "multi_step",
        },
    num_epochs: 50,
}
