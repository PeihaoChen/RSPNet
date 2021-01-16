local hmdb51 = import '../dataset/hmdb51.libsonnet';
local resnet = import '../model/resnet.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: hmdb51,
    model: resnet.resnet18,
    model_type: 'multitask',
    temporal_transforms+: {
        size: 16,
        validate: {
                    stride: 1,
                    n_crop: 1,
                    final_n_crop: 3,
                },
    },
    batch_size: 64,
    validate: {
        batch_size: 64,
    },
    final_validate: {
        batch_size: 64,
    },
    optimizer+: {
        lr: 0.01,
        milestones: [50, 70, 90],
        schedule: "multi_step",
        },
}
