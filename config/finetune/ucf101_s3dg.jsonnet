local ucf101 = import "../dataset/ucf101.libsonnet";
local s3dg = import "../model/s3dg.libsonnet";
local default = import './default.libsonnet';

default {
    dataset: ucf101,
    model: s3dg,
    model_type: 'multitask',
    spatial_transforms+: {
        size: 224
    },
    temporal_transforms+: {
        size: 64,
        frame_rate: 25
    },
    batch_size: 4,
    validate: {
        batch_size: 4,
    },
    final_validate: {
        batch_size: 4,
    },
    optimizer+: {lr: 0.005},
    num_epochs: 50,
}
