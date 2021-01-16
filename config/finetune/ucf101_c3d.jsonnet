local ucf101 = import '../dataset/ucf101.libsonnet';
local c3d = import '../model/c3d.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: ucf101,
    model: c3d,
    model_type: 'multitask',
    batch_size: 20,
    validate: {
        batch_size: 40,
    },
    final_validate: {
        batch_size: 4,
    },
    optimizer+: {lr: 0.005},
    num_epochs: 30,
}
