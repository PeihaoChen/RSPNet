local ucf101 = import '../dataset/ucf101.libsonnet';
local c3d = import '../model/c3d.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: ucf101,
    model: c3d,
    model_type: 'multitask',
    batch_size: 8,
    validate: {
        batch_size: 8,
    },
    final_validate: {
        batch_size: 8,
    },
}
