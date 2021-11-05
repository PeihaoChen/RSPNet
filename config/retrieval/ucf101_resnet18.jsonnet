local ucf101 = import '../dataset/ucf101.libsonnet';
local resnet = import '../model/resnet.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: ucf101,
    model: resnet.resnet18,
    model_type: 'multitask',
    batch_size: 8,
    validate: {
        batch_size: 8,
    },
    final_validate: {
        batch_size: 8,
    },
}
