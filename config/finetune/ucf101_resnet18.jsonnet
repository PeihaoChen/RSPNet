local ucf101 = import '../dataset/ucf101.libsonnet';
local resnet = import '../model/resnet.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: ucf101,
    model: resnet.resnet18,
    model_type: 'multitask',
    temporal_transforms+: {
        size: 16,
        frame_rate: 25
    },
    local batch_size_factor =112*112*8 / self.temporal_transforms.size / self.spatial_transforms.size / self.spatial_transforms.size,
    batch_size: 64 * batch_size_factor,
    validate: {
        batch_size: 128 * batch_size_factor,
    },
    final_validate: {
        batch_size: 16 * batch_size_factor,
    },
    optimizer+: {lr: 0.1},
    num_epochs: 30,
}
