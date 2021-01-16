local hmdb51 = import '../dataset/hmdb51.libsonnet';
local r2plus1d = import '../model/r2plus1d.libsonnet';
local default = import './default.libsonnet';

default {
    dataset: hmdb51,
    model: r2plus1d,
    model_type: 'multitask',
    local batch_size_factor =112*112*16 / self.temporal_transforms.size / self.spatial_transforms.size / self.spatial_transforms.size,
    batch_size: 16 * batch_size_factor,
    validate: {
        batch_size: 8 * batch_size_factor,
    },
    final_validate: {
        batch_size: 4 * batch_size_factor,
    },
    optimizer+: {lr: 0.1},
}