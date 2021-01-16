local base = import "moco-train-base.jsonnet";

base {
    batch_size: 64,
    num_workers: 8,

    arch: 'tsm',
    model+: {
        arch: $.arch,
        num_segments: 8,
        base_model: 'resnet18',
    },

    temporal_transforms+: {
        _size:: 8,
    }
}
