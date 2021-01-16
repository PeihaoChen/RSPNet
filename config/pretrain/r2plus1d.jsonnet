local base = import "moco-train-base.jsonnet";

base {
    batch_size: 32,
    num_workers: 4,

    arch: 'r2plus1d-vcop',

    spatial_transforms+: {
        size: 112,
    },
    temporal_transforms+: {
        _size: 16
    },
    optimizer+: {
        lr: 0.05
    }
}
