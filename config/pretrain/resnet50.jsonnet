local base = import "moco-train-base.jsonnet";

base {
    batch_size: 32,
    num_workers: 4,

    arch: 'resnet50',
}
