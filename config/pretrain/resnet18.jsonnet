local base = import "moco-train-base.jsonnet";

base {
    batch_size: 64,
    num_workers: 8,

    arch: 'resnet18',
}
