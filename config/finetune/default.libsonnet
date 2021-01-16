local SGD = import '../optimizer/sgd.libsonnet';

{
    method: 'from-scratch',
    optimizer: SGD,

    log_interval: 10,
    num_workers: 8,
    base_batch_size:: 64,
    batch_size: self.base_batch_size,
    num_epochs: 30,

    model_type: '1stream',

    temporal_transforms: {
        size: 16,
        strides: [
            {stride: 1, weight: 1},
        ],

        validate: {
            stride: 1,
            n_crop: 1,
            final_n_crop: 10,
        },

        frame_rate: null
    },

    spatial_transforms: {
        size: 112,
        crop_area: {
            min: 0.25,
            max: 1.0,
        },
        gray_scale: 0,
        color_jitter: {
            brightness: 0,
            contrast: 0,
            saturation: 0,
            hue: 0,
        },
    },

    validate: {
        batch_size: std.floor($.base_batch_size * 2 / $.temporal_transforms.validate.n_crop),
    },
    final_validate: {
        // TODO: batch_size 1 cause problems. investigating.
        batch_size: std.max(std.floor($.validate.batch_size / $.temporal_transforms.validate.final_n_crop), 2),
    },
}
