local SGD = import '../optimizer/sgd.libsonnet';

{
    method: 'from-scratch',
    optimizer: SGD,

    log_interval: 10,
    num_workers: 8,
    base_batch_size:: 64,
    batch_size: 2,
    model_type: '1stream',

    temporal_transforms: {
        size: 16,
        type: "clip",
        force_n_crop: true,
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
        batch_size: 2,
    },
    final_validate: {
        // TODO: batch_size 1 cause problems. investigating.
        batch_size: 2,
    },
}
