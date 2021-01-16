{
    no_random_gray: {
        spatial_transforms+: {
            gray_scale: 0,
        },
    },
    no_color_jitter: {
        spatial_transforms+: {
            color_jitter: {
                brightness: 0,
                contrast: 0,
                saturation: 0,
                hue: 0,
            },
        },
    },
    random_stride: {
        temporal_transforms+: {
            strides: [
                {stride: 1, weight: 1},
                {stride: 2, weight: 1},
                {stride: 4, weight: 1},
            ],
        },
    },
    weighted_stride: {
        temporal_transforms+: {
            strides: [
                {stride: 1, weight: 8},
                {stride: 2, weight: 1},
                {stride: 4, weight: 1},
            ],
        },
    },
    M0: {
        loss_lambda+: {
            M: 0,
        },
    },
    A0: {
        loss_lambda+: {
            A: 0,
        },
    },
    fps25: {
        temporal_transforms+: {
            frame_rate: 25,
        },
    }
}
