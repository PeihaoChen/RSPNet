{
    finetune:: {
        num_epochs: if super.dataset.name == 'hmdb51' then 70 else 30,
    },
    linear:: {
        only_train_fc: true,
        batch_size: super.base_batch_size * 8,
    },
    smth_linear:: $.linear {
        num_epochs: 16,
        optimizer+: {
            lr: 0.05,
            schedule: 'multi_step',
            milestones: [10,14],
        },
    },
    multitask: {
        model_type: 'multitask'
    },
    model_2stream: {
        model_type: '2stream',
        optimizer+: {
            lr: 0.01
        }
    },
    addtrans: {
        spatial_transforms+: {
            gray_scale: 0.2,
            color_jitter: {
                brightness: 0.4,
                contrast: 0.4,
                saturation: 0.4,
                hue: 0.4,
            },
        },
    },
    tsm_16f:: {
        assert self.model.arch == 'tsm',
        temporal_transforms+: {
            size: 16
        },
    },
    tsm_224:: {
        assert self.model.arch == 'tsm',
        spatial_transforms+: {
            size: 224
        },
    },
    sp_224: {
        spatial_transforms+: {
            size: 224
        },
    },
    r18k400: {
        model: {
            arch: "torchvision-resnet18",
            pretrain: true
         }
    },
    tsm_smthv2_finetune: {
        num_epochs: 50,
        optimizer+: {
            lr: 0.01,
            schedule: 'multi_step',
            milestones: [20,40],
        },
    }
}
