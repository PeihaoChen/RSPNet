
local getSlowFastConfig(name) = 'config/slowfast-configs/Kinetics/%s.yaml' % name;

{
    arch: 'slowfast',
    cfg_file: {
        slowfast_4x16_r50: getSlowFastConfig('SLOWFAST_4x16_R50'),
        slowfast_nln_4x16_r50: getSlowFastConfig('SLOWFAST_NLN_4x16_R50'),
    }
}