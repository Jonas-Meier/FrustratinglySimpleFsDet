import os
import shlex
from fsdet.config.config import get_cfg
cfg = get_cfg()


def main():
    dataset = "coco"  # coco, isaid{...}, fair1m{...}, fairsaid
    coco_class_split = "voc_nonvoc"  # voc_nonvoc, none_all
    isaid_class_split = "vehicle_nonvehicle"  # vehicle_nonvehicle, none_all, experiment1, experiment2, experiment3
    fair1m_class_split = "none_all"  # none_all, experiment1
    fairsaid_class_split = "none_all"  # none_all
    gpu_ids = [0]
    num_threads = 2  # two threads seem to be a bit faster than just one, but four threads are as fast as two threads!
    bs = 16
    lr = 0.02  # 0.02 for bs=16. Set to -1 for automatic linear scaling!
    layers = 50  # 50, 101
    resume = False  # Try to resume on the latest checkpoint. Do not set together with 'force_retrain'!
    force_retrain = False  # If the save directory is not empty, delete its content. Do not set together with 'resume'!
    # Choose from 'ResizeShortestEdgeLimitLongestEdge', 'RandomHFlip', 'RandomVFlip', 'RandomFourAngleRotation',
    #  'Random50PercentContrast', 'Random50PercentBrightness', 'Random50PercentSaturation', 'RandomAlexNetLighting'
    #  Default: ["ResizeShortestEdgeLimitLongestEdge", "RandomHFlip"]
    augmentations = [
        "ResizeShortestEdgeLimitLongestEdge",
        "RandomHFlip"
    ]
    override_config = True
    augmentation_params = {
        "ResizeShortestEdgeLimitLongestEdge": {

        },
        "RandomHFlip": {
            "PROB": 0.5,
        },
        "RandomVFlip": {
            "PROB": 0.5,
        },
        "Random50PercentContrast": {
            "INTENSITY_MIN": 0.5,
            "INTENSITY_MAX": 1.5
        },
        "Random50PercentBrightness": {
            "INTENSITY_MIN": 0.5,
            "INTENSITY_MAX": 1.5
        },
        "Random50PercentSaturation": {
            "INTENSITY_MIN": 0.5,
            "INTENSITY_MAX": 1.5
        },
        "RandomAlexNetLighting": {
            "SCALE": 0.1
        },
        "AlbumentationsGaussNoise": {
            "P": 0.5,
            "VAR_LIMIT": (10, 50),
        },
        "AlbumentationsISONoise": {
            "P": 0.5,
            "COLOR_SHIFT": (0.01, 0.05),
            "INTENSITY": (0.1, 0.5)
        },
        "AlbumentationsGaussBlur": {
            "P": 0.5,
            "BLUR_LIMIT": (3, 7)
        },
    }
    # ---------------------------------------------------------------------------------------------------------------- #
    opts = []
    if dataset == "coco":
        class_split = coco_class_split
    elif dataset.startswith("isaid"):
        class_split = isaid_class_split
    elif dataset.startswith("fair1m"):
        class_split = fair1m_class_split
    elif dataset == "fairsaid":
        class_split = fairsaid_class_split
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    opts.extend(get_augmentation_opts_list(augmentations, augmentation_params))
    run_base_training(dataset, class_split, gpu_ids, num_threads, layers, augmentations, bs, lr,
                      override_config, resume, force_retrain, opts)


def run_base_training(dataset, class_split, gpu_ids, num_threads, layers, augmentations, bs, lr=-1.0,
                      override_config=False, resume=False, force_retrain=False, opts=None):
    base_cmd = "python3 -m tools.run_base_training"
    override_config_str = ' --override-config' if override_config else ''
    resume_str = ' --resume' if resume else ''
    force_retrain_str = ' --force-retrain' if force_retrain else ''
    opts_str = '' if not opts else ' --opts ' + separate(opts, ' ')
    cmd = "{} --dataset {} --class-split {} --gpu-ids {} --num-threads {} --layers {} " \
          "--augmentations {} --bs {} --lr {}{}{}{}{}"\
        .format(base_cmd, dataset, class_split, separate(gpu_ids, ' '), num_threads, layers,
                separate(augmentations, ' '), bs, lr, override_config_str, resume_str, force_retrain_str, opts_str)
    os.system(cmd)


def get_augmentation_opts_list(augmentations, augmentation_params):
    augs_opts = []
    cfg_base = 'INPUT.AUG.AUGS'
    aug_name_to_cfg_name = {v.NAME: k for k, v in cfg.INPUT.AUG.AUGS.items()}
    for aug_name, param_dict in augmentation_params.items():
        if aug_name not in augmentations or aug_name not in aug_name_to_cfg_name:
            continue
        for param_name, value in param_dict.items():
            assert param_name in cfg.INPUT.AUG.AUGS.get(aug_name_to_cfg_name[aug_name])
            augs_opts.extend([
                "{}.{}.{}".format(cfg_base, aug_name_to_cfg_name[aug_name], param_name),
                value
            ])
    return augs_opts


def separate(elements, separator):
    res = ''
    if not isinstance(elements, (list, tuple)):
        return shlex.quote(str(elements))
    assert len(elements) > 0, "need at least one element in the collection {}!".format(elements)
    for element in elements:
        res += '{}{}'.format(shlex.quote(str(element)), separator)
    return res[:-1]  # remove trailing separator


if __name__ == '__main__':
    main()
