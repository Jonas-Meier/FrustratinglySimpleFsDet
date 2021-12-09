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
    # Set both (FT on different dataset and classes) or none.
    # WARNING: Use this experimental feature with care, since it is unknown what possible side effects are possible
    #  since it is unknown what exact assumptions have to be made for certain setups (of datasets and class splits ...)
    # If the alternative dataset does not use the same classes than the "regular" dataset, you probably should
    #  at least discard the base class predictor weights (keep_base_weights = False) or, if the classes are the same,
    #  except for the name and id, set a bijective mapping between those classes in 'class_splits:CLASS_NAME_TRANSFORMS'
    alternative_dataset, alternative_class_split = "", ""
    ft_subset = "all"  # default: fine-tuning on "all" classes. Set to "novel" if base classes are available but only novel class fine-tuning is intended.
    gpu_ids = [0]
    num_threads = 2  # two threads seem to be a bit faster than just one, but four threads are as fast as two threads!
    bs = 16
    lr = 0.001  # 0.001 for bs=16. Set to -1 for automatic linear scaling!
    shots = [10]
    seeds = [0]  # single seed or two seeds representing a range, 2nd argument inclusive!
    explicit_seeds = False  # set to True to specify the exact seeds to train, rather than a range of seeds
    layers = 50  # 50, 101
    resume = False  # Try to resume on the latest checkpoint. So nt set together with 'force_retrain'!
    force_retrain = False  # If the save directory is not empty, delete its content. Do not set together with 'resume'!
    # Choose from 'ResizeShortestEdgeLimitLongestEdge', 'RandomHFlip', 'RandomVFlip', 'RandomFourAngleRotation',
    #  'Random50PercentContrast', 'Random50PercentBrightness', 'Random50PercentSaturation', 'RandomAlexNetLighting'
    #  Default: ["ResizeShortestEdgeLimitLongestEdge", "RandomHFlip"]
    augmentations = [
        "ResizeShortestEdgeLimitLongestEdge",
        "RandomHFlip"
    ]
    # Set following three variables to -1 for using default hard-coded value depending on dataset and shot
    max_iter = -1  # maximum iteration
    # Force no steps by using a single value greater than max_iter, behaviour of empty list is unknown!
    lr_decay_steps = [-1]  # learning rate decay steps
    ckpt_interval = -1  # interval to create checkpoints
    classifier = 'fc'  # fc, cosine
    tfa = False  # False: randinit surgery
    keep_base_weights = True  # keep predictor weights of base classes. Default: True
    keep_bg_weights = True  # keep predictor weights for background class. Default: True
    # experimental: different heads for base classes and novel classes. Only works with 'randinit' surgery (tfa==False)
    double_head = False  # TODO: set 'override_surgery' if 'double_head' == True?
    # Unfreeze settings. 'unfreeze' setting combines the three single settings.
    # Unfreeze settings are combined with 'or', therefore a part of the feature extractor is unfreezed if
    #  either unfreeze==True OR if the corresponding part is unfreezed
    unfreeze = False  # False: freeze feature extractor (backbone + proposal generator + roi head) while fine-tuning
    unfreeze_backbone = False
    unfreeze_proposal_generator = False
    #unfreeze_roi_head = False
    # Note: separate conv and fc unfreezing is disabled for double_head!
    unfreeze_roi_box_head_convs = []  # []: we have no box head conv layers!
    unfreeze_roi_box_head_fcs = []  # [2]: unfreeze the second of both fc layers (1024x1024)
    # Override existing config, force re-creation of surgery checkpoint
    override_config = True
    override_surgery = True
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
    if resume:
        override_config = override_surgery = False
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
    assert (alternative_dataset and alternative_class_split) or \
           (not alternative_dataset and not alternative_class_split)
    opts.extend(get_augmentation_opts_list(augmentations, augmentation_params))
    run_fine_tuning(dataset, class_split, shots, seeds, gpu_ids, num_threads, layers, augmentations, bs, lr,
                    alternative_dataset, alternative_class_split, ft_subset, max_iter, lr_decay_steps,
                    ckpt_interval, explicit_seeds, double_head, keep_base_weights,
                    keep_bg_weights, tfa, unfreeze, unfreeze_backbone,
                    unfreeze_proposal_generator, unfreeze_roi_box_head_convs, unfreeze_roi_box_head_fcs,
                    classifier, override_config, override_surgery, resume, force_retrain,
                    opts)


def run_fine_tuning(dataset, class_split, shots, seeds, gpu_ids, num_threads, layers, augmentations, bs, lr=-1.0,
                    alt_dataset="", alt_class_split="", ft_subset="all", max_iter=-1, lr_decay_steps=[-1],
                    ckpt_interval=-1, explicit_seeds=False, double_head=False, keep_base_weights=True,
                    keep_bg_weights=True, tfa=False, unfreeze=False, unfreeze_backbone=False,
                    unfreeze_proposal_generator=False, unfreeze_roi_box_head_convs=[], unfreeze_roi_box_head_fcs=[],
                    classifier='fc', override_config=False, override_surgery=False, resume=False, force_retrain=False,
                    opts=None):
    base_cmd = "python3 -m tools.run_experiments"
    explicit_seeds_str = ' --explicit-seeds' if explicit_seeds else ''
    surgery_str = ''  # combine different surgery settings to spare some space
    surgery_str = surgery_str + ' --tfa' if tfa else surgery_str
    surgery_str = surgery_str + ' --double-head' if double_head else surgery_str
    keep_weights_str = ''
    keep_weights_str = keep_weights_str + ' --discard-base-weights' if not keep_base_weights else keep_weights_str
    keep_weights_str = keep_weights_str + ' --discard-bg-weights' if not keep_bg_weights else keep_weights_str
    unfreeze_str = ''
    unfreeze_str = unfreeze_str + ' --unfreeze' if unfreeze else unfreeze_str
    unfreeze_str = unfreeze_str + ' --unfreeze-backbone' if unfreeze_backbone else unfreeze_str
    unfreeze_str = unfreeze_str + ' --unfreeze-proposal-generator' if unfreeze_proposal_generator else unfreeze_str
    if unfreeze_roi_box_head_convs:
        unfreeze_str = unfreeze_str + ' --unfreeze-roi-box-head-convs ' + separate(unfreeze_roi_box_head_convs, ' ')
    if unfreeze_roi_box_head_fcs:
        unfreeze_str = unfreeze_str + ' --unfreeze-roi-box-head-fcs ' + separate(unfreeze_roi_box_head_fcs, ' ')
    alt_dataset_class_split_str = ""
    if alt_dataset:  # alt_class_split should be set as well!
        alt_dataset_class_split_str = " --alt-dataset {} --alt-class-split {}".format(alt_dataset, alt_class_split)
    override_config_str = ' --override-config' if override_config else ''
    override_surgery_str = ' --override-surgery' if override_surgery else ''
    resume_str = ' --resume' if resume else ''
    force_retrain_str = ' --force-retrain' if force_retrain else ''
    opts_str = '' if not opts else ' --opts ' + separate(opts, ' ')
    cmd = "{} --dataset {} --class-split {} --shots {} --seeds {}  --gpu-ids {} " \
          "--num-threads {} --layers {} --augmentations {} --bs {} --lr {} --max-iter {} --lr-decay-steps {}  " \
          "--ckpt-interval {} --classifier {} --target-class-set {}{}{}{}{}{}{}{}{}{}{}"\
        .format(base_cmd, dataset, class_split, separate(shots, ' '), separate(seeds, ' '), separate(gpu_ids, ' '),
                num_threads, layers, separate(augmentations, ' '), bs, lr, max_iter, separate(lr_decay_steps, ' '),
                ckpt_interval, classifier, ft_subset, surgery_str, keep_weights_str, unfreeze_str, override_config_str,
                override_surgery_str, explicit_seeds_str, alt_dataset_class_split_str, resume_str, force_retrain_str,
                opts_str)
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
