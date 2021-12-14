import os
from fsdet.config.config import get_cfg


def main():
    cfg = get_cfg()
    dataset = "coco"  # coco, isaid
    coco_class_splits = ["voc_nonvoc"]  # voc_nonvoc, none_all
    isaid_class_splits = ["vehicle_nonvehicle"]  # vehicle_nonvehicle, none_all, experiment1, experiment2, experiment3
    # Evaluate on a different dataset.
    # WARNING: Use this experimental feature with care, since it is not fully clear what assumptions have to be made
    #  (depending on e.g. dataset, class_split and phase ...)!
    # Defining an alternative class split is normally not necessary, since the classes to be evaluated should be the
    # same, but the implementation requires this class split to be existent in CLASS_SPLITS[alternative_inference_dataset]!
    alternative_inference_dataset = ""
    alternative_inference_class_split = ""
    ft_mode = "all"  # equivalent to 'ft_subset' in 'wrapper_fine_tuning'
    gpu_ids = [0]
    num_threads = 2
    phase = 2  # phase 1: base-training, phase 2: fine-tuning
    #bs = 16
    shots = [10]  # shots to evaluate on
    seeds = [0]  # seeds to evaluate on
    eval_mode = 'single'  # all, single, last
    # normally, 10k steps. Note: it automatically subtracts 1 to fit the odd iteration in the checkpoint file names
    iterations = [10000]
    layers = 50  # 50, 101
    classifier = 'fc'  # fc, cosine
    tfa = False  # False: randinit surgery
    unfreeze = False  # False: freeze feature extractor while fine-tuning
    # Modify test config options (e.g. for quick test hyperparameter tuning).
    #  Note: these configs are not saved into a config file, the change is just temporary for this certain run!
    opts = [
        'INPUT.AUG.AUGS.RESIZE_SHORTEST_EDGE_LIMIT_LONGEST_EDGE.MIN_SIZE_TEST', 800,
        'INPUT.AUG.AUGS.RESIZE_SHORTEST_EDGE_LIMIT_LONGEST_EDGE.MAX_SIZE_TEST', 1333,
        'TEST.FILTER_EMPTY_ANNOTATIONS', False,
    ]
    # Test-Time Augmentation (TTA) options
    tta_min_sizes = [700, 800, 900]  # [400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
    opts.extend([
        'TEST.AUG.ENABLED', False,  # Set to True to enable
        'TEST.AUG.MIN_SIZES', '\\({}\\)'.format(separate(tta_min_sizes, ',', trailing_sep=True)),
        'TEST.AUG.MAX_SIZE', 4000,
        'TEST.AUG.FLIP', True
    ])
    # dataset-specific options
    if dataset == "coco":
        class_splits = coco_class_splits
        opts.extend([
            'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.05,
            'TEST.DETECTIONS_PER_IMAGE', 100,
            'MODEL.RPN.PRE_NMS_TOPK_TEST', 1000,
            'MODEL.RPN.POST_NMS_TOPK_TEST', 1000
        ])
    elif dataset == "isaid":
        class_splits = isaid_class_splits
        opts.extend([
            'MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.01,
            'TEST.DETECTIONS_PER_IMAGE', 300,
            'MODEL.RPN.PRE_NMS_TOPK_TEST', 2000,
            'MODEL.RPN.POST_NMS_TOPK_TEST', 1500
        ])
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    # ---------------------------------------------------------------------------------------------------------------- #
    if alternative_inference_dataset:
        assert alternative_inference_class_split
        # same pattern as in 'run_base_training.py' and 'run_experiments.py'
        test_dataset_name = "{}_{}_{}_{}".format(
            alternative_inference_dataset,
            alternative_inference_class_split,
            cfg.TEST_SPLIT[alternative_inference_dataset],
            "all" if phase == 2 else "base"
        )
        opts.extend(['DATASETS.TEST', '\\(\\"{}\\",\\)'.format(test_dataset_name)])
    if eval_mode != 'single':  # to prevent multiple execution of inference on all or the last checkpoint!
        iterations = [-1]
    if phase == 1:
        mode = 'base'
        pattern = 'faster_rcnn_R_{}_FPN_{}.yaml'
        for class_split in class_splits:
            for iteration in iterations:
                config_file = os.path.join(
                    cfg.CONFIG_DIR_PATTERN[dataset].format(class_split),
                    pattern.format(layers, mode)
                )
                run_inference(gpu_ids, num_threads, config_file, eval_mode, iteration, opts)
    else:
        assert phase == 2
        assert len(shots) > 0 and len(seeds) > 0
        mode = 'all'
        pattern = 'faster_rcnn_R_{}_FPN_ft{}_{}{}{}{}{}.yaml'
        classifier_str = '_{}'.format(classifier)
        unfreeze_str = '_unfreeze' if unfreeze else ''
        tfa_str = '_TFA' if tfa else ''
        for class_split in class_splits:
            for shot in shots:
                for seed in seeds:
                    for iteration in iterations:
                        # TODO: Possible problems in future:
                        #  1. We hard-code mode 'all'
                        #  2. We don't use a suffix (as in 'run_experiments.py')
                        config_file = os.path.join(
                            cfg.CONFIG_DIR_PATTERN[dataset].format(class_split),
                            'seed{}'.format(seed),
                            'ft_only_novel' if mode == 'novel' else 'ft' + classifier_str + unfreeze_str,  # sub dir
                            pattern.format(layers, classifier_str, ft_mode, '_{}shot'.format(shot), unfreeze_str, tfa_str, '')
                        )
                        run_inference(gpu_ids, num_threads, config_file, eval_mode, iteration, opts)


def run_inference(gpu_ids, num_threads, config_file, eval_mode, iteration, opts):
    assert eval_mode in ['all', 'single', 'last']
    if eval_mode == 'single':  # certain iteration
        eval_mode_str = "--eval-only --eval-iter {}".format(iteration)
    elif eval_mode == 'all':  # all available iterations
        eval_mode_str = "--eval-all"
    else:  # only last iteration
        eval_mode_str = "--eval-only"
    opts_str = '' if not opts else '--opts ' + separate(opts, ' ')
    base_cmd = "python3 -m tools.test_net"
    cmd = "OMP_NUM_THREADS={} CUDA_VISIBLE_DEVICES={} {} --config-file {} --num-gpus {} {} {}"\
        .format(num_threads, separate(gpu_ids, ','), base_cmd, config_file, len(gpu_ids), eval_mode_str, opts_str)
    os.system(cmd)


# note: separate(elements, ' ') == *elements
def separate(elements, separator, trailing_sep=False):
    res = ''
    if not isinstance(elements, (list, tuple)):
        return str(elements)
    assert len(elements) > 0, "need at least one element in the collection {}!".format(elements)
    for element in elements:
        res += '{}{}'.format(str(element), separator)
    if not trailing_sep:
        return res[:-1]  # remove trailing separator
    return res


if __name__ == '__main__':
    main()
