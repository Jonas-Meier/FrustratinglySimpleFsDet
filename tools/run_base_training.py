import argparse
import os
import shutil
import shlex
import time
from subprocess import PIPE, STDOUT, Popen

import yaml

from class_splits import CLASS_SPLITS
from fsdet.config.config import get_cfg
cfg = get_cfg()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=cfg.DATASETS.SUPPORTED_DATASETS)
    parser.add_argument('--class-split', type=str, required=True)
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--layers', type=int, default=50, choices=[50, 101], help='Layers of ResNet backbone')
    parser.add_argument('--bs', type=int, default=16, help='Total batch size, not per GPU!')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate. Set to -1 for automatic linear scaling')
    parser.add_argument('--augmentations', type=str, nargs='*',
                        default=['ResizeShortestEdgeLimitLongestEdge', 'RandomHFlip'],
                        help='The augmentations to be used at base training.')
    # parser.add_argument('--ckpt-freq', type=int, default=10, help='Frequency of saving checkpoints')
    parser.add_argument('--override-config', default=False, action='store_true',
                        help='Override config file if it already exists')
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--root', type=str, default='./', help='Root of data')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Try to resume a training on the latest checkpoint. Do not set this argument together '
                             'with --force-retrain!')
    parser.add_argument('--force-retrain', default=False, action='store_true',
                        help='If it is set to False (default) and the target directory already exists, the program '
                             'aborts. If it is set to True, it will delete the already existent target directory and '
                             'starts the training. Do not set this argument together with --resume!')
    parser.add_argument('--opts', nargs='*', default=[],
                        help='Override configs for this very call.')
    return parser.parse_args()


def get_empty_base_config():
    # TODO: probably replace by OrderedDicts!
    return {
        '_BASE_': str,
        'MODEL': {
            'WEIGHTS': str,
            'MASK_ON': False,
            'RESNETS': {
                'DEPTH': int
            },
            'ANCHOR_GENERATOR': {
                'SIZES': [[int]]
            },
            'RPN': {
                'PRE_NMS_TOPK_TRAIN': int,
                'PRE_NMS_TOPK_TEST': int,
                'POST_NMS_TOPK_TRAIN': int,
                'POST_NMS_TOPK_TEST': int
            },
            'ROI_HEADS': {
                'NUM_CLASSES': int,
                'SCORE_THRESH_TEST': float
            }
        },
        'DATASETS': {
            'TRAIN': (str,),
            'TEST': (str,)
        },
        'SOLVER': {
            'IMS_PER_BATCH': int,
            'BASE_LR': float,
            'STEPS': (int,),
            'MAX_ITER': int,
            'CHECKPOINT_PERIOD': int,
            'WARMUP_ITERS': 1000  # default
        },
        'INPUT': {
            'AUG': {
                'TYPE': str,
                'PIPELINE': [str],
                'AUGS': {
                    'RESIZE_SHORTEST_EDGE_LIMIT_LONGEST_EDGE': {
                        'MIN_SIZE_TRAIN': (int,)
                    }
                }
            }
        },
        'TEST': {
            'DETECTIONS_PER_IMAGE': int
        },
        'OUTPUT_DIR': str
    }


def load_yaml_file(fname):
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_exp(config_file, config):
    """
    Run training and evaluation scripts based on given config files.
    """
    run_train(config_file, config)
    run_test(config_file, config)


def run_train(config_file, config):
    output_dir = config['OUTPUT_DIR']
    save_dir = os.path.join(args.root, output_dir)
    base_cmd = 'python3 -m tools.train_net'  # 'python tools/train_net.py' or 'python3 -m tools.train_net'
    resume_str = '' if not args.resume else ' --resume'
    opts_str = '' if not args.opts else ' --opts ' + separate(args.opts, ' ')
    train_cmd = 'OMP_NUM_THREADS={} CUDA_VISIBLE_DEVICES={} {} ' \
                '--dist-url auto --num-gpus {} --config-file {}{}{}'. \
        format(args.num_threads, separate(args.gpu_ids, ','), base_cmd, len(args.gpu_ids), config_file,
               resume_str, opts_str)
    assert not (args.resume and args.force_retrain)

    if os.listdir(save_dir) and not args.resume:
        # Save directory is not empty and --resume is not set.
        # --force-override has to be specified to clean the directory prior to training
        if args.force_retrain:
            print("INFO: Deleting non-empty directory '{}'".format(save_dir))
            time.sleep(0.1)
            shutil.rmtree(save_dir)
            os.makedirs(save_dir)
        else:
            print("Error, model save path '{}' is not empty. Set '--resume' to resume to the latest checkpoint or"
                  " '--force-override' to clean the directory and start a fresh training! Aborting..."
                  .format(save_dir))
            exit(1)
    run_cmd(train_cmd)


def run_test(config_file, config):
    output_dir = config['OUTPUT_DIR']
    res_path = os.path.join(args.root, output_dir, 'inference', 'res_final.json')
    if not os.path.exists(res_path):
        base_cmd = 'python3 -m tools.test_net'  # 'python tools/test_net.py' or 'python3 -m tools.test_net'
        test_cmd = 'OMP_NUM_THREADS={} CUDA_VISIBLE_DEVICES={} {} ' \
                   '--dist-url auto --num-gpus {} --config-file {} --resume --eval-only'. \
            format(args.num_threads, separate(args.gpu_ids, ','), base_cmd, len(args.gpu_ids), config_file)
        run_cmd(test_cmd)


def run_cmd(cmd):
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        line = p.stdout.readline().decode('utf-8')
        if not line:
            break
        print(line)


def get_base_dataset_names(dataset, class_split, mode='base', train_split='trainval', test_split='test'):
    assert mode == 'base', 'Only support mode \'base\' currently!'
    # TODO: think about hard-coding mode 'base' if we can rule out other possibilities
    return (
        '{}_{}_{}_{}'.format(dataset, class_split, train_split, mode),
        '{}_{}_{}_{}'.format(dataset, class_split, test_split, mode)
    )


def get_config(override_if_exists=False):  # TODO: default 'override_if_exists' to True?
    if args.dataset == 'coco':
        ITERS = (110000, (85000, 100000))  # tuple(max_iter, tuple(<steps>))
    elif args.dataset == 'isaid':
        ITERS = (60000, (25000, 40000))
    else:
        raise ValueError("Dataset {} is not supported!".format(args.dataset))

    mode = 'base'  # TODO: for base training with all classes, should the mode be 'base' as well?
    config_dir = cfg.CONFIG_DIR_PATTERN[args.dataset].format(args.class_split)
    ckpt_dir = os.path.join(
        cfg.CKPT_DIR_PATTERN[args.dataset].format(args.class_split),
        'faster_rcnn'
    )
    base_cfg = '../../Base-RCNN-FPN.yaml'  # adjust depth depending on 'config_dir'
    train_split = cfg.TRAIN_SPLIT[args.dataset]
    test_split = cfg.TEST_SPLIT[args.dataset]

    training_identifier = 'faster_rcnn_R_{}_FPN_{}'.format(args.layers, mode)

    # "detectron2://ImageNetPretrained/MSRA/R-50.pkl", "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
    # TODO: add possibility to use a pretrained detector (add an argument to argparse ... !)
    pretrained = 'pretrained/ImageNetPretrained/MSRA/R-{}.pkl'.format(args.layers)
    # Save dir for base training
    base_ckpt_save_dir = os.path.join(ckpt_dir, training_identifier)
    os.makedirs(base_ckpt_save_dir, exist_ok=True)
    # Save dir for config file
    config_save_dir = os.path.join(args.root, config_dir)
    os.makedirs(config_save_dir, exist_ok=True)
    config_save_file = os.path.join(config_save_dir, training_identifier + '.yaml')

    # If the config already exists, return it (if we don't want to override it)
    if os.path.exists(config_save_file) and not override_if_exists:
        print("Config already exists, returning the existing config...")
        return config_save_file, load_yaml_file(config_save_file)
    print("Creating a new config file: {}".format(config_save_file))

    # Set all values in the empty config
    # TODO: probably add dataset specific definitions!
    new_config = get_empty_base_config()  # get an empty config and fill it appropriately
    new_config['_BASE_'] = base_cfg
    new_config['MODEL']['WEIGHTS'] = pretrained
    new_config['MODEL']['RESNETS']['DEPTH'] = args.layers
    new_config['MODEL']['ANCHOR_GENERATOR']['SIZES'] = str([[32], [64], [128], [256], [512]])
    new_config['MODEL']['RPN']['PRE_NMS_TOPK_TRAIN'] = 2000  # Per FPN level. TODO: per batch or image?
    new_config['MODEL']['RPN']['PRE_NMS_TOPK_TEST'] = 1000  # Per FPN level. TODO: per batch or image?
    new_config['MODEL']['RPN']['POST_NMS_TOPK_TRAIN'] = 1000  # TODO: per batch or image?
    new_config['MODEL']['RPN']['POST_NMS_TOPK_TEST'] = 1000  # TODO: per batch or image?
    num_base_classes = len(CLASS_SPLITS[args.dataset][args.class_split]['base'])
    new_config['MODEL']['ROI_HEADS']['NUM_CLASSES'] = num_base_classes
    new_config['MODEL']['ROI_HEADS']['SCORE_THRESH_TEST'] = 0.05
    (train_data, test_data) = get_base_dataset_names(args.dataset, args.class_split, mode,
                                                     train_split, test_split)
    new_config['DATASETS']['TRAIN'] = str((train_data,))
    new_config['DATASETS']['TEST'] = str((test_data,))
    new_config['SOLVER']['IMS_PER_BATCH'] = args.bs  # default: 16
    lr_scale_factor = args.bs / 16
    new_config['SOLVER']['BASE_LR'] = args.lr if args.lr != -1 else 0.02 * lr_scale_factor
    new_config['SOLVER']['STEPS'] = str(ITERS[1])
    new_config['SOLVER']['MAX_ITER'] = ITERS[0]  # TODO: increase MAX_ITER if batch size is < 16?
    new_config['SOLVER']['CHECKPOINT_PERIOD'] = 10000  # ITERS[0] // args.ckpt_freq. Old default: 5000
    new_config['SOLVER']['WARMUP_ITERS'] = 1000  # TODO: ???
    # custom activates the custom pipeline, new augmentation configs and disables configs that were needed for the
    # Detectron2 default augmentations
    new_config['INPUT']['AUG']['TYPE'] = 'custom'
    new_config['INPUT']['AUG']['PIPELINE'] = str(args.augmentations)
    new_config['INPUT']['AUG']['AUGS']['RESIZE_SHORTEST_EDGE_LIMIT_LONGEST_EDGE']['MIN_SIZE_TRAIN'] = str((640, 672, 704, 736, 768, 800))  # scales for multi-scale training
    new_config['TEST']['DETECTIONS_PER_IMAGE'] = 100
    new_config['OUTPUT_DIR'] = base_ckpt_save_dir

    if args.dataset == 'coco':
        new_config['MODEL']['ANCHOR_GENERATOR']['SIZES'] = str([[32], [64], [128], [256], [512]])
        new_config['TEST']['DETECTIONS_PER_IMAGE'] = 100
        new_config['INPUT']['AUG']['AUGS']['RESIZE_SHORTEST_EDGE_LIMIT_LONGEST_EDGE']['MIN_SIZE_TRAIN'] = str((640, 672, 704, 736, 768, 800))
    elif args.dataset == 'isaid':
        new_config['MODEL']['ANCHOR_GENERATOR']['SIZES'] = str([[16], [32], [64], [128], [256]])
        new_config['MODEL']['RPN']['PRE_NMS_TOPK_TRAIN'] = 3000
        new_config['MODEL']['RPN']['POST_NMS_TOPK_TRAIN'] = 1500
        new_config['MODEL']['RPN']['PRE_NMS_TOPK_TEST'] = 1000
        new_config['MODEL']['RPN']['POST_NMS_TOPK_TEST'] = 1000
        new_config['TEST']['DETECTIONS_PER_IMAGE'] = 100
        new_config['INPUT']['AUG']['AUGS']['RESIZE_SHORTEST_EDGE_LIMIT_LONGEST_EDGE']['MIN_SIZE_TRAIN'] = str((600, 700, 800, 900, 1000))  #  (608, 672, 736, 800, 864, 928, 992)

    # Add all opts to the dictionary file, because:
    #  1. Otherwise they would only be written into ./checkpoints/.../config.yaml but not into ./configs/.../xxx.yaml
    #     which could cause confusion to the user
    #  2. When the inference is executed, the config file is passed to test_net.py which first reads in the default
    #     configs and then merges the config file into them. If the opts are not present in the config file
    #     (./configs/.../xxx.yaml) but only in ./checkpoints/.../config.yaml, the latter is overridden (by the method
    #     fsdet/engine/defaults.py:default_setup) which causes the opts to be present nowhere which makes later
    #     debugging of training configs difficult.
    for keys, value in zip(args.opts[0::2], args.opts[1::2]):
        set_nested_value(new_config, keys.split('.'), value)

    # Save config and return it
    with open(config_save_file, 'w') as fp:
        yaml.dump(new_config, fp, sort_keys=False)  # TODO: 'sort_keys=False' requires pyyaml >= 5.1

    return config_save_file, new_config


def separate(elements, separator, trailing_sep=False):
    res = ''
    if not isinstance(elements, (list, tuple)):
        return shlex.quote(str(elements))
    assert len(elements) > 0, "need at least one element in the collection {}!".format(elements)
    for element in elements:
        res += '{}{}'.format(shlex.quote(str(element)), separator)
    if not trailing_sep:
        return res[:-1]  # remove trailing separator
    return res


def set_nested_value(dic: dict, keys: [str], value: str):
    """
    Note: This method is much simpler than Detectron's ConfigNode merge methods. For that reason, this method has currently
     no capabilities for checking the type of the passed values, so it's the user's responsibility to pass the correct
     types to --opts argument. Possibly wrong config value type are nonetheless detected by the training, when Detectron
     merges this config file into the default configs.
    """
    if not keys:
        return
    if len(keys) == 1:
        # TODO: probably check for existence of the key -> the old value will be overridden
        dic[keys[0]] = value
        return
    else:
        if keys[0] not in dic:
            dic[keys[0]] = {}
        set_nested_value(dic[keys[0]], keys[1:], value)


def main():
    config_file, config = get_config(override_if_exists=args.override_config)
    #run_exp(config_file, config)
    run_train(config_file, config)


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    main()
