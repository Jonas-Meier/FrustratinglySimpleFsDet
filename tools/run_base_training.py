import argparse
import os
import sys
from subprocess import PIPE, STDOUT, Popen

import yaml

from class_splits import CLASS_SPLITS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--root', type=str, default='./', help='Root of data')
    parser.add_argument('--bs', type=int, default=16, help='Total batch size, not per GPU!')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate. Set to -1 for automatic linear scaling')
    parser.add_argument('--layers', type=int, default=50, choices=[50, 101], help='Layers of ResNet backbone')
    parser.add_argument('--dataset', type=str, required=True, choices=['coco', 'voc'])
    parser.add_argument('--class-split', type=str, required=True)
    # parser.add_argument('--ckpt-freq', type=int, default=10, help='Frequency of saving checkpoints')
    parser.add_argument('--override', default=False, action='store_true')

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
            'ROI_HEADS': {
                'NUM_CLASSES': int,
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
        'OUTPUT_DIR': str
    }


def load_yaml_file(fname):
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_exp(cfg, configs):  # TODO: mor clear argument names: cfg and configs...???
    """
    Run training and evaluation scripts based on given config files.
    """
    run_train(cfg, configs)
    run_test(cfg, configs)


def run_train(cfg, configs):
    output_dir = configs['OUTPUT_DIR']
    model_path = os.path.join(args.root, output_dir, 'model_final.pth')
    if not os.path.exists(model_path):
        base_cmd = 'python3 -m tools.train_net'  # 'python tools/train_net.py' or 'python3 -m tools.train_net'
        train_cmd = 'OMP_NUM_THREADS={} CUDA_VISIBLE_DEVICES={} {} ' \
                    '--dist-url auto --num-gpus {} --config-file {} --resume'.\
            format(args.num_threads, comma_sep(args.gpu_ids), base_cmd, len(args.gpu_ids), cfg)
        # TODO:
        #  --dist-url: just for obtaining a deterministic port to identify orphan processes
        #  --resume: ??? Using resume or not results in fine-tuning starting from iteration 1
        #  --opts: normally not necessary if we have set the config file appropriate
        run_cmd(train_cmd)


def run_test(cfg, configs):
    output_dir = configs['OUTPUT_DIR']
    res_path = os.path.join(args.root, output_dir, 'inference', 'res_final.json')
    if not os.path.exists(res_path):
        base_cmd = 'python3 -m tools.test_net'  # 'python tools/test_net.py' or 'python3 -m tools.test_net'
        test_cmd = 'OMP_NUM_THREADS={} CUDA_VISIBLE_DEVICES={} {} ' \
                   '--dist-url auto --num-gpus {} --config-file {} --resume --eval-only'. \
            format(args.num_threads, comma_sep(args.gpu_ids), base_cmd, len(args.gpu_ids), cfg)
        run_cmd(test_cmd)


def run_cmd(cmd):
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        line = p.stdout.readline().decode('utf-8')
        if not line:
            break
        print(line)


def get_base_dataset_names(dataset, class_split, mode='base', train_split='trainval', test_split='test'):
    return (
        '{}_{}_{}_{}'.format(dataset, class_split, train_split, mode),
        '{}_{}_{}'.format(dataset, test_split, mode)  # TODO: need to add class split? or directly use *_test_all?
    )


def get_config(override_if_exists=False):
    if args.dataset == 'coco':
        ITERS = (110000, (85000, 100000))  # tuple(max_iter, tuple(<steps>))
        mode = 'base'  # TODO: for base training with all classes, should the mode be 'base' as well?
        config_dir = 'configs/COCO-detection/cocosplit_{}'.format(args.class_split)
        ckpt_dir = 'checkpoints/coco_{}/faster_rcnn'.format(args.class_split)
        base_cfg = '../../Base-RCNN-FPN.yaml'  # adjust depth depending on 'config_dir'
    else:
        raise ValueError("Dataset {} is not supported!".format(args.dataset))

    training_identifier = 'faster_rcnn_R_{}_FPN_{}'.format(args.layers, mode)

    # "detectron2://ImageNetPretrained/MSRA/R-50.pkl", "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
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
    new_config = get_empty_base_config()  # get an empty config and fill it appropriately
    new_config['_BASE_'] = base_cfg
    new_config['MODEL']['WEIGHTS'] = pretrained
    new_config['MODEL']['RESNETS']['DEPTH'] = args.layers
    num_base_classes = len(CLASS_SPLITS[args.dataset][args.class_split]['base'])
    new_config['MODEL']['ROI_HEADS']['NUM_CLASSES'] = num_base_classes
    (train_data, test_data) = get_base_dataset_names(args.dataset, args.class_split, mode,
                                                     train_split='trainval', test_split='test')
    new_config['DATASETS']['TRAIN'] = str((train_data,))
    new_config['DATASETS']['TEST'] = str((test_data,))
    new_config['SOLVER']['IMS_PER_BATCH'] = args.bs  # default: 16
    lr_scale_factor = args.bs / 16
    new_config['SOLVER']['BASE_LR'] = args.lr if args.lr != -1 else 0.02 * lr_scale_factor
    new_config['SOLVER']['STEPS'] = str(ITERS[1])
    new_config['SOLVER']['MAX_ITER'] = ITERS[0]
    new_config['SOLVER']['CHECKPOINT_PERIOD'] = 10000  # ITERS[0] // args.ckpt_freq. Old default: 5000
    new_config['SOLVER']['WARMUP_ITERS'] = 1000  # TODO: ???
    new_config['OUTPUT_DIR'] = base_ckpt_save_dir

    # Save config and return it
    with open(config_save_file, 'w') as fp:
        yaml.dump(new_config, fp, sort_keys=False)  # TODO: 'sort_keys=False' requires pyyaml >= 5.1

    return config_save_file, new_config


def comma_sep(elements):
    res = ''
    if not isinstance(elements, (list, tuple)):
        return str(elements)
    assert len(elements) > 0, "need at least one element in the collection {}!".format(elements)
    if len(elements) == 1:
        return str(elements[0])
    for element in elements:
        res += '{},'.format(str(element))
    return res[:-1]  # remove trailing space


def main():
    cfg, configs = get_config(override_if_exists=args.override)
    #run_exp(cfg, configs)
    run_train(cfg, configs)


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    main()
