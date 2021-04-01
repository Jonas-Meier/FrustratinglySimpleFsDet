import argparse
import os
import yaml
from subprocess import PIPE, STDOUT, Popen

from class_splits import CLASS_SPLITS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--gpu-ids', type=int, nargs='+', default=[0])
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 3, 5, 10],
                        help='Shots to run experiments over')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 20],
                        help='Range of seeds to run')
    parser.add_argument('--root', type=str, default='./', help='Root of data')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of path')
    parser.add_argument('--bs', type=int, default=16, help='Total batch size, not per GPU!')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. Set to -1 for automatic linear scaling')
    parser.add_argument('--ckpt-freq', type=int, default=10,
                        help='Frequency of saving checkpoints')
    parser.add_argument('--override', default=False, action='store_true')
    # Model
    parser.add_argument('--layers', type=int, default=50, choices=[50, 101], help='Layers of ResNet backbone')
    parser.add_argument('--fc', action='store_true',
                        help='Model uses FC instead of cosine')
    parser.add_argument('--tfa', action='store_true',
                        help='Two-stage fine-tuning')
    parser.add_argument('--unfreeze', action='store_true',
                        help='Unfreeze feature extractor')
    # TODO: add argument --eval-only which will just execute evaluations!
    #  -> How can we tell get_cfg that we just want the correct config without doing a surgery?
    # Dataset
    parser.add_argument('--dataset', type=str, required=True, choices=['coco', 'voc'])
    parser.add_argument('--class-split', type=str, required=True)  # TODO: allow multiple class splits?
    # PASCAL arguments
    parser.add_argument('--split', '-s', type=int, default=1, help='Data split')

    args = parser.parse_args()
    return args


def get_empty_ft_config():
    return {
        '_BASE_': str,
        'MODEL': {
            'WEIGHTS': str,
            'MASK_ON': False,  # constant!
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
                'FREEZE_FEAT': bool
            },
            'BACKBONE': {
                'FREEZE': bool
            },
            'PROPOSAL_GENERATOR': {
                'FREEZE': bool
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
            'WARMUP_ITERS': int
        },
        'INPUT': {
            'MIN_SIZE_TRAIN': (int,)
        },
        'OUTPUT_DIR': str
    }


def load_yaml_file(fname):
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_cmd(cmd):
    p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    while True:
        line = p.stdout.readline().decode('utf-8')
        if not line:
            break
        print(line)


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


def run_ckpt_surgery(dataset, class_split, src1, method, save_dir, src2=None):
    assert method in ['randinit', 'remove', 'combine'], 'Wrong method: {}'.format(method)
    src2_str = ''
    if method == 'combine':
        assert src2 is not None, 'Need a second source for surgery method \'combine\'!'
        src2_str = '--src2 {}'.format(src2)
    base_command = 'python3 -m tools.ckpt_surgery'  # 'python tools/ckpt_surgery.py' or 'python3 -m tools.ckpt_surgery'
    command = '{} --dataset {} --class-split {} --method {} --src1 {} --save-dir {} {}'\
        .format(base_command, dataset, class_split, method, src1, save_dir, src2_str)
    run_cmd(command)


def get_training_id(layers, mode, shots, fc=False, unfreeze=False, tfa=False, suffix=''):
    # A consistent string used
    #   - as directory name to save checkpoints
    #   - as name for configuration files
    pattern = 'faster_rcnn_R_{}_FPN_ft{}_{}{}{}{}{}'
    fc_str = '_fc' if fc else ''
    unfreeze_str = '_unfreeze' if unfreeze else ''
    tfa_str = '_TFA' if tfa else ''
    shot_str = '_{}shot'.format(shots)
    return pattern.format(layers, fc_str, mode, shot_str, unfreeze_str, tfa_str, suffix)


def get_ft_dataset_names(dataset, class_split, mode, shot, seed, train_split='trainval', test_split='test'):
    # Note: For mode 'all' we evaluate on all classes and would, normally, not need the class split but since we allow
    #  for using different colors for base classes and novel classes, we need the class split to load the correct
    #  mapping of colors to classes
    return (
        '{}_{}_{}_{}_{}shot_seed{}'.format(dataset, class_split, train_split, mode, shot, seed),
        '{}_{}_{}_{}'.format(dataset, class_split, test_split, mode)
    )


# Returns fine-tuning configs. Assumes, that there already exist base-training configs!
# TODO: probably split get_config and doing a surgery? (e.g. if we just want wo iterate over multiple configs to do
#  automated inference?
# TODO: probably think about adding '_cosine' to all cosine fine-tunings. This would be more clear instead of
#  '' being cosine and 'fc' being fc!
def get_config(seed, shot, surgery_method, override_if_exists=False):
    """
    For a given seed and shot, generate a config file based on a template
    config file that is used for training/evaluation.
    You can extend/modify this function to fit your use-case.

    *****Presupposition*****
    Base-Training Checkpoint, stored e.g. at checkpoints/coco/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth

    *****Non-TFA Workflow*****
    ckpt_surgery
    --src1              checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth
    --method            randinit
    --save-dir          checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_all  -> model_reset_surgery.pth

    python3 -m tools.train_net
    --config <config_Kshot>
        -> weights =    checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_all/model_reset_surgery.pth
        -> out-dir =    checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_all_Kshot -> model_final.pth

    *****TFA Workflow*****
    ckpt_surgery
    --src1              checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth
    --method            remove
    --save-dir          checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_novel -> model_reset_remove.pth

    python3 -m tools.train_net
    --config <config_Kshot>
        -> weights =    checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_novel/model_reset_remove.pth
        -> out-dir =    checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_novel_Kshot -> model_final.pth

    ckpt_surgery
    --src1              checkpoints/coco_{}/faster_rcnn/faster_rcnn_R_101_FPN_base/model_final.pth
    --src2              checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_novel_Kshot/model_final.pth
    --method            combine
    --save-dir          checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_novel_Kshot_combine -> model_reset_combine.pth

    python3 -m tools.train_net
        --config <config_Kshot>
        -> weights =    checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_all/model_reset_combine.pth
        -> out-dir =    checkpoints/coco_{}/faster_rcnn/seed{}/faster_rcnn_R_101_FPN_all_Kshot_TFA -> model_final.pth

    Naming conventions:
    - Directory suffix for surgery checkpoints
        - '_all':                   Models to fine-tune entire classifier (=all classes): model_reset_surgery.pth
        - '_novel_Kshot_combine':   Models to fine-tune entire classifier (=all classes): model_reset_combine.pth
        - '_novel':                 Models to fine-tune novel classifier (=just novel classes):model_reset_remove.pth
    - Fine-Tuning directory suffix dependent on approach
        - '':       TFA(===Fine-tuning on 'model_reset_surgery.pth' weights)
        - '_TFA':   Non-TFA(===Fine-tuning on 'model_reset_combine.pth' weights)
    """
    assert surgery_method in ['randinit', 'remove', 'combine'], 'Wrong surgery method: {}'.format(surgery_method)
    if args.dataset == 'coco':
        # COCO
        # (max_iter, (<steps>), checkpoint_period)
        NOVEL_ITERS = {
            1: (500, (10000,), 500),
            2: (1500, (10000,), 500),
            3: (1500, (10000,), 500),
            5: (1500, (10000,), 500),
            10: (2000, (10000,), 500),
            30: (6000, (10000,), 500),
        }  # To fine-tune novel classifier
        ALL_ITERS = {
            1: (16000, (14400,), 1000),  # 1600
            2: (32000, (28800,), 1000),  # 3200
            3: (48000, (43200,), 2000),  # 4800
            5: (80000, (72000,), 4000),  # 8000
            10: (160000, (144000,), 10000),  # 16000
            30: (240000, (216000,), 12000),  # 24000
        }  # To fine-tune entire classifier

        if surgery_method == 'remove':  # fine-tuning only-novel classifier
            ITERS = NOVEL_ITERS
            mode = 'novel'
            # Note: it would normally be no problem to support fc or unfreeze in novel fine-tune but you would have to
            #  create a default config for those cases in order for being able to read example configs to modify
            assert not args.fc and not args.unfreeze
        else:  # either combine only-novel fine-tuning with base training or directly fine-tune entire classifier
            ITERS = ALL_ITERS
            mode = 'all'
        split = temp_split = ''
        temp_mode = mode
        train_split = 'trainval'
        test_split = 'test'
        config_dir = 'configs/COCO-detection/cocosplit_{}'.format(args.class_split)
        ckpt_dir = 'checkpoints/coco_{}/faster_rcnn'.format(args.class_split)
        base_cfg = '../../../../Base-RCNN-FPN.yaml'  # adjust depth to 'config_save_dir'
    elif args.dataset == 'voc':
        # PASCAL VOC
        # Note: we could as well support all types of surgery here, but we do not intend to use PASCAL VOC dataset!
        raise NotImplementedError("Fine-Tuning logic changed! Please refer to "
                                  "COCO-Workflow for reference.")
        assert not args.tfa, 'Only supports random weights for PASCAL now'
        ITERS = {
            1: (3500, 4000),
            2: (7000, 8000),
            3: (10500, 12000),
            5: (17500, 20000),
            10: (35000, 40000),
        }
        split = 'split{}'.format(args.split)
        mode = 'all{}'.format(args.split)
        temp_split = 'split1'
        temp_mode = 'all1'
        train_split = 'trainval'
        test_split = 'test'
        config_dir = 'configs/PascalVOC-detection'
        ckpt_dir = 'checkpoints/voc/faster_rcnn'
        base_cfg = '../../../Base-RCNN-FPN.yaml'
    else:
        raise ValueError("Dataset {} is not supported!".format(args.dataset))

    # Needed to exchange seed and shot in the example config
    seed_str = 'seed{}'.format(seed)  # also used as a directory name
    shot_str = '{}shot'.format(shot)
    # Needed to create appropriate sub directories for the config files
    fc_str = '_fc' if args.fc else ''
    cosine_str = '_cosine' if not args.fc else ''
    unfreeze_str = '_unfreeze' if args.unfreeze else ''
    # sub-directories 'ft_cosine', 'ft_cosine_unfreeze', 'ft_fc', 'ft_(only_)novel' to indicate the type of fine-tuning
    sub_dir_str = 'ft_only_novel' if surgery_method == 'remove' else 'ft' + fc_str + cosine_str + unfreeze_str

    # Set paths depending on surgery method...
    base_ckpt = os.path.join(ckpt_dir, 'faster_rcnn_R_{}_FPN_base'.format(args.layers), 'model_final.pth')
    train_ckpt_base_dir = os.path.join(args.root, ckpt_dir, seed_str)
    os.makedirs(train_ckpt_base_dir, exist_ok=True)
    if surgery_method == 'randinit':
        surgery_ckpt_name = 'model_reset_surgery.pth'
        novel_ft_ckpt = None
        surgery_ckpt_save_dir = os.path.join(ckpt_dir, 'faster_rcnn_R_{}_FPN_all'.format(args.layers))
        training_identifier = get_training_id(layers=args.layers, mode=mode, shots=shot, fc=args.fc,
                                              unfreeze=args.unfreeze, tfa=False, suffix=args.suffix)
    elif surgery_method == 'remove':
        # Note: it would normally be no problem to support fc or unfreeze in novel fine-tune but you would have to
        #  create a default config for those cases in order for being able to read example configs to modify
        assert not args.fc and not args.unfreeze, 'Do not support fc or unfreeze in novel fine-tune!'
        surgery_ckpt_name = 'model_reset_remove.pth'
        novel_ft_ckpt = None
        surgery_ckpt_save_dir = os.path.join(ckpt_dir, 'faster_rcnn_R_{}_FPN_novel'.format(args.layers))
        # Note: we currently have args.tfa set, but we do not yet need it in our directory name
        training_identifier = get_training_id(layers=args.layers, mode=mode, shots=shot, fc=False,
                                              unfreeze=False, tfa=False, suffix=args.suffix)
    else:
        assert surgery_method == 'combine', surgery_method
        surgery_ckpt_name = 'model_reset_combine.pth'
        # Note: novel_ft_ckpt has to match train_ckpt_save_dir of 'remove' surgery!
        # Note: we hard-code the mode to 'novel' because in this phase our actual mode is 'all' but we have to read the
        #  checkpoint of earlier novel fine-tuning whose mode was 'novel'
        novel_ft_ckpt = os.path.join(train_ckpt_base_dir,
                                     get_training_id(layers=args.layers, mode='novel', shots=shot, fc=False,
                                                     unfreeze=False, tfa=False, suffix=args.suffix),
                                     'model_final.pth')
        assert os.path.exists(novel_ft_ckpt), 'Novel weights do not exist!'
        # Note: Here, we also need a seed string in the save directory for the surgery checkpoint, because since we
        #  combine a novel trained checkpoint (which has shot-data and therefore a certain seed), the shot and seed
        #  is imposed by this certain novel training!
        surgery_ckpt_save_dir = os.path.join(ckpt_dir,
                                             seed_str,
                                             'faster_rcnn_R_{}_FPN_novel_{}_combine'.format(args.layers, shot_str))
        training_identifier = get_training_id(layers=args.layers, mode=mode, shots=shot, fc=args.fc,
                                              unfreeze=args.unfreeze, tfa=True, suffix=args.suffix)

    train_weight = surgery_ckpt = os.path.join(surgery_ckpt_save_dir, surgery_ckpt_name)
    train_ckpt_save_dir = os.path.join(train_ckpt_base_dir, training_identifier)

    config_prefix = training_identifier

    # config save dir + save file name
    config_save_dir = os.path.join(args.root, config_dir, split, seed_str, sub_dir_str)
    os.makedirs(config_save_dir, exist_ok=True)
    # config_save_file = os.path.join(config_save_dir, prefix + '.yaml')
    config_save_file = os.path.join(config_save_dir, config_prefix + '.yaml')

    if os.path.exists(config_save_file) and not override_if_exists:
        # If the requested config already exists and we do not want to override it, make sure that the necessary
        #  surgery checkpoints exist and return it
        assert os.path.exists(surgery_ckpt)  # if the config exists, the valid surgery checkpoint has to be existent!
        print("Config already exists, returning the existing config...")
        return config_save_file, load_yaml_file(config_save_file)
    print("Creating a new config file: {}".format(config_save_file))
    # Set all values in the empty config
    new_config = get_empty_ft_config()  # get an empty config and fill it appropriately
    new_config['_BASE_'] = base_cfg

    if args.dataset == 'coco':
        if not os.path.exists(surgery_ckpt):
            # surgery model does not exist, so we have to do a surgery!
            run_ckpt_surgery(dataset='coco', class_split=args.class_split, method=surgery_method,
                             src1=base_ckpt, src2=novel_ft_ckpt, save_dir=surgery_ckpt_save_dir)
            assert os.path.exists(surgery_ckpt)
            print("Saved surgery checkpoint as: {}".format(surgery_ckpt))
        new_config['MODEL']['WEIGHTS'] = train_weight
    elif args.dataset == 'voc':
        new_config['MODEL']['WEIGHTS'] = new_config['MODEL']['WEIGHTS'].replace('base1', 'base{}'.format(args.split))
        for dset in ['TRAIN', 'TEST']:
            new_config['DATASETS'][dset] = (
                new_config['DATASETS'][dset][0].replace(temp_mode, 'all' + str(args.split))
                ,)
    else:
        raise ValueError("Dataset {} is not supported!".format(args.dataset))

    new_config['MODEL']['RESNETS']['DEPTH'] = args.layers
    new_config['MODEL']['ANCHOR_GENERATOR']['SIZES'] = str([[32], [64], [128], [256], [512]])
    new_config['MODEL']['RPN']['PRE_NMS_TOPK_TRAIN'] = 2000  # Per FPN level. TODO: per batch or image?
    new_config['MODEL']['RPN']['PRE_NMS_TOPK_TEST'] = 1000  # Per FPN level. TODO: per batch or image?
    new_config['MODEL']['RPN']['POST_NMS_TOPK_TRAIN'] = 1000  # TODO: per batch or image?
    new_config['MODEL']['RPN']['POST_NMS_TOPK_TEST'] = 1000  # TODO: per batch or image?
    num_novel_classes = len(CLASS_SPLITS[args.dataset][args.class_split]['novel'])
    num_all_classes = len(CLASS_SPLITS[args.dataset][args.class_split]['base']) + num_novel_classes
    new_config['MODEL']['ROI_HEADS'][
        'NUM_CLASSES'] = num_novel_classes if surgery_method == 'remove' else num_all_classes
    print(type(args.unfreeze))
    print(type(not args.unfreeze))
    new_config['MODEL']['ROI_HEADS']['FREEZE_FEAT'] = not args.unfreeze
    new_config['MODEL']['BACKBONE']['FREEZE'] = not args.unfreeze
    new_config['MODEL']['PROPOSAL_GENERATOR']['FREEZE'] = not args.unfreeze
    (train_data, test_data) = get_ft_dataset_names(args.dataset, args.class_split, mode, shot, seed,
                                                   train_split, test_split)
    new_config['DATASETS']['TRAIN'] = str((train_data,))
    new_config['DATASETS']['TEST'] = str((test_data,))
    new_config['SOLVER']['IMS_PER_BATCH'] = args.bs  # default: 16
    lr_scale_factor = args.bs / 16
    new_config['SOLVER']['BASE_LR'] = args.lr if args.lr != -1 else 0.001 * lr_scale_factor
    new_config['SOLVER']['STEPS'] = str(ITERS[shot][1])
    new_config['SOLVER']['MAX_ITER'] = ITERS[shot][0]
    new_config['SOLVER']['CHECKPOINT_PERIOD'] = ITERS[shot][2]  # ITERS[shot][0] // args.ckpt_freq
    new_config['SOLVER']['WARMUP_ITERS'] = 0 if args.unfreeze or surgery_method == 'remove' else 10  # TODO: ???
    new_config['INPUT']['MIN_SIZE_TRAIN'] = str((640, 672, 704, 736, 768, 800))  # scales for multi-scale training
    new_config['OUTPUT_DIR'] = train_ckpt_save_dir

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


def main(args):
    for shot in args.shots:
        for seed in range(args.seeds[0], args.seeds[1]):  # TODO: use second seed arg inclusive?
            print('Split: {}, Seed: {}, Shot: {}'.format(args.split, seed, shot))
            if args.tfa:
                cfg, configs = get_config(seed, shot, surgery_method='remove', override_if_exists=args.override)
                run_exp(cfg, configs)  # TODO: probably just run train(cfg, configs) because evaluation on novel fine-tune might be unnecessary!
                cfg, configs = get_config(seed, shot, surgery_method='combine', override_if_exists=args.override)
                run_exp(cfg, configs)
            else:
                cfg, configs = get_config(seed, shot, surgery_method='randinit', override_if_exists=args.override)
                run_exp(cfg, configs)


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    main(args)
