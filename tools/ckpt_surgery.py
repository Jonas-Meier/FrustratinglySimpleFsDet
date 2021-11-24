import numpy
# Note: when calling this script from within another script, we need this numpy import before the torch
# import, see: https://github.com/pytorch/pytorch/issues/37377
import torch

import argparse
import os
from fsdet.config import get_cfg
from class_splits import CLASS_SPLITS, get_ids_from_names, ALL_CLASSES, COMPATIBLE_DATASETS, CLASS_NAME_TRANSFORMS
cfg = get_cfg()


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, default='',
                        help='Path to the main checkpoint')
    parser.add_argument('--src2', type=str, default='',
                        help='Path to the secondary checkpoint (for combining)')
    parser.add_argument('--save-dir', type=str, default='',
                        help='Save directory')
    # Surgery method
    parser.add_argument('--method', choices=['combine', 'remove', 'randinit'],
                        required=True,
                        help='Surgery method. combine = '
                             'combine checkpoints. remove = for fine-tuning on '
                             'novel dataset, remove the final layer of the '
                             'base detector. randinit = randomly initialize '
                             'novel weights.')
    parser.add_argument('--discard-base-weights', action='store_false', dest='keep_base_weights', default=True,
                        help='Specify to discard the base class predictor weights, obtained from base training.')
    parser.add_argument('--discard-bg-weights', action='store_false', dest='keep_background_weights', default=True,
                        help='Specify to discard the weights of the background class, obtained from base training.')
    # Targets
    parser.add_argument('--param-name', type=str, nargs='+',
                        default=['roi_heads.box_predictor.cls_score',
                                 'roi_heads.box_predictor.bbox_pred'],
                        help='Target parameter names')
    parser.add_argument('--tar-name', type=str, default='model_reset',
                        help='Name of the new ckpt')
    parser.add_argument('--double-head', action='store_true', default=False,
                        help="use different heads for base classes and novel classes")
    # Dataset
    parser.add_argument('--dataset', choices=cfg.DATASETS.SUPPORTED_DATASETS,
                        required=True, help='dataset')
    parser.add_argument('--class-split', dest='class_split',  required=True,
                        help='Class split of the dataset into base classes and novel classes')
    parser.add_argument('--alt-dataset', dest='alt_dataset', choices=cfg.DATASETS.SUPPORTED_DATASETS, default='',
                        help='alternative dataset to fine-tune on. In this case, --dataset is used to correctly map '
                             'the predictor weights from the base training model to the target model for fine-tuning.')
    parser.add_argument('--alt-class-split', dest='alt_class_split', default='',
                        help='alternative class split of the --alternative-dataset')
    args = parser.parse_args()
    return args


def ckpt_surgery(args):
    """
    Either remove the final layer weights for fine-tuning on novel dataset or
    append randomly initialized weights for the novel classes.

    Note: The base detector for LVIS contains weights for all classes, but only
    the weights corresponding to base classes are updated during base training
    (this design choice has no particular reason). Thus, the random
    initialization step is not really necessary.
    """
    # Note: this method does not handle the "remove" surgery at all, even if one could think it would, according to the
    # docstings! "remove"-surgery is done by 'surgery_loop'-method, independent on the passed 'surgery'-argument
    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2=None):
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['model'][weight_name]
        prev_cls = pretrained_weight.size(0)
        if 'cls_score' in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
            # Init with 0.01 for cls and 0.001 for bbox (see fast_rcnn:FastRCNNOutputLayers (l.358))
            if 'cls_score' in param_name:
                assert 'bbox_pred' not in param_name
                torch.nn.init.normal_(new_weight, 0, 0.01)
            else:
                assert 'bbox_pred' in param_name
                torch.nn.init.normal_(new_weight, 0, 0.001)
        else:
            new_weight = torch.zeros(tar_size)
        if args.dataset == 'voc':
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]
        else:  # coco, lvis, isaid, etc. (all datasets with idmaps)
            if args.keep_base_weights:
                for i, c in enumerate(BASE_CLASS_IDS):
                    idx = c if args.dataset == 'lvis' else i
                    src_ind = tar_to_src_base_class_ind[idx]
                    tar_ind = ALL_CLASS_ID_TO_IND[c]
                    if 'cls_score' in param_name:
                        new_weight[tar_ind] = pretrained_weight[src_ind]
                    else:
                        new_weight[tar_ind * 4:(tar_ind + 1) * 4] = \
                            pretrained_weight[src_ind*4:(src_ind+1)*4]
        if 'cls_score' in param_name and args.keep_background_weights:
            new_weight[-1] = pretrained_weight[-1]  # bg class
        ckpt['model'][weight_name] = new_weight

    def double_head_surgery(param_name, is_weight, tar_size, ckpt, ckpt2=None):
        del tar_size  # we use different target sizes for base classes and novel classes
        # Special kind of surgery for the experimental double head. For simplicity reasons, it is just supported along with
        #  'randinit' and for may only be used with coco-like datasets
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['model'][weight_name]
        base_tar_size = len(BASE_CLASS_IDS)
        novel_tar_size = len(NOVEL_CLASS_IDS)
        if "cls_score" in param_name:  # +1 for background class
            base_tar_size += 1
            novel_tar_size += 1
        else:  # *4 for bboxes, no bbox parameters for background class necessary
            assert "bbox_pred" in param_name
            base_tar_size *= 4
            novel_tar_size *= 4
        assert pretrained_weight.size(0) == base_tar_size

        if is_weight:
            # old base class predictor's feature size should be the same size we want for novel class predictor as well
            feat_size = pretrained_weight.size(1)
            novel_weights = torch.rand((novel_tar_size, feat_size))
            # Init with 0.01 for cls and 0.001 for bbox (see fast_rcnn:FastRCNNOutputLayers (l.358))
            if 'cls_score' in param_name:
                assert 'bbox_pred' not in param_name
                torch.nn.init.normal_(novel_weights, 0, 0.01)
            else:
                assert 'bbox_pred' in param_name
                torch.nn.init.normal_(novel_weights, 0, 0.001)
        else:
            novel_weights = torch.zeros(novel_tar_size)
        assert args.dataset not in ['voc', 'lvis'], \
            "Double-Head predictor currently not supported for dataset {}".format(args.dataset)
        ckpt['model'][weight_name.replace("box_predictor", "box_predictor1")] = pretrained_weight  # copy old base weights to new base predictor weights
        ckpt['model'][weight_name.replace("box_predictor", "box_predictor2")] = novel_weights  # add new weights for novel class predictor
        del ckpt['model'][weight_name]  # delete old base predictor weights

        # duplicate FC2 layers for fine-tuning!
        fc2_weight_name = "roi_heads.box_head.fc2.weight"
        fc2_bias_name = "roi_heads.box_head.fc2.bias"
        if is_weight and fc2_weight_name in ckpt['model']:
            # duplicate fc2 weight
            ckpt['model']['roi_heads.box_head.fc2:1.weight'] = ckpt['model'][fc2_weight_name]
            ckpt['model']['roi_heads.box_head.fc2:2.weight'] = ckpt['model'][fc2_weight_name]
            # remove old weights
            del ckpt['model'][fc2_weight_name]
        elif not is_weight and fc2_bias_name in ckpt['model']:
            # duplicate fc2 bias
            ckpt['model']['roi_heads.box_head.fc2:1.bias'] = ckpt['model'][fc2_bias_name]
            ckpt['model']['roi_heads.box_head.fc2:2.bias'] = ckpt['model'][fc2_bias_name]
            # remove old bias
            del ckpt['model'][fc2_bias_name]

    if not args.double_head:
        surgery_loop(args, surgery)
    else:
        surgery_loop(args, double_head_surgery)


def combine_ckpts(args):
    """
    Combine base detector with novel detector. Feature extractor weights are
    from the base detector. Only the final layer weights are combined.
    """
    def surgery(param_name, is_weight, tar_size, ckpt, ckpt2=None):
        if not is_weight and param_name + '.bias' not in ckpt['model']:
            return
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['model'][weight_name]
        prev_cls = pretrained_weight.size(0)
        if 'cls_score' in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
        else:
            new_weight = torch.zeros(tar_size)
        if args.dataset == 'voc':
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]
        else:  # coco, lvis, isaid, etc. (all datasets with idmaps)
            for i, c in enumerate(BASE_CLASS_IDS):
                idx = c if args.dataset == 'lvis' else i
                if 'cls_score' in param_name:
                    new_weight[ALL_CLASS_ID_TO_IND[c]] = pretrained_weight[idx]
                else:
                    new_weight[ALL_CLASS_ID_TO_IND[c] * 4:(ALL_CLASS_ID_TO_IND[c] + 1) * 4] = \
                        pretrained_weight[idx * 4:(idx + 1) * 4]
        ckpt2_weight = ckpt2['model'][weight_name]

        if args.dataset == 'voc':
            if 'cls_score' in param_name:
                new_weight[prev_cls:-1] = ckpt2_weight[:-1]
                new_weight[-1] = pretrained_weight[-1]
            else:
                new_weight[prev_cls:] = ckpt2_weight
        else:  # coco, lvis, isaid, etc. (all datasets with idmaps)
            for i, c in enumerate(NOVEL_CLASS_IDS):
                if 'cls_score' in param_name:
                    new_weight[ALL_CLASS_ID_TO_IND[c]] = ckpt2_weight[i]
                else:
                    new_weight[ALL_CLASS_ID_TO_IND[c] * 4:(ALL_CLASS_ID_TO_IND[c] + 1) * 4] = \
                        ckpt2_weight[i * 4:(i + 1) * 4]
            if 'cls_score' in param_name:
                new_weight[-1] = pretrained_weight[-1]
        ckpt['model'][weight_name] = new_weight

    surgery_loop(args, surgery)


def surgery_loop(args, surgery):
    # Load checkpoints
    ckpt = torch.load(args.src1)
    if args.method == 'combine':
        ckpt2 = torch.load(args.src2)
        save_name = args.tar_name + '_combine.pth'
    else:
        ckpt2 = None
        save_name = args.tar_name + '_' + \
            ('remove' if args.method == 'remove' else 'surgery') + '.pth'
    if args.save_dir == '':
        # By default, save to directory of src1
        save_dir = os.path.dirname(args.src1)
    else:
        save_dir = args.save_dir
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_ckpt(ckpt)

    # Remove parameters
    if args.method == 'remove':
        for param_name in args.param_name:
            del ckpt['model'][param_name + '.weight']
            if param_name+'.bias' in ckpt['model']:
                del ckpt['model'][param_name+'.bias']
        save_ckpt(ckpt, save_path)
        return

    # Surgery
    tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
    for idx, (param_name, tar_size) in enumerate(zip(args.param_name,
                                                     tar_sizes)):
        surgery(param_name, True, tar_size, ckpt, ckpt2)
        surgery(param_name, False, tar_size, ckpt, ckpt2)

    # Save to file
    save_ckpt(ckpt, save_path)


def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print('save changed ckpt to {}'.format(save_name))


def reset_ckpt(ckpt):
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0


if __name__ == '__main__':
    args = parse_args()
    print("Called with args:")
    print(args)
    # Note:
    #  - TOTAL_CLASSES is just used for sanity checks
    #  - sorting of novel class ids not necessary (or just necessary for 'combine' surgery), but sorting of base class
    #      ids and all class ids is important!
    assert (args.alt_dataset and args.alt_class_split) or (not args.alt_dataset and not args.alt_class_split)
    if args.alt_dataset:
        # We do not allow 'combine' surgery. (Problematic code parts: access to NOVEL_CLASS_IDS in method combine_ckpt)
        assert args.method != 'combine', "alternative dataset is currently not supported for combine surgeries. " \
                                         "Correctly mapping the novel categories as well would be required for that!"
        if args.dataset == args.alt_dataset or \
                any(args.dataset in comp_dsets and args.alt_dataset in comp_dsets for comp_dsets in COMPATIBLE_DATASETS):
            src_base_classes = CLASS_SPLITS[args.dataset][args.class_split]['base']
            tar_base_classes = CLASS_SPLITS[args.alt_dataset][args.alt_class_split]['base']
            assert set(src_base_classes) == set(tar_base_classes)
            TOTAL_CLASSES = len(ALL_CLASSES[args.alt_dataset])
            BASE_CLASS_IDS = sorted(get_ids_from_names(args.alt_dataset, tar_base_classes))
            NOVEL_CLASS_IDS = sorted(get_ids_from_names(args.alt_dataset, CLASS_SPLITS[args.alt_dataset][args.alt_class_split]['novel']))
            tar_to_src_base_class_ind = {i: i for i in range(len(BASE_CLASS_IDS))}  # dummy-identity-map
        else:
            # very ugly case where the alternative probably has different class names and ids and we now have to
            #  correctly map the base trained weights to the corresponding base class weights of the alternative dataset
            # TODO: probably we just need to make sure that the base classes are compatible. Novel classes probably
            #  don't matter at all!
            assert ((args.alt_dataset, args.alt_class_split), (args.dataset, args.class_split)) in CLASS_NAME_TRANSFORMS
            src_base_classes = CLASS_SPLITS[args.dataset][args.class_split]['base']
            tar_base_classes = CLASS_SPLITS[args.alt_dataset][args.alt_class_split]['base']
            src_base_class_ids = sorted(get_ids_from_names(args.dataset, src_base_classes))
            tar_base_class_ids = sorted(get_ids_from_names(args.alt_dataset, tar_base_classes))
            src_id_to_ind = {src_id: ind for ind, src_id in enumerate(src_base_class_ids)}
            tar_id_to_ind = {tar_id: ind for ind, tar_id in enumerate(tar_base_class_ids)}
            tar_classes_to_src_classes = CLASS_NAME_TRANSFORMS[((args.alt_dataset, args.alt_class_split), (args.dataset, args.class_split))]
            tar_to_src_base_classes = {tar_cls: src_cls
                                       for tar_cls, src_cls in tar_classes_to_src_classes.items()
                                       if tar_cls in tar_base_classes and src_cls in src_base_classes}
            assert set(tar_to_src_base_classes.keys()) == set(tar_base_classes)
            assert set(tar_to_src_base_classes.values()) == set(src_base_classes)
            tar_to_src_base_id = {get_ids_from_names(args.alt_dataset, tar_cls): get_ids_from_names(args.dataset, src_cls)
                                  for tar_cls, src_cls in tar_to_src_base_classes.items()}
            tar_to_src_base_class_ind = {tar_id_to_ind[tar_id]: src_id_to_ind[src_id]
                                         for tar_id, src_id in tar_to_src_base_id.items()}
            TOTAL_CLASSES = len(ALL_CLASSES[args.alt_dataset])
            BASE_CLASS_IDS = tar_base_class_ids
            NOVEL_CLASS_IDS = sorted(get_ids_from_names(args.alt_dataset, CLASS_SPLITS[args.alt_dataset][args.alt_class_split]['novel']))
    else:
        TOTAL_CLASSES = len(ALL_CLASSES[args.dataset])
        BASE_CLASS_IDS = sorted(get_ids_from_names(args.dataset, CLASS_SPLITS[args.dataset][args.class_split]['base']))
        NOVEL_CLASS_IDS = sorted(get_ids_from_names(args.dataset, CLASS_SPLITS[args.dataset][args.class_split]['novel']))

        tar_to_src_base_class_ind = {i: i for i in range(len(BASE_CLASS_IDS))}  # dummy-identity-map (since tar==src)
    ALL_CLASS_IDS = sorted(BASE_CLASS_IDS + NOVEL_CLASS_IDS)
    ALL_CLASS_ID_TO_IND = {v: i for i, v in enumerate(ALL_CLASS_IDS)}
    TAR_SIZE = len(ALL_CLASS_IDS)
    if TAR_SIZE != TOTAL_CLASSES:
        print("Warning: Base and novel classes add up to just {} of {} total classes!"
              .format(TAR_SIZE, TOTAL_CLASSES))

    if args.method == 'combine':
        combine_ckpts(args)
    else:
        ckpt_surgery(args)
