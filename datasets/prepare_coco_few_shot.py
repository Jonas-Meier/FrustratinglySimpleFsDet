import argparse
import json
import os
import random
import shutil
import time

from class_splits import CLASS_SPLITS
from fsdet.config import get_cfg
cfg = get_cfg()  # get default config to obtain the correct load- and save paths for the created data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=cfg.DATASETS.SUPPORTED_DATASETS, required=True,
                        help="Dataset name")
    parser.add_argument("--class-split", type=str, required=True, dest="class_split",
                        help="Split of classes into base classes and novel classes")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 3, 5, 10, 30],
                        help="Amount of annotations per class for fine tuning")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 9],
                        help="Range of seeds to run. Just a single seed or two seeds representing a range with 2nd "
                             "argument being inclusive as well!")
    parser.add_argument("--no-shuffle", action="store_false", default=True, dest="shuffle",
                        help="Shuffle images prior to sampling of annotations.")
    parser.add_argument("--override", action="store_true", default=False, dest="override",
                        help="Force deleting already existent data.")
    args = parser.parse_args()
    return args


def get_data_path():  # get path to training data annotations
    # Note: either set cfg.TRAIN_ANNOS correctly or return a different value, depending on args.dataset
    return os.path.join(cfg.ROOT_DIR, cfg.TRAIN_ANNOS[args.dataset])


def generate_seeds(args):
    start = time.time()
    data_path = get_data_path()
    data = json.load(open(data_path))

    new_all_cats = []  # category "objects"
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i
    # same but shorter: id2img = {i['id']: i for i in data['images']}

    # tuples of category names
    # TODO: base- and novel classes do not matter when sampling few-shot data, but may be important when saving them!
    base_classes = tuple(CLASS_SPLITS[args.dataset][args.class_split]['base'])
    novel_classes = tuple(CLASS_SPLITS[args.dataset][args.class_split]['novel'])
    all_classes = tuple(base_classes + novel_classes)

    coco_cat_id_to_name = {c['id']: c['name'] for c in new_all_cats}
    # Need make sure, 'all_classes' are all contained in 'coco_cat_id_to_name'
    assert len(all_classes) <= len(coco_cat_id_to_name) \
           and len(set(all_classes + tuple(coco_cat_id_to_name.values()))) == len(coco_cat_id_to_name), \
           "Error, inconsistency with categories defined in the dataset and in the class split: {} and {}".\
           format(coco_cat_id_to_name.values(), all_classes)

    cat_name_to_annos = {i: [] for i in all_classes}
    for anno in data['annotations']:
        if anno['iscrowd'] == 1:
            continue
        cat_name = coco_cat_id_to_name[anno['category_id']]
        if cat_name not in cat_name_to_annos:  # if base and novel classes do not sum up to all classes in the dataset
            continue
        else:
            cat_name_to_annos[cat_name].append(anno)

    if len(args.seeds) == 1:
        seeds = [args.seeds[0]]
    else:
        assert len(args.seeds) == 2
        seeds = range(args.seeds[0], args.seeds[1] + 1)
    shots = args.shots
    for i in seeds:
        print("Generating seed {}".format(i))
        save_dir = os.path.join(
            cfg.DATA_SAVE_PATH_PATTERN[args.dataset].format(args.class_split),
            'seed{}'.format(i)
        )

        if os.path.exists(save_dir):
            if args.override:
                print("Cleaning save path directory '{}'".format(save_dir))
                shutil.rmtree(save_dir)
                os.mkdir(save_dir)
            else:
                train_split = cfg.TRAIN_SPLIT[args.dataset]
                # Note: the "incompatible" annotation file originates from 'tools/examine_sampling_strategies.py' which
                #  creates just a single annotation file containing annotations for all classes rather than this script
                #  which creates a single annotation file per class. Since 'fsdet/data/meta_coco.py' searches for
                #  this "incompatible" annotation file, both types may not exist in the same directory. For that reason,
                #  we expect it to not being present (or deleted first) prior to creating annotation files with this
                #  script!
                incomp_anno_pattern = os.path.join(save_dir, "full_box_{}shot_{}.json")
                cls_anno_pattern = os.path.join(save_dir, "full_box_{}shot_{}_{}.json")
                abort = False
                if any(os.path.exists(incomp_anno_pattern.format(shot, train_split)) for shot in shots):
                    print("Error: An incompatible annotation file '{}' already exists for a given target shot {}. "
                          "Set '--override' to delete this directory!".format(incomp_anno_pattern, shots))
                    abort = True
                if any(os.path.exists(cls_anno_pattern.format(shot, cat, train_split)) for shot in shots for cat in all_classes):
                    print("Error: Some data matching the file name '{}' would be overridden by this call for the"
                          "given shots {} and categories {}. Adjust the arguments or set '--override' to delete this "
                          "directory!".format(cls_anno_pattern, shots, all_classes))
                    abort = True
                if abort:
                    exit(1)
                del train_split
                del incomp_anno_pattern
                del cls_anno_pattern
        os.makedirs(save_dir, exist_ok=True)

        for cat_name in all_classes:
            print("Generating data for class {}".format(cat_name))
            img_id_to_annos = {}
            for anno in cat_name_to_annos[cat_name]:
                if anno['image_id'] in img_id_to_annos:
                    img_id_to_annos[anno['image_id']].append(anno)
                else:
                    img_id_to_annos[anno['image_id']] = [anno]

            for shot in shots:
                sample_annos = []  # annotations
                sample_imgs = []  # images
                sample_img_ids = []  # ids of sampled images, just used for duplicate checks
                if cat_name in base_classes:
                    assert cat_name not in novel_classes
                    if cfg.BASE_SHOT_MULTIPLIER == -1:
                        target_shot = len(cat_name_to_annos[cat_name])  # should be all available annos
                        print("Using all available {} annotations for base class {}!"
                              .format(target_shot, cat_name))
                    else:
                        assert cfg.BASE_SHOT_MULTIPLIER > 0
                        target_shot = cfg.BASE_SHOT_MULTIPLIER * shot
                        print("Generating {}x{} shot data for base class {}"
                              .format(cfg.BASE_SHOT_MULTIPLIER, shot, cat_name))
                else:
                    assert cat_name in novel_classes
                    target_shot = shot
                    print("Generating {} shot data for novel class {}"
                          .format(shot, cat_name))
                img_ids = list(img_id_to_annos.keys())
                # while True:
                    # img_ids = random.sample(list(img_id_to_annos.keys()), shot)
                # TODO: probably use random.sample(img_ids, 1) in a 'while True'-loop?
                if args.shuffle:
                    shuffle_seed = i  # Same order for same seeds, but should not matter...
                    random.seed(shuffle_seed)
                    print("shuffling images")
                    random.shuffle(img_ids)
                else:
                    print("not shuffling images prior to sampling!")
                for img_id in img_ids:
                    if img_id in sample_img_ids:  # only necessary if we iterate multiple times through all images
                        continue
                    if len(img_id_to_annos[img_id]) + len(sample_annos) > target_shot:
                        # TODO: This condition may lead to following:
                        #  1. For k=5 shot and if each image had exactly 2 annotations per class we finally only
                        #  have four annotations for that class -> probably too few annotations
                        #  2. In contrast to other approaches, they allow for taking multiple annotations from the
                        #  same image (even more: they only want ALL annotations from an image (for a certain class)
                        #  or none at all) (as support data) -> unknown consequences
                        continue
                    sample_annos.extend(img_id_to_annos[img_id])  # add all annotations of image with id 'img_id' with class 'c'
                    sample_imgs.append(id2img[img_id])  # add the image with id 'img_id'
                    sample_img_ids.append(img_id)
                    assert len(sample_imgs) <= len(sample_annos), \
                        "Error, got {} images but only {} annotations!".format(len(sample_imgs), len(sample_annos))
                    if len(sample_annos) == target_shot:
                        break
                # TODO: Probably convert assertion to a warning.
                assert len(sample_annos) == target_shot, "Wanted {} shot, but only found {} annotations!"\
                    .format(target_shot, len(sample_annos))
                new_data = data.copy()
                new_data['images'] = sample_imgs
                new_data['annotations'] = sample_annos
                new_data['categories'] = new_all_cats

                # Note: even if we sample more annotations for base classes we use the original 'shot' in the file
                # name for clarity!
                save_file = 'full_box_{}shot_{}_{}.json'.format(shot, cat_name, cfg.TRAIN_SPLIT[args.dataset])
                save_path = os.path.join(save_dir, save_file)
                with open(save_path, 'w') as f:
                    # json.dump(new_data, f)
                    json.dump(new_data, f, indent=2)  # Easier to check files manually
    end = time.time()
    m, s = divmod(int(end-start), 60)
    print("Created few-shot data for {} shots and {} seeds in {}m {}s"
          .format(len(args.shots), len(seeds), m, s))


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    generate_seeds(args)
