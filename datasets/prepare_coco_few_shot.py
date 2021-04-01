import argparse
import json
import os
import random
import time

import sys
sys.path.append('..')  # TODO: ugly but works for now
print("Path: {}".format(sys.path))
from class_splits import CLASS_SPLITS
from fsdet.config import get_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["coco"], required=True,
                        help="Dataset name")
    parser.add_argument("--class-split", type=str, required=True, dest="class_split",
                        help="Split of classes into base classes and novel classes")
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 2, 3, 5, 10, 30],
                        help="Amount of annotations per class for fine tuning")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 9],
                        help="Range of seeds. Start and end are both inclusive!")
    args = parser.parse_args()
    return args


def get_data_path():
    # probably use cfg.DATA_DIR[args.dataset] if necessary
    if args.dataset == "coco":
        # TODO: replace this hackish way to get the correct path!
        return os.path.join(cfg.ROOT_DIR, "datasets", "cocosplit", "datasplit", "trainvalno5k.json")
    else:
        raise ValueError("Dataset {} is not supported!".format(args.dataset))


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

    # tuples of category names
    # TODO: base- and novel classes do not matter when sampling few-shot data, but may be important when saving them!
    base_classes = tuple(CLASS_SPLITS[args.dataset][args.class_split]['base'])
    novel_classes = tuple(CLASS_SPLITS[args.dataset][args.class_split]['novel'])
    all_classes = tuple(base_classes + novel_classes)

    coco_cat_id_to_name = {c['id']: c['name'] for c in new_all_cats}
    # Need make sure, 'all_classes' and 'coco_cat_id_to_name' contain the same categories! This should be sufficient
    assert len(all_classes) == len(coco_cat_id_to_name) == len(set(all_classes + tuple(coco_cat_id_to_name.values()))), \
        "Error, inconsistency with categories defined in the dataset and in the class split: {} and {}". \
        format(coco_cat_id_to_name.values(), all_classes)

    cat_name_to_annos = {i: [] for i in all_classes}
    for anno in data['annotations']:
        if anno['iscrowd'] == 1:
            continue
        cat_name = coco_cat_id_to_name[anno['category_id']]
        cat_name_to_annos[cat_name].append(anno)

    seeds = range(args.seeds[0], args.seeds[1] + 1)  # we use start and end inclusively!
    for i in seeds:
        print("Generating seed {}".format(i))
        random.seed(i)
        for cat_name in all_classes:
            print("Generating data for class {}".format(cat_name))
            img_id_to_annos = {}
            for anno in cat_name_to_annos[cat_name]:
                if anno['image_id'] in img_id_to_annos:
                    img_id_to_annos[anno['image_id']].append(anno)
                else:
                    img_id_to_annos[anno['image_id']] = [anno]

            sample_annos = []  # annotations
            sample_imgs = []  # images
            for shots in args.shots:
                print("Generating {} shot data".format(shots))
                img_ids = list(img_id_to_annos.keys())
                random.shuffle(img_ids)  # TODO: probably use random.sample(img_ids, 1) in a loop?
                # while True:
                    # img_ids = random.sample(list(img_id_to_annos.keys()), shots)
                for img_id in img_ids:
                    skip = False
                    for s in sample_annos:
                        if img_id == s['image_id']:  # TODO: only necessary if we iterate multiple times through all images
                            skip = True
                            break
                    if skip:
                        continue
                    if len(img_id_to_annos[img_id]) + len(sample_annos) > shots:  # TODO: This condition may lead to following:
                        # 1. For k=5 shots and if each image had exactly 2 annotations per class we finally only
                        # have four annotations for that class -> probably too few annotations
                        # 2. In contrast to other approaches, they allow for taking multiple annotations from the
                        # same image (even more: they only want ALL annotations from an image (for a certain class)
                        # or none at all) (as support data) -> unknown consequences
                        continue
                    sample_annos.extend(img_id_to_annos[img_id])  # add all annotations of image with id 'img_id' with class 'c'
                    sample_imgs.append(id2img[img_id])  # add the image with id 'img_id'
                    assert len(sample_imgs) <= len(sample_annos), \
                        "Error, got {} images but only {} annotations!".format(len(sample_imgs), len(sample_annos))
                    if len(sample_annos) == shots:
                        break
                # TODO: Probably convert assertion to a warning.
                assert len(sample_annos) == shots, "Wanted {} shots, but only found {} annotations!".format(shots, len(
                    sample_annos))
                new_data = {
                    'info': data['info'],
                    'licenses': data['licenses'],
                    'images': sample_imgs,
                    'annotations': sample_annos,
                }
                save_path = get_save_path_seeds(data_path, cat_name, shots, i)
                new_data['categories'] = new_all_cats
                with open(save_path, 'w') as f:
                    # json.dump(new_data, f)
                    json.dump(new_data, f, indent=2)  # Easier to check files manually
    end = time.time()
    m, s = divmod(int(end-start), 60)
    print("Created few-shot data for {} shots and {} seeds in {}m {}s"
          .format(len(args.shots), len(seeds), m, s))


def get_save_path_seeds(path, cls, shots, seed):
    s = path.split('/')
    prefix = 'full_box_{}shot_{}_trainval'.format(shots, cls)
    save_dir = os.path.join(cfg.DATA_SAVE_PATH_PATTERN[args.dataset].format(args.class_split), 'seed{}'.format(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    cfg = get_cfg()  # get default config to obtain the correct load- and save paths for the created data
    generate_seeds(args)
