import json
import os
import random
import shutil
import time
from multiprocessing import Pool
import threading
from statistics import mean, stdev

from class_splits import CLASS_SPLITS
from fsdet.config.config import get_cfg

"""
This script is a supplemental script for generating few-shot data. The main script for generating such data 
(datasets/prepare_coco_few_shot.py) only supports the sampling strategy to sample 
"all annotations of a single class or none" per image. This sampling strategy is necessary in order for the 
Base Shot Multiplier (BSM) and the Novel-class Oversampling Factor (NOF) to work correctly.
For this reason, another script is used to examine other sampling strategies. Which will cause the BSM and NOF to no 
longer work.
Another reason for the existence of this script is that i.e. meta-learning approaches often need to create support data
for the meta branch, that is i.e. the resizing and cutting of images, which makes the overall few-shot data creation 
slow. This script allows for quickly sampling many sets of images and annotations, analysing them and exporting the 
ones with the desired properties.
"""

cfg = get_cfg()

mean_decimals = 1
std_decimals = 2

dataset = "isaid"
class_split = "experiment3"
anno_dir = cfg.TRAIN_ANNOS[dataset]
save_dir_base_path = cfg.DATA_SAVE_PATH_PATTERN[dataset].format(class_split)
meta_dir = ""  # TODO: set appropriately!

base_class_names = tuple(CLASS_SPLITS[dataset][class_split]['base'])
novel_class_names = tuple(CLASS_SPLITS[dataset][class_split]['novel'])
all_class_names = tuple(base_class_names + novel_class_names)

force_override = False  # Force overriding of already existing datasets


def main():
    # Valid strategies: 'one', 'any', 'all_or_none', 'all_per_class_or_none'
    if force_override:
        print("Warning: Overriding of already existing datasets is activated!")
    shots = 100
    # Sample images with many and few annotations per image (fix strategy: 'all_or_none')
    _sample_high_annotation_ratio_per_image_and_export(shots=shots, pool_size=1000, seed_range=[0, 4], anno_count_min=None)
    _sample_low_annotation_ratio_per_image_and_export(shots=shots, pool_size=1000, seed_range=[5, 9], anno_count_max=None)
    # For comparison, sample data with two different sampling strategies
    _sample_and_export(strategy='one', shots=shots, seed_range=[10, 14])
    _sample_and_export(strategy='any', shots=shots, seed_range=[15, 19])
    _sample_and_export(strategy='all_per_class_or_none', shots=shots, seed_range=[20, 24])
    #_get_image_count(anno_count_min=10, anno_count_max=50)
    #data = _read_sample(seed=0, shots=shots)
    #analyse_sample(data["images"], data["annotations"], img_to_value=None)


def _sample_high_annotation_ratio_per_image_and_export(shots=100, pool_size=10000, seed_range=[0, 4],
                                                       anno_count_min=None):
    """
    Samples few images with many annotations.
    """
    assert len(seed_range) in [1, 2]
    if len(seed_range) == 2:
        seed_range = list(range(seed_range[0], seed_range[1] + 1))
    top_k = len(seed_range)
    samples = []
    start = time.time()
    for _ in range(pool_size):
        imgs, anns = sample(strategy='all_or_none', shots=shots, shuffle=True, silent=True,
                            anno_count_min=anno_count_min, anno_count_max=None)
        samples.append((imgs, anns))
    end = time.time()

    topk_imgs_samples = _get_topk_samples(samples, top_k, 'no_imgs')
    topk_anns_samples = _get_topk_samples(samples, top_k, 'no_anns', reverse=True)
    topk_ratio_samples = _get_topk_samples(samples, top_k, 'ratio_ann_img', reverse=True)

    print("Top {} samples out of a pool with {} samples (ran in {}m {}s)"
          .format(top_k, pool_size, *divmod(int(end-start), 60)))
    print("With fewest images \t With most annotations \t With most annotations per image")
    for seed, (imgs_topk_imgs, anns_topk_imgs), (imgs_topk_anns, anns_topk_anns), (imgs_topk_ratio, anns_topk_ratio) \
            in zip(seed_range, topk_imgs_samples, topk_anns_samples, topk_ratio_samples):
        print("#imgs: {}, #anns: {} \t #imgs: {}, #anns: {} \t #imgs: {}, #anns: {}"
              .format(len(imgs_topk_imgs), len(anns_topk_imgs),
                      len(imgs_topk_anns), len(anns_topk_anns),
                      len(imgs_topk_ratio), len(anns_topk_ratio)))
        imgs, anns = imgs_topk_ratio, anns_topk_ratio
        _export_sample(imgs, anns, file_dir=_get_file_dir(seed), file_name=_get_file_name(shots))


def _sample_low_annotation_ratio_per_image_and_export(shots=100, pool_size=10000, seed_range=[0, 4],
                                                      anno_count_max=None):
    """
    Samples many images with few annotations.
    """
    assert len(seed_range) in [1, 2]
    if len(seed_range) == 2:
        seed_range = list(range(seed_range[0], seed_range[1] + 1))
    top_k = len(seed_range)
    samples = []
    start = time.time()
    for _ in range(pool_size):
        imgs, anns = sample(strategy='all_or_none', shots=shots, shuffle=True, silent=True,
                            anno_count_min=None, anno_count_max=anno_count_max)
        samples.append((imgs, anns))
    end = time.time()
    topk_imgs_samples = _get_topk_samples(samples, top_k, 'no_imgs', reverse=True)
    topk_anns_samples = _get_topk_samples(samples, top_k, 'no_anns', reverse=True)
    topk_ratio_samples = _get_topk_samples(samples, top_k, 'ratio_ann_img')

    print("Top {} samples out of a pool with {} samples (ran in {}m {}s)"
          .format(top_k, pool_size, *divmod(int(end - start), 60)))
    print("With most images \t With most annotations \t With fewest annotations per image")
    for seed, (imgs_topk_imgs, anns_topk_imgs), (imgs_topk_anns, anns_topk_anns), (imgs_topk_ratio, anns_topk_ratio) \
            in zip(seed_range, topk_imgs_samples, topk_anns_samples, topk_ratio_samples):
        print("#imgs: {}, #anns: {} \t #imgs: {}, #anns: {} \t #imgs: {}, #anns: {}"
              .format(len(imgs_topk_imgs), len(anns_topk_imgs),
                      len(imgs_topk_anns), len(anns_topk_anns),
                      len(imgs_topk_ratio), len(anns_topk_ratio)))
        imgs, anns = imgs_topk_ratio, anns_topk_ratio
        _export_sample(imgs, anns, file_dir=_get_file_dir(seed), file_name=_get_file_name(shots))


def _sample_and_export(strategy, shots, seed_range):
    print("Generating samples for strategy {}, {} shots and seed range {}".format(strategy, shots, seed_range))
    assert len(seed_range) in [1, 2]
    if len(seed_range) == 2:
        seed_range = list(range(seed_range[0], seed_range[1] + 1))
    for seed in seed_range:
        imgs, anns = sample(strategy=strategy, shots=shots, shuffle=True, silent=True,
                            anno_count_min=None, anno_count_max=None)
        print("#imgs: {}, #anns: {}".format(len(imgs), len(anns)))
        # analyse_sample(imgs, anns)
        _export_sample(imgs, anns, file_dir=_get_file_dir(seed), file_name=_get_file_name(shots))


def sample(strategy, shots, shuffle=True, silent=False, anno_count_min=None, anno_count_max=None):
    """
    anno_count_min and anno_count_max will only affect the strategies 'all_or_none' and 'all_per_class_or_none'
    """
    # Note: no checks are necessary to prevent sampling annotations from the same image multiple times, because we
    #  iterate over the whole dataset just once
    # TODO: support sampling (and saving) data class-wise for strategies 'one' and 'all_per_class_or_none'?
    #  -> if yes, we probably don't need the script 'prepare_coco_few_shot' anymore...
    #  (class-wise sampling could make sense since it then allows for using BSM + NOF...)
    assert strategy in ['one', 'any', 'all_or_none', 'all_per_class_or_none']
    cls_to_sampled_anns = {class_name: 0 for class_name in all_class_names}
    sampled_anns, sampled_imgs, sampled_img_ids = [], [], []

    def _sample(annotation):
        anno_cls = class_id_to_name[annotation['category_id']]
        assert cls_to_sampled_anns[anno_cls] < shots
        sampled_anns.append(annotation)
        if annotation['image_id'] not in sampled_img_ids:
            sampled_imgs.append(img_id_to_img[annotation['image_id']])
            sampled_img_ids.append(annotation['image_id'])
        cls_to_sampled_anns[anno_cls] += 1
    if strategy in ['one', 'any', 'all_or_none']:
        image_ids = list(img_id_to_anns.keys())
        if shuffle:
            if not silent:
                print("Shuffling images")
            random.shuffle(image_ids)
        else:
            if not silent:
                print("Skip shuffling of images...")
        for img_id in image_ids:
            if min(cls_to_sampled_anns.values()) == shots:
                break
            annos = list(img_id_to_anns[img_id])
            if strategy == 'all_or_none':
                if (anno_count_min and len(annos) < anno_count_min) or (anno_count_max and len(annos) > anno_count_max):
                    continue
                sample_img = True
                for class_name, img_to_anns in class_to_imgs_to_anns.items():
                    if img_id not in img_to_anns:
                        continue
                    if cls_to_sampled_anns[class_name] + len(img_to_anns[img_id]) > shots:
                        sample_img = False
                        break
                if sample_img:   # sample all annotations of this image
                    for anno in annos:
                        _sample(anno)
            else:
                if shuffle:
                    if not silent:
                        print("Shuffling the images' annotations")
                    random.shuffle(annos)
                else:
                    if not silent:
                        print("Skip shuffling of the images' annotations")
                for anno in annos:
                    anno_cls = class_id_to_name[anno['category_id']]
                    if cls_to_sampled_anns[anno_cls] >= shots:
                        continue
                    _sample(anno)
                    if strategy == 'one':
                        break
    else:  # sample all annotations of a single class or none
        for class_name, img_to_anns in class_to_imgs_to_anns.items():
            # TODO: probably add the functionality of saving the annotations sampled for a class separately (as in
            #  'prepare_coco_few_shot'. This way, the Base Shot Multiplier (BSM) and the Novel-class Oversampling Factor
            #  could be used, which is currently not possible (e.g. an image with objects of a base class and a novel
            #  class.)
            image_ids = list(img_to_anns.keys())
            if shuffle:
                if not silent:
                    print("Shuffling images")
                random.shuffle(image_ids)
            else:
                if not silent:
                    print("Skip shuffling of images...")
            for img_id in image_ids:
                if cls_to_sampled_anns[class_name] == shots:
                    break
                if cls_to_sampled_anns[class_name] + len(img_to_anns[img_id]) > shots:
                    continue
                annos = list(img_to_anns[img_id])
                if (anno_count_min and len(annos) < anno_count_min) or (anno_count_max and len(annos) > anno_count_max):
                    continue
                for ann in annos:
                    assert class_name == class_id_to_name[ann['category_id']]
                    _sample(ann)
    return sampled_imgs, sampled_anns


def analyse_sample(images, annotations, img_to_value=None):
    # img_to_value: an optional mapping of an image to any value. Can be used to compute e.g. the average value across
    #  the different classes of the sampled annotations
    num_imgs = len(images)
    num_annos = len(annotations)
    img_id_to_ann_count = {img['id']: 0 for img in images}
    class_name_to_anns = {class_name: [] for class_name in all_class_names}
    class_name_to_image_ids = {class_name: [] for class_name in all_class_names}  # unique image ids the classes appear on
    for ann in annotations:
        img_id = ann['image_id']
        class_name = class_id_to_name[ann['category_id']]
        assert img_id in img_id_to_ann_count
        assert class_name in class_name_to_anns
        assert class_name in class_name_to_image_ids
        img_id_to_ann_count[img_id] += 1
        class_name_to_anns[class_name].append(ann)
        if img_id not in class_name_to_image_ids[class_name]:
            class_name_to_image_ids[class_name].append(img_id)
    assert min(img_id_to_ann_count) > 0
    print("Sampled {} images and {} annotations.".format(num_imgs, num_annos))
    print("Distribution of annotations over images: {}".format(img_id_to_ann_count))
    class_name_to_ann_count = {class_name: len(class_name_to_anns[class_name]) for class_name in all_class_names}
    print("Distribution of annotations over classes: {}".format(class_name_to_ann_count))
    # Some more sophisticated analyses
    class_name_to_image_count = {class_name: len(class_name_to_image_ids[class_name]) for class_name in all_class_names}
    print("Amount of unique images each class appears on: {}".format(class_name_to_image_count))
    class_name_to_ann_sizes = {class_name: [ann["bbox"][2] * ann["bbox"][3] for ann in class_name_to_anns[class_name]] for class_name in all_class_names}
    class_name_to_mstd_ann_sizes = {class_name: "{} +/- {}".format(
        round(mean(class_name_to_ann_sizes[class_name]), mean_decimals),
        round(stdev(class_name_to_ann_sizes[class_name]), std_decimals)
    ) for class_name in all_class_names}
    print("Average object areas per class: {}".format(class_name_to_mstd_ann_sizes))
    if img_to_value:
        values = [img_to_value[img['id']] for img in images if (img['id'] in img_to_value and img_to_value[img["id"]])]
        if len(values) != len(images):
            print("Warning: Only {} values for {} images were available!".format(len(values), len(images)))
        print("Mean and std for the available image values: {} +/- {}".format(
            round(mean(values), mean_decimals),
            round(stdev(values), std_decimals)
        ))


def _get_topk_samples(samples, top_k, sort_by, reverse=False):
    # TODO: for first two 'sort_by' rules, add different behaviors on how to sort elements on its other value
    #  (if two elements have the same first value!)?
    #  -> could be realized by fist sorting for the second dimension, then sorting for the main main dimension
    assert sort_by in ['no_imgs', 'no_anns', 'ratio_ann_img']
    if sort_by == 'no_imgs':
        samples.sort(key=lambda s: len(s[0]), reverse=reverse)
    elif sort_by == 'no_anns':
        samples.sort(key=lambda s: len(s[1]), reverse=reverse)
    else:
        samples.sort(key=lambda s: float(len(s[1]))/len(s[0]), reverse=reverse)
    return samples[:top_k]


def _get_image_count(anno_count_min=None, anno_count_max=None):
    total_images = len(img_id_to_anns)
    msg = "Total images: {}.".format(total_images)
    if anno_count_min:
        ctr = 0
        for annos in img_id_to_anns.values():
            if len(annos) >= anno_count_min:
                ctr += 1
        msg += " {} images with at least {} annotations.".format(ctr, anno_count_min)
    if anno_count_max:
        ctr = 0
        for annos in img_id_to_anns.values():
            if len(annos) <= anno_count_max:
                ctr += 1
        msg += " {} images with at most {} annotations.".format(ctr, anno_count_max)
    print(msg)


def _get_img_to_gsd_map(meta_dir):
    """
    meta_dir: Contains textfiles, one per image. Each textfile is asserted to have at least one line starting with
      'gsd:' or 'GSD:' followed by a float number.
    """
    def _get_gsd(file_name):
        with open(os.path.join(meta_dir, file_name), 'r') as f:
            gsd = None
            for line in f:
                if line.lower().startswith("gsd:"):
                    gsd_str = line.split(":")[1]
                    try:
                        gsd = float(gsd_str)
                    except ValueError:
                        print("Invalid gsd value: {}".format(gsd_str))
                        gsd = None
                        # exit(1)
                    return gsd
            print("Warning: No gsd available in the file {}".format(file_name))
            return gsd
    images = img_id_to_img.values()
    img_id_to_gsd = {img["id"]: None for img in images}
    for file_name in os.listdir(meta_dir):
        gsd = _get_gsd(file_name)
        img_name = file_name.split(".")[0]  # remove file ending
        for img in images:  # TODO: slow solution! preprocessing images with same prefix could increase speed
            if img["file_name"].startswith(img_name):
                img_id_to_gsd[img["id"]] = gsd
    return img_id_to_gsd


def _find_overlapping_patches(images):
    """
    Takes a set of images (e.g. iSAID patches) where image["file_name"] matches the following pattern:
     <base-image-name>_<xmin>_<xmax>_<ymin>_<ymax>.<file-ending>, and prints all pairs of images that origin from the
      same base-image and overlap.
    TODO: Probably add the functionality to check whether there are objects in the overlapping area or not!
    """
    base_img_name_to_patches = {}
    patch_img_name_to_base_img_name = {}
    patch_img_name_to_coords = {}

    def _overlap(x1min, x1max, y1min, y1max, x2min, x2max, y2min, y2max):
        return \
            x2min < x1max or \
            y2min < y1max or \
            x1min < x2max or \
            y1min < y2max

    for img in images:
        patch_img_name = img["file_name"]
        base_img_name, xmin, xmax, ymin, ymax = patch_img_name.split('_')
        if base_img_name not in base_img_name_to_patches:
            base_img_name_to_patches[base_img_name] = []
        base_img_name_to_patches[base_img_name].append(patch_img_name)
        patch_img_name_to_coords[patch_img_name] = (float(xmin), float(xmax), float(ymin), float(ymax))
        patch_img_name_to_base_img_name[patch_img_name] = base_img_name
    for base_img_name, patches in base_img_name_to_patches.items():
        if len(patches) <= 1:
            continue
        for i in range(len(patches)):
            patch1_name = patches[i]["file_name"]
            for j in range(i + 1, len(patches)):
                patch2_name = patches[j]["file_name"]
                if _overlap(*patch_img_name_to_coords[patch1_name], *patch_img_name_to_coords[patch2_name]):
                    print("Patches {} and {} overlap!".format(patch1_name, patch2_name))


def _read_sample(seed, shots):
    path = os.path.join(_get_file_dir(seed), _get_file_name(shots))
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def _export_sample(images, annotations, file_dir, file_name, clear_dir=force_override):
    new_data = data.copy()
    new_data['images'] = images
    new_data['annotations'] = annotations
    if os.path.exists(file_dir):
        if clear_dir:
            print("Cleaning file save path directory '{}'".format(file_dir))
            shutil.rmtree(file_dir)
            os.mkdir(file_dir)
        else:
            print("Error, directory is not empty. Please set 'clear_dir' to True to clear the directory")
            exit(1)
    else:
        os.makedirs(file_dir)
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(new_data, f, indent=2)


def _get_file_dir(seed):
    return os.path.join(save_dir_base_path, "seed{}".format(seed))


def _get_file_name(shots):
    # Note: the file name has to be the same as in 'fsdet/data/meta_coco.py:load_cocolike_json'
    return "full_box_{}shot_{}.json".format(shots, cfg.TRAIN_SPLIT[dataset])


def _create_index():
    print("Creating index...")
    start = time.time()
    class_id_to_name = {i['id']: i['name'] for i in data['categories']}
    class_name_to_id = {i['name']: i['id'] for i in data['categories']}
    # TODO: probably add consistency checks to category names from class splits and the annotation file
    img_id_to_img = {i['id']: i for i in data['images']}
    # map class names to image ids to annotations -> all annotations of a certain class per image
    class_to_imgs_to_anns = {class_name: {} for class_name in all_class_names}
    img_id_to_anns = {}
    for anno in data['annotations']:
        class_name = class_id_to_name[anno['category_id']]
        if anno['image_id'] not in class_to_imgs_to_anns[class_name]:
            class_to_imgs_to_anns[class_name][anno['image_id']] = [anno]
        else:
            class_to_imgs_to_anns[class_name][anno['image_id']].append(anno)
        if anno['image_id'] not in img_id_to_anns:
            img_id_to_anns[anno['image_id']] = [anno]
        else:
            img_id_to_anns[anno['image_id']].append(anno)
    end = time.time()
    print("...index created in {}m {}s".format(*divmod(int(end-start), 60)))
    return class_id_to_name, class_name_to_id, img_id_to_img, class_to_imgs_to_anns, img_id_to_anns


if __name__ == '__main__':
    data = json.load(open(anno_dir, 'r'))
    class_id_to_name, class_name_to_id, img_id_to_img, class_to_imgs_to_anns, img_id_to_anns = _create_index()
    main()
