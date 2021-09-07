import json
import math
import os
import random
import shutil

from fsdet.config.config import get_cfg


"""
This script allows to split a given dataset into two datasets in many possible ways (e.g. by image image count or by 
values assigned to each image). In addition, this script supports analysing dataset splits in order to find a split 
which splits each single class with the same proportions. The datasets obtained by this script may be used to replace
certain datasets at base training or fine-tuning.
"""

cfg = get_cfg()


original_dataset_name = "isaid"
original_anno_dir = cfg.TRAIN_ANNOS[original_dataset_name]

force_override = False  # Force overriding of already existing datasets


def main():
    if force_override:
        print("Warning: Overriding of already existing datasets is activated!")
    pool_size = 1000
    proportion = 0.1
    dataset1_name = "isaid_50_1"
    dataset2_name = "isaid_50_2"
    (imgs1, anns1), (imgs2, anns2) = get_uniform_dataset_split(pool_size=pool_size, proportion=proportion)
    _analyse_sampled_dataset(imgs1, anns1)
    _export_dataset(images=imgs1, annotations=anns1, save_dir=original_anno_dir.replace(original_dataset_name, dataset1_name), file_name=dataset1_name)
    _export_dataset(images=imgs2, annotations=anns2, save_dir=original_anno_dir.replace(original_dataset_name, dataset2_name), file_name=dataset2_name)

    if False:
        samples = []
        pool_size = 100
        proportion = 0.5
        for _ in range(pool_size):
            (images1, annotations1), (images2, annotations2) = split_by_image_count(proportion=proportion)
            samples.append(((images1, annotations1), (images2, annotations2)))
        samples.sort(key=lambda s: _loss(s[0][0], s[0][1], proportion))
        print("Most equal split (loss: {}):".format(_loss(samples[0][0][0], samples[0][0][1], proportion)))
        _analyse_sampled_dataset(samples[0][0][0], samples[0][0][1])
        print("Most unequal split (loss: {}):".format(_loss(samples[-1][0][0], samples[-1][0][1], proportion)))
        _analyse_sampled_dataset(samples[-1][0][0], samples[-1][0][1])


def get_uniform_dataset_split(pool_size=100, proportion=0.5):
    """
    Split the dataset into two datasets (by using the method 'split_by_image_count'). This is repeated 'pool_size'
     times. Those datasets are then sorted, depending on their loss (as defined by the method '_loss') and the dataset
     split with the smallest loss is returned (smaller loss equals a class distribution that is closest to the class
     distribution in the original dataset).
    """
    samples = []
    for _ in range(pool_size):
        (images1, annotations1), (images2, annotations2) = split_by_image_count(proportion=proportion)
        samples.append(((images1, annotations1), (images2, annotations2)))
    samples.sort(key=lambda s: _loss(s[0][0], s[0][1], proportion))
    return samples[0]


def split_by_image_count(proportion=0.5, shuffle=True, silent=True):
    """
    Returns two dataset, one with 'proportion * total_images' images, the other with
    '(1 - proportion) * total_images' images.
    """
    image_ids = list(img_id_to_anns.keys())
    if shuffle:
        if not silent:
            print("Shuffling images")
        random.shuffle(image_ids)
    else:
        if not silent:
            print("Skip shuffling of images...")
    split_index = int(len(image_ids) * proportion)
    image_ids_split1 = image_ids[0:split_index]
    image_ids_split2 = image_ids[split_index: len(image_ids)]
    images_split1 = [img_id_to_img[img_id] for img_id in image_ids_split1]
    images_split2 = [img_id_to_img[img_id] for img_id in image_ids_split2]
    annotations_split1 = [ann for img_id in image_ids_split1 for ann in img_id_to_anns[img_id]]
    annotations_split2 = [ann for img_id in image_ids_split2 for ann in img_id_to_anns[img_id]]
    return (images_split1, annotations_split1), (images_split2, annotations_split2)


def split_by_value(img_id_to_value: {int: float}, proportion, threshold):
    """
    Split the images of a dataset depending on values assigned to each image.
    Either use a fixed proportion or a threshold to split the images.
    Using a proportion will return two datasets, containing the 'proportion * #images' with the lowest value and the
     '(1 - proportion) * #images' with the highest values, respectively.
    For a given threshold s, this method returns two datasets, the first containing images i where value(i) <= s and the
     other dataset containing images i where value(i) > s.
    """
    image_ids = list(img_id_to_anns.keys())
    # Remove images that have no assigned value in the given map
    valid_img_ids = []
    for img_id in image_ids:
        if img_id["id"] in img_id_to_value and img_id_to_value[img_id["id"]]:
            valid_img_ids.append(img_id)
        else:
            print("Warning, no assigned value for image (id:{}, name:{}). It will be discarded!"
                  .format(img_id, img_id_to_img[img_id]["file_name"]))
    # Either use a proportion or a fixed threshold for splitting the images according to the value map
    #  (Note: reverse=True will break threshold-based index computation!)
    valid_img_ids.sort(key=lambda img_id: img_id_to_value[img_id], reverse=False)
    if proportion:
        assert threshold is None
        split_index = int(len(valid_img_ids) * proportion)
    else:
        assert threshold is not None
        split_index = 0
        for imd_id in valid_img_ids:
            if img_id_to_value[imd_id] <= threshold:
                split_index += 1
            else:
                break
    image_ids_split1 = valid_img_ids[0:split_index]
    image_ids_split2 = valid_img_ids[split_index: len(valid_img_ids)]
    images_split1 = [img_id_to_img[img_id] for img_id in image_ids_split1]
    images_split2 = [img_id_to_img[img_id] for img_id in image_ids_split2]
    annotations_split1 = [ann for img_id in image_ids_split1 for ann in img_id_to_anns[img_id]]
    annotations_split2 = [ann for img_id in image_ids_split2 for ann in img_id_to_anns[img_id]]
    return (images_split1, annotations_split1), (images_split2, annotations_split2)


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
                        print("Error, invalid gsd value: {}".format(gsd_str))
                        exit(1)
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


def _analyse_sampled_dataset(images, annotations):
    def _print_formatted(a, b, c, d):
        print("{:<20}{:<20}{:<20}{:<6}".format(a, b, c, "({})".format(d)))
    num_imgs_original = len(original_dataset["images"])
    num_imgs_sampled = len(images)
    num_anns_original = len(original_dataset["annotations"])
    num_anns_sampled = len(annotations)
    all_cat_names = [c["name"] for c in original_dataset["categories"]]
    cls_name_to_ann_count_original = {cat_name: 0 for cat_name in all_cat_names}
    cls_name_to_ann_count_sampled = {cat_name: 0 for cat_name in all_cat_names}
    for ann in original_dataset["annotations"]:
        cls_name_to_ann_count_original[class_id_to_name[ann["category_id"]]] += 1
    for ann in annotations:
        cls_name_to_ann_count_sampled[class_id_to_name[ann["category_id"]]] += 1
    _print_formatted("Subset", "Original Dataset", "Sampled Dataset", "rel.")
    _print_formatted("Images", num_imgs_original, num_imgs_sampled, round(float(num_imgs_sampled) / num_imgs_original, 2))
    _print_formatted("Annotations", num_anns_original, num_anns_sampled, round(float(num_anns_sampled) / num_anns_original, 2))
    for cls in all_cat_names:
        num_cat_anns_original = cls_name_to_ann_count_original[cls]
        num_cat_anns_sampled = cls_name_to_ann_count_sampled[cls]
        _print_formatted(cls, num_cat_anns_original, num_cat_anns_sampled, round(float(num_cat_anns_sampled)/num_cat_anns_original, 2))


def _loss(images, annotations, proportion=0.5):
    def _distance(sampled, original):
        relation = float(sampled) / original
        return -math.log(1 - abs(relation - proportion))
    loss = 0.0
    loss += _distance(len(images), len(original_dataset["images"]))
    loss += _distance(len(annotations), len(original_dataset["annotations"]))
    all_cat_names = [c["name"] for c in original_dataset["categories"]]
    cls_name_to_ann_count_original = {cat_name: 0 for cat_name in all_cat_names}
    cls_name_to_ann_count_sampled = {cat_name: 0 for cat_name in all_cat_names}
    for ann in original_dataset["annotations"]:
        cls_name_to_ann_count_original[class_id_to_name[ann["category_id"]]] += 1
    for ann in annotations:
        cls_name_to_ann_count_sampled[class_id_to_name[ann["category_id"]]] += 1
    for cls in all_cat_names:
        loss += _distance(cls_name_to_ann_count_sampled[cls], cls_name_to_ann_count_original[cls])
    return loss


def _export_dataset(images, annotations, save_dir, file_name, clear_dir=force_override):
    new_dataset = original_dataset.copy()
    new_dataset['images'] = images
    new_dataset['annotations'] = annotations
    if os.path.exists(save_dir):
        if clear_dir:
            print("Cleaning file save path directory '{}'".format(save_dir))
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)
        else:
            print("Error, directory is not empty. Please set 'clear_dir' to True to clear the directory")
            exit(1)
    else:
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, 'w') as f:
        json.dump(save_dir, f, indent=2)


def _create_index():
    print("Creating index...")
    class_id_to_name = {i['id']: i['name'] for i in original_dataset['categories']}
    class_name_to_id = {i['name']: i['id'] for i in original_dataset['categories']}
    img_id_to_img = {i['id']: i for i in original_dataset['images']}
    class_to_imgs_to_anns = {class_name: {} for class_name in class_id_to_name.values()}
    img_id_to_anns = {}
    for anno in original_dataset['annotations']:
        class_name = class_id_to_name[anno['category_id']]
        if anno['image_id'] not in class_to_imgs_to_anns[class_name]:
            class_to_imgs_to_anns[class_name][anno['image_id']] = [anno]
        else:
            class_to_imgs_to_anns[class_name][anno['image_id']].append(anno)
        if anno['image_id'] not in img_id_to_anns:
            img_id_to_anns[anno['image_id']] = [anno]
        else:
            img_id_to_anns[anno['image_id']].append(anno)
    return class_id_to_name, class_name_to_id, img_id_to_img, class_to_imgs_to_anns, img_id_to_anns


if __name__ == '__main__':
    original_dataset = json.load(open(original_anno_dir, 'r'))
    class_id_to_name, class_name_to_id, img_id_to_img, class_to_imgs_to_anns, img_id_to_anns = _create_index()
    main()
