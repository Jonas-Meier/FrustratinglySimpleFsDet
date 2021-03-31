import numpy as np
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

import contextlib
import io
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from class_splits import CLASS_SPLITS
from fsdet.config import get_cfg
cfg = get_cfg()

"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""

__all__ = ["register_meta_cocolike", "get_cocolike_metadata_names"]


def get_cocolike_metadata_names(dataset='coco', train_name='trainval', test_name='test'):
    assert train_name not in test_name
    assert test_name not in train_name
    ret = {}
    # TODO: unnecessary to save dataset, train_name and test_name as args because they have to be known prior to calling
    #  this method?
    # register whole training and testing datasets. Whole test dataset is used for fine-tuning inference
    #  TODO: what is the complete training dataset used for?
    for split in [train_name, test_name]:
        args = (dataset, split)
        ret['{}_{}_all'.format(*args)] = args
    # register class-split dependent datasets (just needed to get the correct metadata!):
    #  base training datasets
    #  testing datasets for base training and for fine tuning just the novel detector
    for class_split in CLASS_SPLITS[dataset].keys():
        args = (dataset, class_split, train_name, 'base')
        ret['{}_{}_{}_{}'.format(*args)] = args
        for prefix in ['base', 'novel']:
            args = (dataset, class_split, test_name, prefix)
            ret['{}_{}_{}_{}'.format(*args)] = args
    # register training datasets for fine tuning the whole detector
    for class_split in CLASS_SPLITS['coco'].keys():
        for prefix in ['all', 'novel']:
            for shot in cfg.VALID_FEW_SHOTS:
                for seed in range(cfg.MAX_SEED_VALUE + 1):  # maximum seed value is inclusive!
                    args = (dataset, class_split, train_name, prefix, shot, seed)
                    ret['{}_{}_{}_{}_{}shot_seed{}'.format(*args)] = args
    return ret


def load_cocolike_json(dataset, json_file, image_root, metadata, dataset_name):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.
    Args:
        dataset(str): dataset identifier
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        metadata: meta data associated with dataset_name
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    # TODO: either take 'dataset' as argument or use dataset_name.split('_')[0] as dataset
    if dataset == 'coco':
        train_name = 'trainval'
        test_name = 'test'
    else:
        raise ValueError("Dataset {} is not supported!".format(dataset))
    cocolike_metadata_names = get_cocolike_metadata_names(dataset, train_name, test_name)
    assert dataset_name in cocolike_metadata_names
    dataset_labels = cocolike_metadata_names[dataset_name]
    id_map = metadata["thing_dataset_id_to_contiguous_id"]
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "category_id"]
    if len(dataset_labels) == 2 or len(dataset_labels) == 4:
        # Note: we don't care about the actual labels. We load the whole dataset and all annotations nonetheless.
        #  The labels are just important to obtain the correct metadata.
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(json_file)
        # sort indices for reproducible results
        img_ids = sorted(list(coco_api.imgs.keys()))
        imgs = coco_api.loadImgs(img_ids)
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
        imgs_anns = list(zip(imgs, anns))
        for (img_dict, anno_dict_list) in imgs_anns:
            record = {}
            record["file_name"] = os.path.join(
                image_root, img_dict["file_name"]
            )
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            image_id = record["image_id"] = img_dict["id"]

            objs = []
            for anno in anno_dict_list:
                assert anno["image_id"] == image_id
                assert anno.get("ignore", 0) == 0

                obj = {key: anno[key] for key in ann_keys if key in anno}

                obj["bbox_mode"] = BoxMode.XYWH_ABS
                if obj["category_id"] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    else:  # ~ 'is_shots'
        assert len(dataset_labels) == 6
        assert 'shot' in dataset_name  # normally not necessary, but resembles the old version of this code
        (_, class_split, _, prefix, shot, seed) = dataset_labels
        fileids = {}
        split_dir = cfg.DATA_SAVE_PATH_PATTERN[dataset].format(class_split)
        split_dir = os.path.join(split_dir, 'seed{}'.format(seed))
        for idx, cls in enumerate(metadata["thing_classes"]):
            json_file = os.path.join(split_dir, "full_box_{}shot_{}_trainval.json".format(shot, cls))
            json_file = PathManager.get_local_path(json_file)
            with contextlib.redirect_stdout(io.StringIO()):
                coco_api = COCO(json_file)
            img_ids = sorted(list(coco_api.imgs.keys()))
            imgs = coco_api.loadImgs(img_ids)
            anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
            fileids[idx] = list(zip(imgs, anns))
        for _, fileids_ in fileids.items():
            dicts = []
            for (img_dict, anno_dict_list) in fileids_:
                for anno in anno_dict_list:
                    record = {}
                    record["file_name"] = os.path.join(
                        image_root, img_dict["file_name"]
                    )
                    record["height"] = img_dict["height"]
                    record["width"] = img_dict["width"]
                    image_id = record["image_id"] = img_dict["id"]

                    assert anno["image_id"] == image_id
                    assert anno.get("ignore", 0) == 0

                    obj = {key: anno[key] for key in ann_keys if key in anno}

                    obj["bbox_mode"] = BoxMode.XYWH_ABS
                    obj["category_id"] = id_map[obj["category_id"]]
                    record["annotations"] = [obj]
                    dicts.append(record)
            if len(dicts) > int(shot):
                dicts = np.random.choice(dicts, int(shot), replace=False)
            dataset_dicts.extend(dicts)
    return dataset_dicts


def register_meta_cocolike(dataset, name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_cocolike_json(dataset, annofile, imgdir, metadata, name),
    )

    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]

    MetadataCatalog.get(name).set(  # TODO: probably also need to add class_split to MetadataCatalog?
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        dirname="datasets/{}".format(dataset),
        **metadata,
    )
