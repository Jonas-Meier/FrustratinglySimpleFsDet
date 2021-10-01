from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C
import os
from class_splits import CLASS_SPLITS, ALL_CLASSES

# adding additional default values built on top of the default values in detectron2

_CC = _C

# Some dataset and class split specific patterns
_CC.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_CC.DATASETS.SUPPORTED_DATASETS = ["coco", "isaid", "isaid_100","isaid_50", "isaid_25", "isaid_10", "isaid_low_gsd",
                                   "isaid_high_gsd", "fair1m", "fair1m_groupcats"]  # all supported datasets. Others won't work
# datasets that have the same annotation format as MS COCO (those datasets are directly supported and will require the
#  fewest amount of adaptions!)
_CC.DATASETS.COCOLIKE_DATASETS = ["coco", "isaid", "isaid_100","isaid_50", "isaid_25", "isaid_10", "isaid_low_gsd",
                                  "isaid_high_gsd", "fair1m", "fair1m_groupcats"]
# Note: We need to create a dictionary within a new config node CN(), otherwise, dicts won't work!
_CC.DATA_DIR = CN({
    "coco": os.path.join(_CC.ROOT_DIR, 'datasets', 'coco'),
    "isaid": os.path.join(_CC.ROOT_DIR, 'datasets', 'isaid'),
    "isaid_100": os.path.join(_CC.ROOT_DIR, 'datasets', 'isaid'),
    "isaid_50": os.path.join(_CC.ROOT_DIR, 'datasets', 'isaid_50'),
    "isaid_25": os.path.join(_CC.ROOT_DIR, 'datasets', 'isaid_25'),
    "isaid_10": os.path.join(_CC.ROOT_DIR, 'datasets', 'isaid_10'),
    "isaid_low_gsd": os.path.join(_CC.ROOT_DIR, 'datasets', 'isaid_low_gsd'),
    "isaid_high_gsd": os.path.join(_CC.ROOT_DIR, 'datasets', 'isaid_high_gsd'),
    "fair1m": os.path.join(_CC.ROOT_DIR, 'datasets', 'fair1m'),
    "fair1m_groupcats": os.path.join(_CC.ROOT_DIR, 'datasets', 'fair1m_groupcats'),
})

_CC.DATA_SAVE_PATH_PATTERN = CN({
    "coco": os.path.join(_CC.ROOT_DIR, 'datasets', "cocosplit", "cocosplit_{}"),
    "isaid": os.path.join(_CC.ROOT_DIR, 'datasets', "isaidsplit", "isaidsplit_{}"),
    "isaid_100": os.path.join(_CC.ROOT_DIR, 'datasets', "isaid_100split", "isaid_100split_{}"),
    "isaid_50": os.path.join(_CC.ROOT_DIR, 'datasets', "isaid_50split", "isaid_50split_{}"),
    "isaid_25": os.path.join(_CC.ROOT_DIR, 'datasets', "isaid_25split", "isaid_25split_{}"),
    "isaid_10": os.path.join(_CC.ROOT_DIR, 'datasets', "isaid_10split", "isaid_10split_{}"),
    "isaid_low_gsd": os.path.join(_CC.ROOT_DIR, 'datasets', "isaid_low_gsdsplit", "isaid_low_gsd_split_{}"),
    "isaid_high_gsd": os.path.join(_CC.ROOT_DIR, 'datasets', "isaid_high_gsdsplit", "isaid_high_gsdsplit_{}"),
    "fair1m": os.path.join(_CC.ROOT_DIR, 'datasets', "fair1msplit", "fair1msplit_{}"),
    "fair1m_groupcats": os.path.join(_CC.ROOT_DIR, 'datasets', "fair1m_groupcatssplit", "fair1m_groupcatssplit_{}"),
})

# relative to repository root
_CC.CONFIG_DIR_PATTERN = CN({
    "coco": os.path.join('configs', 'COCO-detection', "cocosplit_{}"),
    "isaid": os.path.join('configs', 'iSAID-detection', "isaidsplit_{}"),
    "isaid_100": os.path.join('configs', 'iSAID-100-detection', "isaid_100split_{}"),
    "isaid_50": os.path.join('configs', 'iSAID-50-detection', "isaid_50split_{}"),
    "isaid_25": os.path.join('configs', 'iSAID-25-detection', "isaid_25split_{}"),
    "isaid_10": os.path.join('configs', 'iSAID-10-detection', "isaid_10split_{}"),
    "isaid_low_gsd": os.path.join('configs', 'iSAID-low-gsd-detection', "isaid_low_gsdsplit_{}"),
    "isaid_high_gsd": os.path.join('configs', 'iSAID-high-gsd-detection', "isaid_high_gsdsplit_{}"),
    "fair1m": os.path.join('configs', 'FAIR1M-detection', "fair1msplit_{}"),
    "fair1m_groupcats": os.path.join('configs', 'FAIR1M-groupcats-detection', "fair1m_groupcatssplit_{}"),
})

# relative to repository root
_CC.CKPT_DIR_PATTERN = CN({
    "coco": os.path.join('checkpoints', 'coco_{}'),
    "isaid": os.path.join('checkpoints', 'isaid_{}'),
    "isaid_100": os.path.join('checkpoints', 'isaid_100_{}'),
    "isaid_50": os.path.join('checkpoints', 'isaid_50_{}'),
    "isaid_25": os.path.join('checkpoints', 'isaid_25_{}'),
    "isaid_10": os.path.join('checkpoints', 'isaid_10_{}'),
    "isaid_low_gsd": os.path.join('checkpoints', 'isaid_low_gsd_{}'),
    "isaid_high_gsd": os.path.join('checkpoints', 'isaid_high_gsd_{}'),
    "fair1m": os.path.join('checkpoints', 'fair1m_{}'),
    "fair1m_groupcats": os.path.join('checkpoints', 'fair1m_groupcats_{}'),
})

_CC.TRAIN_SPLIT = CN({
    "coco": 'trainval',
    "isaid": 'train',
    "isaid_100": 'train',
    "isaid_50": 'train',
    "isaid_25": 'train',
    "isaid_10": 'train',
    "isaid_low_gsd": 'train',
    "isaid_high_gsd": 'train',
    "fair1m": 'train',
    "fair1m_groupcats": 'train',
})

_CC.TEST_SPLIT = CN({
    "coco": 'test',
    "isaid": 'test',
    "isaid_100": 'test',
    "isaid_50": 'test',
    "isaid_25": 'test',
    "isaid_10": 'test',
    "isaid_low_gsd": 'test',
    "isaid_high_gsd": 'test',
    "fair1m": 'test',
    "fair1m_groupcats": 'test',
})


# following dirs and files: are relative to the repository root!
_CC.TRAIN_IMG_DIR = CN({
    "coco": os.path.join('datasets', 'coco', 'trainval2014'),
    "isaid": os.path.join('datasets', 'isaid', 'images', 'train'),
    "isaid_100": os.path.join('datasets', 'isaid', 'images', 'train'),
    "isaid_50": os.path.join('datasets', 'isaid', 'images', 'train'),
    "isaid_25": os.path.join('datasets', 'isaid', 'images', 'train'),
    "isaid_10": os.path.join('datasets', 'isaid', 'images', 'train'),
    "isaid_low_gsd": os.path.join('datasets', 'isaid', 'images', 'train'),
    "isaid_high_gsd": os.path.join('datasets', 'isaid', 'images', 'train'),
    "fair1m": os.path.join('datasets', 'fair1m', 'images', 'train'),
    "fair1m_groupcats": os.path.join('datasets', 'fair1m_groupcats', 'images', 'train'),
})

_CC.TEST_IMG_DIR = CN({
    "coco": os.path.join('datasets', 'coco', 'trainval2014'),
    "isaid": os.path.join('datasets', 'isaid', 'images', 'val'),
    "isaid_100": os.path.join('datasets', 'isaid', 'images', 'val'),
    "isaid_50": os.path.join('datasets', 'isaid', 'images', 'val'),
    "isaid_25": os.path.join('datasets', 'isaid', 'images', 'val'),
    "isaid_10": os.path.join('datasets', 'isaid', 'images', 'val'),
    "isaid_low_gsd": os.path.join('datasets', 'isaid', 'images', 'val'),
    "isaid_high_gsd": os.path.join('datasets', 'isaid', 'images', 'val'),
    "fair1m": os.path.join('datasets', 'fair1m', 'images', 'val'),
    "fair1m_groupcats": os.path.join('datasets', 'fair1m_groupcats', 'images', 'val'),
})

_CC.TRAIN_ANNOS = CN({
    "coco": os.path.join('datasets', 'coco', 'annotations', 'trainvalno5k.json'),
    "isaid": os.path.join('datasets', 'isaid', 'annotations', 'instancesonly_filtered_train.json'),
    "isaid_100": os.path.join('datasets', 'isaid', 'annotations', 'instancesonly_filtered_train.json'),
    "isaid_50": os.path.join('datasets', 'isaid_50', 'annotations', 'instancesonly_filtered_train.json'),
    "isaid_25": os.path.join('datasets', 'isaid_25', 'annotations', 'instancesonly_filtered_train.json'),
    "isaid_10": os.path.join('datasets', 'isaid_10', 'annotations', 'instancesonly_filtered_train.json'),
    "isaid_low_gsd": os.path.join('datasets', 'isaid_low_gsd', 'annotations', 'instancesonly_filtered_train.json'),
    "isaid_high_gsd": os.path.join('datasets', 'isaid_high_gsd', 'annotations', 'instancesonly_filtered_train.json'),
    "fair1m": os.path.join('datasets', 'fair1m', 'annotations', 'instancesonly_filtered_HBB_train.json'),
    "fair1m_groupcats": os.path.join('datasets', 'fair1m_groupcats', 'annotations', 'instancesonly_filtered_HBB_groupcats_train.json'),
})

_CC.TEST_ANNOS = CN({
    "coco": os.path.join('datasets', 'coco', 'annotations', '5k.json'),
    "isaid": os.path.join('datasets', 'isaid', 'annotations', 'instancesonly_filtered_val.json'),
    "isaid_100": os.path.join('datasets', 'isaid', 'annotations', 'instancesonly_filtered_val.json'),
    "isaid_50": os.path.join('datasets', 'isaid', 'annotations', 'instancesonly_filtered_val.json'),
    "isaid_25": os.path.join('datasets', 'isaid', 'annotations', 'instancesonly_filtered_val.json'),
    "isaid_10": os.path.join('datasets', 'isaid', 'annotations', 'instancesonly_filtered_val.json'),
    "isaid_low_gsd": os.path.join('datasets', 'isaid_low_gsd', 'annotations', 'instancesonly_filtered_val.json'),
    "isaid_high_gsd": os.path.join('datasets', 'isaid_high_gsd', 'annotations', 'instancesonly_filtered_val.json'),
    "fair1m": os.path.join('datasets', 'fair1m', 'annotations', 'instancesonly_filtered_HBB_val.json'),
    "fair1m_groupcats": os.path.join('datasets', 'fair1m_groupcats', 'annotations', 'instancesonly_filtered_HBB_groupcats_val.json'),
})

# How many annotations to use per image while fine-tuning:
#  'all' uses all available annotations
#  'one' duplicates images with more than one annotation and only adds a single annotation per image instance
_CC.FT_ANNOS_PER_IMAGE = 'all'  # 'all' or 'one'. Default: 'one'

_CC.VALID_FEW_SHOTS = [1, 2, 3, 5, 10, 20, 30, 50, 100]

_CC.MAX_SEED_VALUE = 100  # Increase if necessary. Note that a large value will blow up the DatasetCatalog!


# BASE_SHOT_MULTIPLIER is used for both, sampling data from original annotations and used by data preparation for
#  training. It determines how much base-class annotations are sampled since their amount in the training dataset is
#  often much higher than the shot parameter K.
# NOVEL_OVERSAMPLING_FACTOR is just used for data preparation for training. It determines how often the sampled images,
#  containing K annotations, are duplicated, to allow for more balanced datasets if the BASE_SHOT_MULTIPLIER was used to
#  sample more than K annotations for base classes.
# Following combinations of values may be used in the config, X, Y and K are integers > 0
#  (BASE_SHOT_MULTIPLIER | NOVEL_OVERSAMPLING_FACTOR -> base class data used | novel class data used)
#  X  |  Y  ->  X * K   | Y * K
#  X  | -1  ->  X * K   | X * K
#  -1 |  Y  ->  all     | Y * K      (Note: base class data remains imbalanced!)
#  -1 | -1  ->  all classes balanced to amount of class with most annotations
_CC.BASE_SHOT_MULTIPLIER = 5  # default: 1, -1 for using all data
_CC.NOVEL_OVERSAMPLING_FACTOR = -1  # default: 1, -1 for same amount as base classes

_CC.EVENT_WRITER_PERIOD = 100  # default: 20

# Choose from 'ResizeShortestEdgeLimitLongestEdge', 'RandomHFlip', 'RandomVFlip', 'RandomFourAngleRotation'
#  Note: ["ResizeShortestEdgeLimitLongestEdge", "RandomHFlip"] equals the Detectron2 default
_CC.INPUT.AUGMENTATIONS = [
    "ResizeShortestEdgeLimitLongestEdge",
    "RandomHFlip"
]

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_BOX_HEAD.FREEZE_CONVS = []  # freeze bbox head conv layers with given id (starting by 1)
_CC.MODEL.ROI_BOX_HEAD.FREEZE_FCS = []  # freeze bbox head fc layers with given id (starting by 1)
# Multi-Head configs
_CC.MODEL.ROI_HEADS.MULTIHEAD_NUM_CLASSES = [60, 20]  # num classes for each head
_CC.MODEL.ROI_BOX_HEAD.NUM_HEADS = 2
_CC.MODEL.ROI_BOX_HEAD.SPLIT_AT_FC = 2  # no. of fc layer where to split the head

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

# Backward Compatible options.
_CC.MUTE_HEADER = True


assert all(dataset in _CC.DATASETS.SUPPORTED_DATASETS for dataset in _CC.DATASETS.COCOLIKE_DATASETS)
assert all(dataset in dictionary for dataset in _CC.DATASETS.SUPPORTED_DATASETS for dictionary in [
    _CC.DATA_DIR, _CC.DATA_SAVE_PATH_PATTERN, _CC.CONFIG_DIR_PATTERN, _CC.CKPT_DIR_PATTERN, _CC.TRAIN_SPLIT,
    _CC.TEST_SPLIT, _CC.TRAIN_IMG_DIR, _CC.TEST_IMG_DIR, _CC.TRAIN_ANNOS, _CC.TEST_ANNOS
])
assert all(dataset in CLASS_SPLITS for dataset in _CC.DATASETS.SUPPORTED_DATASETS)
assert all(dataset in ALL_CLASSES for dataset in _CC.DATASETS.SUPPORTED_DATASETS)
