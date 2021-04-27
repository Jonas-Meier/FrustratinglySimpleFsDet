from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C
import os

# adding additional default values built on top of the default values in detectron2

_CC = _C

# Some dataset and class split specific patterns
_CC.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Note: We need to create a dictionary within a new config node CN(), otherwise, dicts won't work!
_CC.DATA_DIR = CN({
    "coco": os.path.join(_CC.ROOT_DIR, 'datasets', 'coco'),
    "isaid": os.path.join(_CC.ROOT_DIR, 'datasets', 'isaid')
})

_CC.DATA_SAVE_PATH_PATTERN = CN({
    "coco": os.path.join(_CC.ROOT_DIR, 'datasets', "cocosplit", "cocosplit_{}"),
    "isaid": os.path.join(_CC.ROOT_DIR, 'datasets', "isaidsplit", "isaidsplit_{}")
})

# relative to repository root
_CC.CONFIG_DIR_PATTERN = CN({
    "coco": os.path.join('configs', 'COCO-detection', "cocosplit_{}"),
    "isaid": os.path.join('configs', 'iSAID-detection', "isaidsplit_{}")
})

# relative to repository root
_CC.CONFIG_CKPT_DIR_PATTERN = CN({
    "coco": os.path.join('checkpoints', 'coco_{}'),
    "isaid": os.path.join('checkpoints', 'isaid_{}')
})

_CC.TRAIN_SPLIT = CN({
    "coco": 'trainval',
    "isaid": 'train'
})

_CC.TEST_SPLIT = CN({
    "coco": 'test',
    "isaid": 'test'
})


# following dirs and files: are relative to the repository root!
_CC.TRAIN_IMG_DIR = CN({
    "coco": os.path.join('datasets', 'coco', 'trainval2014'),
    "isaid": os.path.join('datasets', 'isaid', 'images', 'train')
})

_CC.TEST_IMG_DIR = CN({
    "coco": os.path.join('datasets', 'coco', 'val2014'),
    "isaid": os.path.join('datasets', 'isaid', 'images', 'val')
})

_CC.TRAIN_ANNOS = CN({
    "coco": os.path.join('datasets', 'cocosplit', 'datasplit', 'trainvalno5k.json'),
    "isaid": os.path.join('datasets', 'isaid', 'annotations', 'instancesonly_filtered_train.json')
})

_CC.TEST_ANNOS = CN({
    "coco": os.path.join('datasets', 'cocosplit', 'datasplit', '5k.json'),
    "isaid": os.path.join('datasets', 'isaid', 'annotations', 'instancesonly_filtered_val.json')
})

# How many annotations to use per image while fine-tuning:
#  'all' uses all available annotations
#  'one' duplicates images with mire than one annotation and only adds a single annotation per image instance
_CC.FT_ANNOS_PER_IMAGE = 'all'  # 'all' or 'one'. Default: 'one'

_CC.VALID_FEW_SHOTS = [1, 2, 3, 5, 10, 20, 30, 50, 100]

_CC.MAX_SEED_VALUE = 9  # Increase if necessary. Note that a large value will blow up the DatasetCatalog!

# For fine-tuning, sample 'BASE_SHOT_MULTIPLIER' x 'shots' shots for base classes
# used by data preparation and annotation reading
_CC.BASE_SHOT_MULTIPLIER = 5  # default: 1
# For fine-tuning, oversample novel classes
# Just used in annotation reading
_CC.NOVEL_OVERSAMPLING_FACTOR = 5  # default: 1

_CC.EVENT_WRITER_PERIOD = 100  # default: 20

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
# Deprecated: Use two configs below.
#  False: Activates the two configs below
#  True: overrides the two configs below, mainly for backward-compatibility of configs
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
_CC.MODEL.ROI_BOX_HEAD.FREEZE_CONVS = []  # freeze bbox head conv layers with given id (starting by 1)
_CC.MODEL.ROI_BOX_HEAD.FREEZE_FCS = []  # freeze bbox head fc layers with given id (starting by 1)

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

# Backward Compatible options.
_CC.MUTE_HEADER = True