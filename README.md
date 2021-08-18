[//]: # (This may be the most platform independent comment)

# Few-Shot Object Detection (FsDet)

FsDet contains the official few-shot object detection implementation of the ICML 2020 paper [Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/abs/2003.06957).

[Original Repository](https://github.com/ucbdrive/few-shot-object-detection).

![TFA Figure](https://user-images.githubusercontent.com/7898443/76520006-698cc200-6438-11ea-864f-fd30b3d50cea.png)

## Setup
This repository has been successfully tested with following configuration:
1. CUDA 10.1(.243) (CUDA 10.0 and 10.2 should work as well)
2. cuDNN 7.6.3 or 7.6.4 for CUDA 10.1
3. gcc/g++ 7.5 (anything >= 5.0 should work)

## Variable Denotation
* We assume the repository to be located at `<FSDET_ROOT>` (e.g. `/home/<user>/workspace/frustratingly-simple-fsdet`)
* CUDA is located at `CUDA_ROOT` (e.g. `/home/<user>/cuda-10.1`)

## Build
1. Create an environment (e.g. with conda) and activate it
``` bash
conda create --name fs-fsdet
conda activate fs-fsdet
```
2.Install PyTorch, depending on your local CUDA version (e.g. PyTorch 1.6 for CUDA 10.1). See PyTorch [actual version](https://pytorch.org/get-started/locally/) and [old versions](https://pytorch.org/get-started/previous-versions/).
``` bash
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```
3. Install Detectron2 v0.2.1, depending on PyTorch and CUDA version (e.g. for PyTorch 1.6 and CUDA 10.1). See [detectron2 releases](https://github.com/facebookresearch/detectron2/releases) for pre-built linux binaries.
``` bash
python3 -m pip install detectron2==0.2.1 -f  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```
4. Install other requirements
``` bash
python3 -m pip install -r requirements.txt
```
## Code Structure
- **configs**: Configuration files
- **datasets**: Dataset files (see [Dataset Preparation](#dataset-preparation) for more details)
- **fsdet**
  - **checkpoint**: Checkpoint code.
  - **config**: Configuration code and default configurations.
  - **engine**: Contains training and evaluation loops and hooks.
  - **layers**: Implementations of different layers used in models.
  - **modeling**: Code for models, including backbones, proposal networks, and prediction heads.
- **tools**
  - **train_net.py**: Training script.
  - **test_net.py**: Testing script.
  - **ckpt_surgery.py**: Surgery on checkpoints.
  - **run_experiments.py**: Running experiments across many seeds.
  - **run_base_training.py**: Same as `run_experiments.py` but for base trainings and without seeds.
  - **aggregate_seeds.py**: Aggregating results from many seeds.
  - **collect_metrics.py**: Aggregate results from many seeds and compute mean and standard deviation.
- **wrapper_base_training.py**: Wrapper for script `run_base_training.py`.
- **wrapper_fine_tuning.py**: Wrapper for script `run_experiments.py`.
- **wrapper_inference.py**: Easy parametrization for inference.


## Dataset Preparation
Exemplary dataset preparation for COCO. For datasets Pascal VOC and LVIS please refer to [Dataset Preparation](https://github.com/ucbdrive/few-shot-object-detection#data-preparation) of the original repository.

### Base Dataset
Create symlinks of your COCO data to the `datasets` directory of the repository (`<FSDET_ROOT>/dataset/`). The expected dataset structure is:
```
├── coco
│   ├── annotations
│       ├── instances_train2014.json
│       ├── instances_val2014.json
│   ├── train2014
│   ├── val2014
```  
See [here](datasets/README.md#base-datasets) for more information on base datasets.

### Few-Shot Dataset
We use COCO 2014 and extract 5k images from the val set for evaluation and use the rest for training.

Create a directory `cocosplit` inside the `datasets` directory of the repository. Its expected structure is:
```
├── cocosplit
│   ├── datasetplit
│       ├── trainvalno5k.json
│       ├── 5k.json
```
See [here](datasets/README.md#few-shot-datasets) for more information on few-shot datasets.

Download the [dataset split files](http://dl.yf.io/fs-det/datasets/cocosplit/datasplit/) and put them into `datasetsplit` directory.

Create few-shot data, e.g. for coco voc_nonvoc split with 10 shots and five seed groups:
``` bash
python3 -m datasets.prepare_coco_few_shot --dataset coco --class-split voc_nonvoc --shots 10 --seeds 0 4
```
Following arguments are accepted by `prepare_coco_few_shot.py`:
* --dataset: dataset used (e.g. `coco`, `isaid`, etc.)
* --class-split: class split into base classes and novel classes (e.g. `voc_nonvoc` for dataset coco)
* --shots: list of shots
* --seeds: Single seed or a range of seeds with both, start and end being inclusive!

You may also download existing seeds [here](http://dl.yf.io/fs-det/datasets/cocosplit/)

## Custom Dataset
In general, it's recommended to preprocess the dataset annotations to be in the same format as the MS-COCO dataset, since those restrictions allow for re-using existant code fragments. For that reason, we further assume that the dataset is already in a coco-like format.
1. For the new dataset, add entries to following config dictionaries of `fsdet/config/defaults.py`: `DATA_DIR`, `DATA_SAVE_PATH_PATTERN`, `CONFIG_DIR_PATTERN`, `CONFIG_CKPT_DIR_PATTERN`, `TRAIN_SPLIT`, `TEST_SPLIT`, `TRAIN_IMG_DIR`, `TEST_IMG_DIR`, `TRAIN_ANNOS` and `TEST_ANNOS`
2. For dataset preparation, at `datasets/prepare_coco_few_shot.py`, for the new dataset:
    1. Add a case to the method `get_data_path`
    2. Add an entry to the choices of the `--dataset` argument
3. For setting correct Meta Datasets and mappings to annotations:
    1. In `fsdet/data/builtin_meta.py` adjust the method `_get_builtin_metadata` to add cases for `<DATASET>` and `<DATASET>_fewshot` with the approprite call of `_get_cocolike_instances_meta` and `_get_cocolike_fewshot_instances_meta`, respectively
    2. In `fsdet/data/builtin.py`, add a new register method and call that method at the bottom of the file
    3. In `fsdet/data/__init__.py`, import the newly created register method
4. In the surgery (`tools/ckpt_surgery.py`):
    1. If your dataset is "coco-like", you may just add the dataset name to the first case of the main entry point (and the TOTAL_CLASSES dict - for sanity checks).
    2. Add the dataset to the choices of the `--dataset` argument
5. For Training and Testing
    1. In `tools/test_net.py` and `tools/train_net.py`: add a case for the evaluator_type
    2. In `tools/run_base_training.py`: add the dataset to choices of `--dataset` argument, add dataset-specific constants in a case at the beginning of the `get_config` method, probably adjust base training config pattern and folder structures for configs and checkpoints
    3. In `tools/run_experiments.py`: probably need to adjust config patterns and folder structures for configs and checkpoints as well.

## Training
Note: You can also download the ImageNet pretrained backbones [ResNet-50](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl), [ResNet-101](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl) before starting to train, so it doesn't have to be downloaded prior to every training you start. You can put it into a directory `<FSDET_ROOT>/pretrained` and then adjust the `WEIGHTS`-parameter in the training configs.

See the original documentation on the [TFA training procedure](docs/TRAIN_INST.md) for more detailed information.

Note: The original workflow was to modify previously created dummy-configs. Instead, we now create fresh configs every time a new trainign is started, no config is read in and then modified. For those purpose, we refactored the existing script `tools/run_experiments.py` to parametrize fine-tunings and created a new script `tools/base_training.py` for easy parametrization of base-trainings. Further information on both script can be found in the sections [Base-Training](#base-training) and [Fine-Tuning](#fine-tuning)

### Pre-trained Models
Benchmark results and pretrained models are available [here](docs/MODEL_ZOO.md). More models and configs are available [here](fsdet/model_zoo/model_zoo.py)

### Base-Training
Since we do not use pre-defined configs and calculate fresh configs for each new training, it's best to run base trainings with the script `tools/run_base_training.py` (or easily parametrized with the corresponding wrapper `wrapper_base_training.py`):
```bash
python3 -m tools.run_base_training --dataset isaid --class-split vehicle_nonvehicle --gpu-ids 0 1 2 3 --layers 50 --bs 16 --lr 0.02
```
Following arguments are supported:
* --dataset: dataset to be used (e.g. `coco` or `isaid`)
* --class-split: the class split used. Has to be a existant key in the dictionary CLASS_SPLIT[dataset] of the file class_splits.py
* --gpu-ids: gpu ids to run the base-training on. Accepts multiple gpu ids, sets the internally used --num-gpus argument and the CUDA_VISIBLE_DEVICES environment variable appropriately
* --layers: ResNet backbone layers (default: `50`)
* --bs: total batch size, not per gpu! (default: `16`)
* --lr: learning rate (default: `0.02` for batch size 16). Set to `-1` for automatically linear scaling depending on the batch size
* --override-config: force overriding of already existant configs
* --num-threads: limit the amount of threads using `OMP_NUM_THREADS` environment variable. (Default: `1`) 

### Fine-Tuning
Similar to the base-trainings, fine-tunings are best run with the appropriate script, `tools/run_experiments.py`. We modified the original script to create a fresh config for each training and to not read in existing configs and modifying them, which required the existance of an example config for every possible configuration. This way, we are more flexible and the config/-directory is more clean since we just store configs we really need. Since the amount of possible arguments is very large, we recommend using the corresponding wrapper `wrapper_fine_tuning.py` for starting fine-tunings. The most important arguments are:
* --dataset, --class-split, --gpu-ids, --num-threads, --layers, --bs and --override-config work the same way as for the base-training
* --classfier: use regular `fc` or `cosine` classifier 
* --tfa: use two-stage fine-tuning approach (Trains a net on only novel classes to obtain novel class initialization for regular fine-tuning), turned off by default. When turned off, this equals the `randinit` surgery type.
* --unfreeze: unfreeze the whole net (backbone + proposal generator + roi head convs + roi head fcs)
* Unfreeze certain parts of the net:
  * --unfreeze-backbone: unfreeze backbone
  * --unfreeze-proposal-generator: unfreeze proposal generator (e.g. RPN)
  * --unfreeze-roi-box-head-convs: unfreeze certain ROI-Box-Head conv layers (if any). Set indices starting by 1.
  * --unfreeze-roi-box-head-fcs: unfreeze certain ROI-Box-Head fc layers (if any). Set indices starting by 1.
* --double-head: experimental setting with separate heads for base classes and novel classes. Requires the usage of exact two FC layers in the ROI Box Head and requires the heads to be split at index 2 (config ROI_BOX_HEAD.SPLIT_AT_FC)
* --shots: shot parameter(s)
* --seeds: seed(s) representing different data groups (single seed or two seeds, representing a range with both start and end being inclusive!)
* --explicit-seeds: Interpret the list of seeds as explicit seeds rather than as a range of seeds.
* --lr: learning rate (Default: `0.001` for batch size 16). Set to -1 for automatic linear scaling dependent on batch size.
* --override-surgery: rerun surgery even if surgery model already exists (e.g. necessary when using same settings but different `double_head` setting)
* The maximum iteration, the learning rate decay steps and the checkpoint interval may be overridden using the arguments --max-iter, --lr-decay-steps and --ckpt-interval, respectively. If not specified, hard-coded values depending on dataset and shot are used.

## Inference
Inference can either be run directly in the command line via:
```bash
python3 -m tools.test_net --num-gpus 8 --config-file configs/<path-to-config>/<config>.yaml --eval-only
```
or by using the corresponding wrapper `wrapper_inference.py` for easy parametrization.

Note: 
* --eval-only evaluates just the last checkpoint. Add --eval-iter to evaluate a certain checkpoint iteration. Use --eval-all to evaluate all saved checkpoints.
* --opts can be used to override some test-specific configs without having to modify the config file directly

## Aggregate Results of many Seeds
Due to the heavily modified repository workflow (including file and directory names as well as the directory hierarchy), it's unclear if the script `tools/aggregate_seeds.py` still works. Thus, we recommend using the script `tools/collect_metrics.py` which is directly adapted to the actual repository workflow. Adjust the variables to match your training's configuration and run:

```bash
python3 -m tools.collect_metrics
```

## Legacy Notes
We sample multiple groups of few-shot training examples for multiple runs of the experiments and report evaluation results on both the base classes and the novel classes.

We also provide benchmark results and pre-trained models for our two-stage fine-tuning approach (TFA). In TFA, we first train the entire object detector on the data-abundant base classes, and then only fine-tune the last layers of the detector on a small balanced training set.

The code has been upgraded to detectron2 v0.2.1.  If you need the original released code, please checkout the release [v0.1](https://github.com/ucbdrive/few-shot-object-detection/tags) in the tag.

If you find this repository useful for your publications, please consider citing our paper.
```angular2html
@article{wang2020few,
    title={Frustratingly Simple Few-Shot Object Detection},
    author={Wang, Xin and Huang, Thomas E. and  Darrell, Trevor and Gonzalez, Joseph E and Yu, Fisher}
    booktitle = {International Conference on Machine Learning (ICML)},
    month = {July},
    year = {2020}
}
