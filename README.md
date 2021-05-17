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
  - **run_base_training.py**: Same as 'run_experiments.py' but for base trainings.
  - **aggregate_seeds.py**: Aggregating results from many seeds.


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

Create few-shot data with 
``` bash
cd datasets/
python prepare_coco_few_shot.py
```
or
``` bash
python3 -m datasets.prepare_coco_few_shot
```
Following arguments are accepted by `prepare_coco_few_shot.py`:
* --dataset: dataset used (e.g. `coco`, `isaid`, etc.)
* --class-split: class split into base classes and novel classes (e.g. `voc_nonvoc` for dataset coco)
* --shots: list of shots
* --seeds: range of seeds. Start and end are both inclusive!

You may also download existing seeds [here](http://dl.yf.io/fs-det/datasets/cocosplit/)

## Custom Dataset
In general, it's recommended to preprocess the dataset annotations to be in the same format as the MS-COCO dataset, since those restrictions allow for re-using existant code fragments. For that reason, we further assume that the dataset is already in a coco-like format.
1. For the new dataset, add entries to following config dictionaries of `fsdet/config/defaults.py`: `DATA_DIR`, `DATA_SAVE_PATH_PATTERN`, `CONFIG_DIR_PATTERN`, `CONFIG_CKPT_DIR_PATTERN`, `TRAIN_SPLIT`, `TEST_SPLIT`, `TRAIN_IMG_DIR`, `TEST_IMG_DIR`, `TRAIN_ANNOS` and `TEST_ANNOS`
2. For dataset preparation, at `datasets/prepare_coco_few_shot.py`, for the new dataset:
    1. Add a case to the method `get_data_path`
    2. Add an entry to the choices of the `--dataset` argument
3. For setting correct Meta Datasets and mappings to annotations:
    1. In `fsdet/data/builtin_meta.py` adjust the method `_get_builtin_metadata` to add cases for `\<DATASET\>` and `\<DATASET\>_fewshot` with the approprite call of `_get_cocolike_instances_meta` and `_get_cocolike_fewshot_instances_meta`, respectively
    2. In `fsdet/data/builtin.py`, add a new register method and call that method at the bottom of the file
    3. In `fsdet/data/__init__.py`, import the newly created register method
4. In the surgery (`tools/ckpt_surgery.py`):
    1. Add a new case to the main entry point and set the following variables: `NOVEL_CLASSES`, `BASE_CLASSES`, `IDMAP` and `TAR_SIZE`
    2. Add the dataset to the choices of the `--dataset` argument
5. For Training and Testing
    1. In `tools/test_net.py` and `tools/train_net.py`: add a case for the evaluator_type
    2. In `tools/run_base_training.py`: add the dataset to choices of `--dataset` argument, add dataset-specific constants in a case at the beginning of the `get_config` method, probably adjust base training config pattern and folder structures for configs and checkpoints
    3. In `tools/run_experiments.py`: probably need to adjust config patterns and folder structures for configs and checkpoints as well.

## Training

Note: You can also download the ImageNet pretrained backbones [ResNet-50](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl), [ResNet-101](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl) before starting to train, so it doesn't have to be downloaded prior to every trainng you start. You can put it into a directory `<FSDET_ROOT>/pretrained` and then adjust the `WEIGHTS`-parameter in the training configs.

See the original documentation on the [TFA training procedure](docs/TRAIN_INST.md) for more detailed information.

### Pre-trained Models
Benchmark results and pretrained models are available [here](docs/MODEL_ZOO.md). More models and configs are available [here](fsdet/model_zoo/model_zoo.py)


[//]: # (Old documentation below this comment! TODO: adjust an remove unnecessary parts!)


### Training & Evaluation in Command Line

To train a model, run
```angular2html
python3 -m tools.train_net --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml
```

To evaluate the trained models, run
```angular2html
python3 -m tools.test_net --num-gpus 8 \
        --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_1shot.yaml \
        --eval-only
```

### Multiple Runs

For ease of training and evaluation over multiple runs, we provided several helpful scripts in `tools/`.

You can use `tools/run_experiments.py` to do the training and evaluation. For example, to experiment on 30 seeds of the first split of PascalVOC on all shots, run
```angular2html
python3 -m tools.run_experiments --num-gpus 8 \
        --shots 1 2 3 5 10 --seeds 0 30 --split 1
```

After training and evaluation, you can use `tools/aggregate_seeds.py` to aggregate the results over all the seeds to obtain one set of numbers. To aggregate the 3-shot results of the above command, run
```angular2html
python3 -m tools.aggregate_seeds --shots 3 --seeds 30 --split 1 \
        --print --plot
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
