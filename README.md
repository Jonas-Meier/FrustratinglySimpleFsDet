[//]: # (This may be the most platform independent comment)

# Few-Shot Object Detection (FsDet)

FsDet contains the official few-shot object detection implementation of the ICML 2020 paper [Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/abs/2003.06957).

Original [Repository](https://github.com/ucbdrive/few-shot-object-detection).

![TFA Figure](https://user-images.githubusercontent.com/7898443/76520006-698cc200-6438-11ea-864f-fd30b3d50cea.png)

## Setup
This repository has been successfully tested with following configuration:
1. CUDA 10.1 (CUDA 10.0 and 10.2 should work as well)
2. cuDNN 7.6.3 for CUDA 10.1
3. gcc/g++ 7.5

## Variable Denotation
* We assume the repository to be located at `<FSDET_ROOT>` (e.g. `/home/<user>/workspace/frustratingly-simple-fsdet`)
* CUDA is located at `CUDA_ROOT` (e.g. `/home/<user>/cuda-10.1`)

## Build
1. Create an environment (e.g. with conda) and activate it
``` bash
conda create --name fs-fsdet
conda activate fs-fsdet
```
2.Install PyTorch, depending on your local CUDA version (e.g. PyTorch 1.6 for CUDA 10.1)
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

## Dataset Preparation
Exemplary dataset preparation for COCO. For datasets Pascal VOC and LVIS please refer to [Dataset Preparation](https://github.com/ucbdrive/few-shot-object-detection#data-preparation) of the original repository.

Create symlinks of your COCO data to the `datasets` directory of the repository (`<FSDET_ROOT>/dataset/`). The expected dataset structure is:
```
├── coco
│   ├── annotations
│       ├── instances_train2014.json
│       ├── instances_val2014.json
│   ├── train2014
│   ├── val2014
```  



## About the repository, legacy notes
We sample multiple groups of few-shot training examples for multiple runs of the experiments and report evaluation results on both the base classes and the novel classes. These are described in more detail in [Data Preparation](#data-preparation).

We also provide benchmark results and pre-trained models for our two-stage fine-tuning approach (TFA). In TFA, we first train the entire object detector on the data-abundant base classes, and then only fine-tune the last layers of the detector on a small balanced training set. See [Models](#models) for our provided models and [Getting Started](#getting-started) for instructions on training and evaluation.

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
```


[//]: # (Old documentation below this comment! TODO: adjust an remove unnecessary parts!)










## Code Structure
- **configs**: Configuration files
- **datasets**: Dataset files (see [Data Preparation](#data-preparation) for more details)
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
  - **aggregate_seeds.py**: Aggregating results from many seeds.


## Data Preparation
We evaluate our models on three datasets:
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/): We use the train/val sets of PASCAL VOC 2007+2012 for training and the test set of PASCAL VOC 2007 for evaluation. We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 3 random splits. The splits can be found in [fsdet/data/datasets/builtin_meta.py](fsdet/data/datasets/builtin_meta.py).
- [COCO](http://cocodataset.org/): We use COCO 2014 and extract 5k images from the val set for evaluation and use the rest for training. We use the 20 object classes that are the same with PASCAL VOC as novel classes and use the rest as base classes.
- [LVIS](https://www.lvisdataset.org/): We treat the frequent and common classes as the base classes and the rare categories as the novel classes.

See [datasets/README.md](datasets/README.md) for more details.


## Models
We provide a set of benchmark results and pre-trained models available for download in [MODEL_ZOO.md](docs/MODEL_ZOO.md).


## Getting Started

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from
  [model zoo](fsdet/model_zoo/model_zoo.py),
  for example, `COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml`.
2. We provide `demo.py` that is able to run builtin standard models. Run it with:
```
python3 -m demo.demo --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml \
  --input input1.jpg input2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS fsdet://coco/tfa_cos_1shot/model_final.pth
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

For details of the command line arguments, see `demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run __on your webcam__, replace `--input files` with `--webcam`.
* To run __on a video__, replace `--input files` with `--video-input video.mp4`.
* To run __on cpu__, add `MODEL.DEVICE cpu` after `--opts`.
* To save outputs to a directory (for images) or a file (for webcam or video), use `--output`.

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

For more detailed instructions on the training procedure of TFA, see [TRAIN_INST.md](docs/TRAIN_INST.md).

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
