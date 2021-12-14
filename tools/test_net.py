"""
Detection Testing Script.

This scripts reads a given config file and runs the evaluation.
It is an entry point that is made to evaluate standard models in FsDet.

In order to let one script support evaluation of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""
import numpy as np
import torch

from fsdet.data.transforms.augmentations_impl import build_augmentation
from fsdet.modeling import GeneralizedRCNNWithTTA

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup
from fsdet.evaluation.evaluator import inference_context

import detectron2.utils.comm as comm
import json
import logging
import os
import time
from collections import OrderedDict
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.engine import hooks, launch
from fsdet.evaluation import (
    COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, verify_results)

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None, file_suffix=""):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if cfg.TEST.FILTER_EMPTY_ANNOTATIONS:
            output_folder = os.path.join(output_folder, "only_nonempty_imgs")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in cfg.DATASETS.COCOLIKE_DATASETS:
            dataset = MetadataCatalog.get(dataset_name).dataset
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder, dataset=dataset, file_suffix=file_suffix)
            )
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_augmentations(cls, cfg, is_train):
        if cfg.INPUT.AUG.TYPE == 'default':
            return None  # Trigger detectron2's creation of default augmentations
        else:
            assert cfg.INPUT.AUG.TYPE == 'custom'
        if is_train:
            return [build_augmentation(aug, cfg, is_train) for aug in cfg.INPUT.AUG.PIPELINE]
        else:
            # Similar to Detectron2, we hard-code the resize transform to be the only transform used during testing
            #  (see detectron2/data/dataset_mapper.py:from_config and
            #  detectron2/data/detection_utils.py:build_augmentation)
            return [build_augmentation("ResizeShortestEdgeLimitLongestEdge", cfg, is_train)]

    @classmethod
    def test_with_TTA(cls, cfg, model, file_suffix=""):
        # TODO: probably also add this method to 'train_net' to register an EvalHook with it
        #  (see Detectron2's train_net script)
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                # TODO: why do we need to pass a file suffix to both, the test method AND the evaluators?
                #  -> test normally calls cls.build_evaluators, but only if the passed evaluators are none.
                #     since we build evaluators and then pass them to test, we would rather need to set the file_suffix
                #     in th evaluators and would then not need to pass them to test!
                #  --> probably adjust the strange call flow of the file_suffix!
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"), file_suffix=file_suffix
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators, file_suffix=file_suffix)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def export_t_sne_features(cls, cfg, model):
        # TODO: to support multi-gpu, make all data collection, figure creation, etc. inside 'if comm.is_main' or similar!
        debug = False
        ##########
        dataset_name = cfg.DATASETS.TEST[0]  # ugly but should be ok for first experiments with t-sne
        data_loader = cls.build_test_loader(cfg, dataset_name)
        all_box_features = []  # Tensors of size [TEST.DETECTIONS_PER_IMAGE, MODEL.ROI_BOX_HEAD.FC_DIM]
        all_gt_classes = []  # Tensors of size [TEST.DETECTIONS_PER_IMAGE]
        # similar to fsdet.evaluation.evaluator:inference_on_dataset
        with inference_context(model), torch.no_grad():
            for idx, inputs in enumerate(data_loader, start=1):
                if cfg.TEST.TSNE.MAX_NUM_IMGS and idx > cfg.TEST.TSNE.MAX_NUM_IMGS:
                    break
                if debug:
                    print("Processing image {}".format(inputs[0]["file_name"]))
                if idx % 50 == 0:
                    print("Processing image {}/{}".format(idx, len(data_loader)))
                outputs = model(inputs)
                torch.cuda.synchronize()
                box_features, gt_classes = outputs
                all_box_features.append(box_features)
                all_gt_classes.append(gt_classes)
        # TODO: save all box_features and all gt_classes to a file (which format? text format?)
        #  -> for now, we directly create a t-SNE plot instead of saving the features and gt's beforehand
        # (see https://github.com/spmallick/learnopencv/blob/master/TSNE/tsne.py)
        # collect features (only fc features, gt labels are needed later) (stack as np.arrays)
        features = all_box_features[0].cpu().numpy()
        for current_features in all_box_features[1:]:
            features = np.concatenate((features, current_features.cpu().numpy()), axis=0)
        labels = all_gt_classes[0].cpu().numpy()
        for current_labels in all_gt_classes[1:]:
            labels = np.concatenate((labels, current_labels.cpu().numpy()), axis=0)
        assert features.shape[0] == labels.shape[0]
        # call tsne -> returns x and y coordinates
        start = time.time()
        tsne = TSNE(n_components=2).fit_transform(features)
        end = time.time()
        print("Created t-SNE from {} features in {}m {}s".format(features.shape[0], *divmod(int(end - start), 60)))
        tx = tsne[:, 0]
        ty = tsne[:, 1]
        # normalize x- and y- tsne-coordinates to [0,1] range
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
        # visualize normalized tsne points together with the gt labels
        metadata = MetadataCatalog.get(dataset_name)
        class_names = metadata.thing_classes
        colors = plt.cm.hsv(np.linspace(0, 1, len(class_names))).tolist()
        if cfg.TEST.TSNE.WITH_BG:
            class_names.append("Background")  # add it later because we hard-code a color for this class
            colors.append([0, 0, 0])  # black color
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for ind, label in enumerate(class_names):  # add a separate scatter plot for each class
            indices = np.where(labels == ind)[0]
            if debug:
                print("label: {}, #indices: {}".format(label, len(indices)))
            if 0 < cfg.TEST.TSNE.MAX_DOTS_PER_CLASS < len(indices):
                indices = np.random.choice(len(indices), size=cfg.TEST.TSNE.MAX_DOTS_PER_CLASS, replace=False)
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            color = np.array(colors[ind])  # should already be in correct format...
            ax.scatter(current_tx, current_ty, color=color, label=label, s=cfg.TEST.TSNE.DOT_AREA)
        # ncol/nrow useful for datasets with many classes (as coco)
        # loc=best
        # place right of the figure, start on lower left
        ax.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0, ncol=cfg.TEST.TSNE.LEGEND_NCOLS)
        if cfg.TEST.TSNE.SAVE:
            save_path = cfg.TEST.TSNE.SAVE_PATH
            if not save_path:
                print("Error, cannot save figure to an empty path!")
                exit(1)
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, "t_sne_{}.png".format(dataset_name)), bbox_inches="tight")
        if cfg.TEST.TSNE.SHOW:
            plt.subplots_adjust(right=0.7)  # leave space on the right side of the figure for the legend
            plt.show()
        plt.close(fig)


class Tester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = Trainer.build_model(cfg)
        self.check_pointer = DetectionCheckpointer(
            self.model, save_dir=cfg.OUTPUT_DIR
        )

        self.best_res = None
        self.best_file = None
        self.all_res = {}

    def test(self, ckpt_file):
        ckpt = self.check_pointer._load_file(ckpt_file)
        iteration = ckpt["iteration"]  # TODO: probably need to first access the key "model"
        self.check_pointer._load_model(ckpt)
        print("evaluating checkpoint {}".format(ckpt_file))
        res = Trainer.test(self.cfg, self.model, file_suffix="_iter_{}".format(iteration))

        if comm.is_main_process():
            verify_results(self.cfg, res)
            print(res)
            if (self.best_res is None) or (
                self.best_res is not None
                and self.best_res["bbox"]["AP"] < res["bbox"]["AP"]
            ):
                self.best_res = res
                self.best_file = ckpt_file
            print("best results from checkpoint {}".format(self.best_file))
            print(self.best_res)
            self.all_res["best_file"] = self.best_file
            self.all_res["best_res"] = self.best_res
            self.all_res[ckpt_file] = res
            os.makedirs(
                os.path.join(self.cfg.OUTPUT_DIR, "inference"), exist_ok=True
            )
            with open(
                os.path.join(self.cfg.OUTPUT_DIR, "inference", "all_res.json"),
                "w",
            ) as fp:
                json.dump(self.all_res, fp)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        if args.eval_iter != -1:
            # load checkpoint at specified iteration
            iteration = args.eval_iter - 1
            ckpt_file = os.path.join(
                cfg.OUTPUT_DIR, "model_{:07d}.pth".format(iteration)
            )
            iter_str = "_iter_{}".format(iteration)
            resume = False
        else:
            # load checkpoint at last iteration
            ckpt_file = cfg.MODEL.WEIGHTS
            iter_str = "_final"
            resume = True
        ckpt = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            ckpt_file, resume=resume
        )
        if cfg.TEST.TSNE.ENABLED:
            Trainer.export_t_sne_features(cfg, model)
            return
        res_file = "res{}.json".format(iter_str)
        # TODO: remove file_suffix?
        #  -> We would then need to hard-code this file suffix here (for res*.json) and in the evaluator!
        # TODO: (see Detectron2's train_net script) Probably run TTA additionally and not instead?
        if cfg.TEST.AUG.ENABLED:  # Test-time Augmentation (TTA)
            res = Trainer.test_with_TTA(cfg, model, file_suffix=iter_str)
            save_path = os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
        else:  # "regular" inference
            res = Trainer.test(cfg, model, file_suffix=iter_str)
            save_path = os.path.join(cfg.OUTPUT_DIR, "inference")
        if cfg.TEST.FILTER_EMPTY_ANNOTATIONS:
            save_path = os.path.join(save_path, "only_nonempty_imgs")
        if comm.is_main_process():
            verify_results(cfg, res)
            # save evaluation results in json
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, res_file), "w") as fp:
                json.dump(res, fp)
        return res
    elif args.eval_all:
        tester = Tester(cfg)
        all_ckpts = sorted(tester.check_pointer.get_all_checkpoint_files())
        for i, ckpt in enumerate(all_ckpts):
            ckpt_iter = ckpt.split("model_")[-1].split(".pth")[0]
            if ckpt_iter.isnumeric() and int(ckpt_iter) + 1 < args.start_iter:
                # skip evaluation of checkpoints before start iteration
                continue
            if args.end_iter != -1:
                if (
                    not ckpt_iter.isnumeric()
                    or int(ckpt_iter) + 1 > args.end_iter
                ):
                    # skip evaluation of checkpoints after end iteration
                    break
            tester.test(ckpt)
        return tester.best_res
    elif args.eval_during_train:
        tester = Tester(cfg)
        saved_checkpoint = None
        while True:
            if tester.check_pointer.has_checkpoint():
                current_ckpt = tester.check_pointer.get_checkpoint_file()
                if (
                    saved_checkpoint is None
                    or current_ckpt != saved_checkpoint
                ):
                    saved_checkpoint = current_ckpt
                    tester.test(current_ckpt)
            time.sleep(10)
    else:
        if comm.is_main_process():
            print(
                "Please specify --eval-only, --eval-all, or --eval-during-train"
            )


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if args.eval_during_train or args.eval_all:
        args.dist_url = "tcp://127.0.0.1:{:05d}".format(
            np.random.choice(np.arange(0, 65534))
        )
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
