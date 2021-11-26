import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import torch
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import BoxMode
from detectron2.utils.logger import create_small_table

from fsdet.evaluation.evaluator import DatasetEvaluator
from class_splits import CLASS_SPLITS, COMPATIBLE_DATASETS, CLASS_NAME_TRANSFORMS, get_ids_from_names

PRECISION_METRIC_TO_IOU_THR = {
    "AP": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "APs": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "APm": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "APl": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "AP5": [0.05],
    "AP10": [0.10],
    "AP15": [0.15],
    "AP20": [0.20],
    "AP25": [0.25],
    "AP30": [0.30],
    "AP35": [0.35],
    "AP40": [0.40],
    "AP45": [0.45],
    "AP50": [0.50],
    "AP55": [0.55],
    "AP60": [0.60],
    "AP65": [0.65],
    "AP70": [0.70],
    "AP75": [0.75],
    "AP80": [0.80],
    "AP85": [0.85],
    "AP90": [0.90],
    "AP95": [0.95],
}
PRECISION_METRIC_TO_AREA = {
    "APs": "small",
    "APm": "medium",
    "APl": "large",
    **{m: "all" for m in ["AP", "AP5", "AP10", "AP15", "AP20", "AP25", "AP30", "AP35", "AP40", "AP45", "AP50",
                          "AP55", "AP60", "AP65", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95"]}
}


class COCOEvaluator(DatasetEvaluator):
    """
    Evaluate instance detection outputs using COCO's metrics and APIs.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None, dataset='coco', file_suffix=""):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                    so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True):
                if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._precision_summary_metrics = cfg.TEST.METRICS.PRECISION.SUMMARY
        self._precision_per_class_metrics = cfg.TEST.METRICS.PRECISION.PER_CLASS
        self._distributed = distributed
        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)  # just to be sure
        self._dataset_name = dataset_name
        # save file names
        # raw detections passed to this class
        self._raw_predictions_file = os.path.join(self._output_dir, "instances_predictions{}.pth".format(file_suffix))
        # coco detections used to build coco_eval object
        self._coco_detections_file = os.path.join(self._output_dir, "coco_instances_results{}.json".format(file_suffix))
        self._summary_file = os.path.join(self._output_dir, "summary_results{}.txt".format(file_suffix))
        open(self._summary_file, 'w').close()  # create empty file, or delete content is existent

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        ### handle the rare case where train dataset (and/or class split) differ ###
        train_dataset_name = cfg.DATASETS.TRAIN[0]
        train_metadata = MetadataCatalog.get(train_dataset_name)
        # Note: the class names and ids in 'self._metadata' are the same as is the dataset we evaluate on.
        # However, the model was trained on the dataset and class split as in 'train_metadata'
        if self._metadata.dataset == train_metadata.dataset:
            if self._metadata.class_split == train_metadata.class_split:
                # Nothing to do!
                self._train_ind_to_test_ind = {i: i for i in self._metadata.thing_dataset_id_to_contiguous_id.values()}
            else:
                # TODO: would we want to allow an inference on only base classes (base training) where base classes are
                #  the same, but ft-classes differ?
                raise ValueError("Cannot evaluate when the model was trained on dataset {} and class split {}, but the "
                                 "evaluation is to be done on dataset {} and class split {}"
                                 .format(self._metadata.dataset, self._metadata.class_split,
                                         train_metadata.dataset, train_metadata.class_split))
        else:
            # Case 1: Both datasets are different but are compatible (e.g. have exact same classes)
            if any(self._metadata.dataset in comp_dsets and train_metadata.dataset in comp_dsets
                   for comp_dsets in COMPATIBLE_DATASETS):
                self._train_ind_to_test_ind = {i: i for i in self._metadata.thing_dataset_id_to_contiguous_id.values()}
            # Case 2: Both datasets are different but in some sense incompatible (e.g. have same underlying classes but
            #  probably different names and/or order). In this case, we need a deposited mapping of the one dataset's
            #  classes to the other dataset's classes.
            else:
                assert ((train_metadata.dataset, train_metadata.class_split),
                        (self._metadata.dataset, self._metadata.class_split)) in CLASS_NAME_TRANSFORMS
                train_classes_to_test_classes = \
                    CLASS_NAME_TRANSFORMS[((train_metadata.dataset, train_metadata.class_split),
                                           (self._metadata.dataset, self._metadata.class_split))]
                train_ids_to_test_ids = {
                    get_ids_from_names(train_metadata.dataset, [train_cls])[0]: get_ids_from_names(self._metadata.dataset, [test_cls])[0]
                    for train_cls, test_cls in train_classes_to_test_classes.items()
                }
                self._train_ind_to_test_ind = {
                    train_metadata.thing_dataset_id_to_contiguous_id[train_id]: self._metadata.thing_dataset_id_to_contiguous_id[test_id]
                    for train_id, test_id in train_ids_to_test_ids.items()
                }
        ###
        # TODO: problematic when using class_split 'none_all' (or one containing 'baseball'...)
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
            or "novel" in dataset_name
        self._class_split = self._metadata.class_split
        # Note: we use 'thing_ids' over 'all_ids' because 'meta_coco' will override 'thing_ids' appropriately
        self._all_class_ids = self._metadata.thing_ids
        self._base_class_ids = self._metadata.get("base_ids", None)
        self._novel_class_ids = self._metadata.get("novel_ids", None)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
                # The prediction's category_id is one of the classes of the training dataset. If the training
                # and testing dataset differ, we have to adjust this category_id to match the test dataset.
                # Note: The category_id here is merely just an index and is later (by method '_eval_predictions')
                # transformed to an actual id
                for inst in prediction["instances"]:
                    inst["category_id"] = self._train_ind_to_test_ind[inst["category_id"]]
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            with PathManager.open(self._raw_predictions_file, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            self._logger.info("Saving results to {}".format(self._coco_detections_file))
            with PathManager.open(self._coco_detections_file, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        if self._is_splits:
            self._results["bbox"] = {}
            for split, classes, names in [
                    ("all", self._all_class_ids, self._metadata.get("thing_classes")),
                    ("base", self._base_class_ids, self._metadata.get("base_classes")),
                    ("novel", self._novel_class_ids, self._metadata.get("novel_classes"))]:
                if "all" not in self._dataset_name and \
                        split not in self._dataset_name:
                    continue
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, "bbox", classes, precision_metrics=self._precision_summary_metrics
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res_ = self._derive_coco_results(
                    coco_eval, "bbox", class_names=names, split=split
                )
                res = {}
                for metric in res_.keys():
                    if len(metric) <= 4:
                        if split == "all":
                            res[metric] = res_[metric]
                        elif split == "base":
                            res["b"+metric] = res_[metric]
                        elif split == "novel":
                            res["n"+metric] = res_[metric]
                self._results["bbox"].update(res)

            # add "AP" if not already in
            if "AP" not in self._results["bbox"]:
                if "nAP" in self._results["bbox"]:
                    self._results["bbox"]["AP"] = self._results["bbox"]["nAP"]
                else:
                    self._results["bbox"]["AP"] = self._results["bbox"]["bAP"]
        else:
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, self._coco_results, "bbox", precision_metrics=self._precision_summary_metrics
                )
                if len(self._coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )
            res = self._derive_coco_results(
                coco_eval, "bbox",
                class_names=self._metadata.get("thing_classes")
            )
            self._results["bbox"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None, split=''):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """
        def log_info_and_append(file, str):
            with open(file, 'a') as f:
                f.write(str + 2*'\n')
            self._logger.info(str)

        def get_per_category_ap_table(coco_eval, class_names, iou_low=0.5, iou_high=0.95):
            # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
            def _get_thr_ind(coco_eval, thr):
                ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                               (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
                iou_thr = coco_eval.params.iouThrs[ind]
                assert np.isclose(iou_thr, thr)
                return ind
            precisions = coco_eval.eval["precision"]
            # precision has dims (iou, recall, cls, area range, max dets)
            assert len(class_names) == precisions.shape[2], "{},{}".format(len(class_names), precisions.shape[2])
            ind_lo = _get_thr_ind(coco_eval, iou_low)
            ind_hi = _get_thr_ind(coco_eval, iou_high)
            results_per_category = []
            for idx, name in enumerate(class_names):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                precision = precisions[ind_lo:(ind_hi + 1), :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float("nan")
                results_per_category.append(("{}".format(name), float(ap * 100)))
            # tabulate it
            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)])
            if iou_low == 0.5 and iou_high == 0.95:
                ap_name = "AP"
            elif iou_low == iou_high:
                ap_name = "AP{}".format(round(iou_low * 100))  # e.g. AP50
            else:  # very rare case
                ap_name = "AP{}-{}".format(round(iou_low * 100), round(iou_high * 100))  # e.g. AP65-80
            table = tabulate(
                results_2d,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category", ap_name] * (N_COLS // 2),
                numalign="left",
            )
            return results_per_category, table

        metrics = self._precision_summary_metrics  # ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        split_str = '({} classes)'.format(split) if split != '' else split
        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100) \
                for idx, metric in enumerate(metrics)
        }
        tmp_str = "Evaluation results for {} {}: \n".format(iou_type, split_str) + create_small_table(results)
        log_info_and_append(self._summary_file, tmp_str)

        if not class_names:
            return results

        # get per-class metrics, log the results, append them to the log file and update 'results'
        for metric in self._precision_per_class_metrics:
            iou_low = PRECISION_METRIC_TO_IOU_THR[metric][0]
            iou_high = PRECISION_METRIC_TO_IOU_THR[metric][-1]
            results_per_category, table = get_per_category_ap_table(coco_eval, class_names, iou_low, iou_high)
            tmp_str = "Per-category {} {} {}: \n".format(iou_type, metric, split_str) + table
            log_info_and_append(self._summary_file, tmp_str)
            results.update({"{}-".format(metric) + name: ap for name, ap in results_per_category})

        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        results.append(result)
    return results


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, catIds=None,
                                  precision_metrics=None, recall_metrics=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if recall_metrics:
        raise NotImplementedError("Custom Recall metrics are not supported yet!")

    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) & (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds

    # Set just the needed iou thresholds to speed up evaluation
    all_iou_thrs = [elem for sublist in list(map(lambda x: PRECISION_METRIC_TO_IOU_THR[x], precision_metrics)) for elem in sublist]
    iou_thr = sorted(set(all_iou_thrs))  # unique and sorted iou thresholds
    coco_eval.params.iouThrs = np.array(iou_thr)

    coco_eval.evaluate()
    coco_eval.accumulate()
    # TODO: probably redirect stdout to self._summary_file, in order to also get the output of coco_eval.summarize()
    #  into the text file?
    precisions = coco_eval.eval["precision"]
    stats = np.zeros((len(precision_metrics),))
    for i, precision_metric in enumerate(precision_metrics):
        lo_iou_ind = _get_thr_ind(coco_eval, PRECISION_METRIC_TO_IOU_THR[precision_metric][0])
        hi_iou_ind = _get_thr_ind(coco_eval, PRECISION_METRIC_TO_IOU_THR[precision_metric][-1])
        area_ind = [i for i, aRng in enumerate(coco_eval.params.areaRngLbl) if aRng == PRECISION_METRIC_TO_AREA[precision_metric]]
        max_dets_ind = [i for i, mDet in enumerate(coco_eval.params.maxDets) if mDet == coco_eval.params.maxDets[2]]
        precision = precisions[lo_iou_ind:(hi_iou_ind + 1), :, :, area_ind, max_dets_ind]
        if len(precision[precision > -1]) == 0:
            mean_prec = -1
        else:
            mean_prec = np.mean(precision[precision > -1])
        stats[i] = mean_prec
    coco_eval.stats = stats

    # Note: we don't call 'summarize' but, instead, compute the metrics, given by the config still saving them in the
    # same order in coco_eval.stats, s.t. the caller of this method can obtain the metrics he wants by accessing the
    # elements of coco_eval.stats in the same order as in the config
    # coco_eval.summarize()

    return coco_eval
