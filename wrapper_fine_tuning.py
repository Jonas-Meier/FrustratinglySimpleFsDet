import os


def main():
    dataset = "coco"  # coco, isaid
    coco_class_split = "voc_nonvoc"  # voc_nonvoc, none_all
    isaid_class_split = "vehicle_nonvehicle"  # vehicle_nonvehicle, none_all, experiment1, experiment2, experiment3
    gpu_ids = [0]
    layers = 50  # 50, 101
    bs = 16
    lr = 0.001  # 0.001 for bs=16. Set to -1 for automatic linear scaling!
    shots = [10]
    seeds = [0]  # single seed or two seeds representing a range, 2nd argument inclusive!
    classifier = 'fc'  # fc, cosine
    tfa = False  # False: randinit surgery
    unfreeze = False  # False: freeze feature extractor while fine-tuning
    # override-config=True
    # override-surgery=True
    # --num-threads=1
    if dataset == "coco":
        class_split = coco_class_split
    elif dataset == "isaid":
        class_split = isaid_class_split
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    run_fine_tuning(dataset, class_split, shots, seeds, gpu_ids, layers, bs, tfa, unfreeze, classifier, lr)


def run_fine_tuning(dataset, class_split, shots, seeds, gpu_ids, layers, bs,
                    tfa=False, unfreeze=False, classifier='fc', lr=-1.0):
    base_cmd = "python3 -m tools.run_experiments"
    tfa_str = ' --tfa' if tfa else ''
    unfreeze_str = ' --unfreeze' if unfreeze else ''
    cmd = "{} --dataset {} --class-split {} --shots {} --seeds {} " \
          "--gpu-ids {} --layers {} --bs {} --lr {} --classifier {}{}{}"\
        .format(base_cmd, dataset, class_split, separate(shots, ' '), separate(seeds, ' '),
                separate(gpu_ids, ','), layers, bs, lr, classifier, tfa_str, unfreeze_str)
    os.system(cmd)


# note: separate(elements, ' ') == *elements
def separate(elements, separator):
    res = ''
    if not isinstance(elements, (list, tuple)):
        return str(elements)
    assert len(elements) > 0, "need at least one element in the collection {}!".format(elements)
    if len(elements) == 1:
        return str(elements[0])
    for element in elements:
        res += '{}{}'.format(str(element), separator)
    return res[:-1]  # remove trailing separator



if __name__ == '__main__':
    main()
