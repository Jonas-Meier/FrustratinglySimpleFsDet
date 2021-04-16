import os


def main():
    dataset = "coco"  # coco, isaid
    coco_class_split = "voc_nonvoc"  # voc_nonvoc, none_all
    isaid_class_split = "vehicle_nonvehicle"  # vehicle_nonvehicle, none_all, experiment1, experiment2, experiment3
    gpu_ids = [0]
    layers = 50  # 50, 101
    bs = 16
    lr = 0.02  # 0.02 for bs=16. Set to -1 for automatic linear scaling!
    override_config = True
    # --num-threads=1
    if dataset == "coco":
        class_split = coco_class_split
    elif dataset == "isaid":
        class_split = isaid_class_split
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    run_base_training(dataset, class_split, gpu_ids, layers, bs, lr, override_config)


def run_base_training(dataset, class_split, gpu_ids, layers, bs, lr=-1.0, override_config=False):
    base_cmd = "python3 -m tools.run_base_training"
    override_config_str = ' --override-config' if override_config else ''
    cmd = "{} --dataset {} --class-split {} --gpu-ids {} --layers {} --bs {} --lr {}{}"\
        .format(base_cmd, dataset, class_split, comma_sep(gpu_ids), layers, bs, lr, override_config_str)
    os.system(cmd)


def comma_sep(elements):
    res = ''
    if not isinstance(elements, (list, tuple)):
        return str(elements)
    assert len(elements) > 0, "need at least one element in the collection {}!".format(elements)
    if len(elements) == 1:
        return str(elements[0])
    for element in elements:
        res += '{},'.format(str(element))
    return res[:-1]  # remove trailing space


if __name__ == '__main__':
    main()
