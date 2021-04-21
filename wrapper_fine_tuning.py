import os


def main():
    dataset = "coco"  # coco, isaid
    coco_class_split = "voc_nonvoc"  # voc_nonvoc, none_all
    isaid_class_split = "vehicle_nonvehicle"  # vehicle_nonvehicle, none_all, experiment1, experiment2, experiment3
    gpu_ids = [0]
    num_threads = 1
    bs = 16
    lr = 0.001  # 0.001 for bs=16. Set to -1 for automatic linear scaling!
    shots = [10]
    seeds = [0]  # single seed or two seeds representing a range, 2nd argument inclusive!
    layers = 50  # 50, 101
    classifier = 'fc'  # fc, cosine
    tfa = False  # False: randinit surgery
    # Unfreeze settings. 'unfreeze' setting combines the three single settings.
    # Unfreeze settings are combined with 'or', therefore a part of the feature extractor is unfreezed if
    #  either unfreeze==True OR if the corresponding part is unfreezed
    unfreeze = False  # False: freeze feature extractor (backbone + proposal generator + roi head) while fine-tuning
    unfreeze_backbone = False
    unfreeze_proposal_generator = False
    unfreeze_roi_head = False
    # Override existing config, force re-creation of surgery checkpoint
    override_config = True
    override_surgery = True
    if dataset == "coco":
        class_split = coco_class_split
    elif dataset == "isaid":
        class_split = isaid_class_split
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    run_fine_tuning(dataset, class_split, shots, seeds, gpu_ids, num_threads, layers, bs, lr, tfa,
                    unfreeze, unfreeze_backbone, unfreeze_proposal_generator, unfreeze_roi_head,
                    classifier, override_config, override_surgery)


def run_fine_tuning(dataset, class_split, shots, seeds, gpu_ids, num_threads, layers, bs, lr=-1.0, tfa=False,
                    unfreeze=False, unfreeze_backbone=False, unfreeze_proposal_generator=False, unfreeze_roi_head=False,
                    classifier='fc', override_config=False, override_surgery=False):
    base_cmd = "python3 -m tools.run_experiments"
    tfa_str = ' --tfa' if tfa else ''
    unfreeze_str = ''
    unfreeze_str = unfreeze_str + ' --unfreeze' if unfreeze else unfreeze_str
    unfreeze_str = unfreeze_str + ' --unfreeze-backbone' if unfreeze_backbone else unfreeze_str
    unfreeze_str = unfreeze_str + ' --unfreeze-proposal-generator' if unfreeze_proposal_generator else unfreeze_str
    unfreeze_str = unfreeze_str + ' --unfreeze-roi-head' if unfreeze_roi_head else unfreeze_str
    override_config_str = ' --override-config' if override_config else ''
    override_surgery_str = ' --override-surgery' if override_surgery else ''
    cmd = "{} --dataset {} --class-split {} --shots {} --seeds {} " \
          "--gpu-ids {} --num-threads {} --layers {} --bs {} --lr {} --classifier {}{}{}{}{}"\
        .format(base_cmd, dataset, class_split, separate(shots, ' '), separate(seeds, ' '), separate(gpu_ids, ','),
                num_threads, layers, bs, lr, classifier, tfa_str, unfreeze_str, override_config_str, override_surgery_str)
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
