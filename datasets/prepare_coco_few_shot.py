import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 10],  # TODO: range(1,10) = 1..9!!! 9 not 10 seeds
                        help="Range of seeds")
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data_path = 'datasets/cocosplit/datasplit/trainvalno5k.json'
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    for a in data['annotations']:
        if a['iscrowd'] == 1:
            continue
        anno[a['category_id']].append(a)

    for i in range(args.seeds[0], args.seeds[1]):
        print("Generating seed {}".format(i))
        random.seed(i)
        for c in ID2CLASS.keys():
            print("Generating data for class {} with id {}".format(ID2CLASS[c],c))
            img_ids = {}
            for a in anno[c]:
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]

            sample_shots = []  # annotations
            sample_imgs = []  # images
            for shots in [1, 2, 3, 5, 10, 30]:
                print("Generating {} shot data".format(shots))
                while True:
                    imgs = random.sample(list(img_ids.keys()), shots)  # TODO: unnecessary to take 'shot' images
                    # because they finally only need 'shots' annotations. They have an infinite loop and sample
                    # 'shots' images because they never know how many "useful" annotations per class they will find
                    # per image. Instead, they could just use an infinite loop and always sample one image at a time
                    # until they have enough annotations. This code is confusing!
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:  # TODO: This condition may lead to following:
                            # 1. For k=5 shots and if each image had exactly 2 annotations per class we finally only
                            # have four annotations for that class -> probably too few annotations
                            # 2. In contrast to other approaches, they allow for taking multiple annotations from the
                            # same image (even more: they only want ALL annotations from an image (for a certain class)
                            # or none at all) (as support data) -> unknown consequences
                            continue
                        sample_shots.extend(img_ids[img])  # add all annotations of image with id 'img' with class 'c'
                        sample_imgs.append(id2img[img])  # add the image with id 'img'
                        assert len(sample_imgs) <= len(sample_shots), "Error, got {} images but only {} annotations!".format(len(sample_imgs), len(sample_shots))
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                assert len(sample_shots) == shots, "Wanted {} shots, but only found {} annotations!".format(shots, len(sample_shots))
                new_data = {
                    'info': data['info'],
                    'licenses': data['licenses'],
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                save_path = get_save_path_seeds(data_path, ID2CLASS[c], shots, i)
                new_data['categories'] = new_all_cats
                with open(save_path, 'w') as f:
                    # json.dump(new_data, f)
                    json.dump(new_data, f, indent=2)  # Easier to check files manually


def get_save_path_seeds(path, cls, shots, seed):
    s = path.split('/')
    prefix = 'full_box_{}shot_{}_trainval'.format(shots, cls)
    save_dir = os.path.join('datasets', 'cocosplit', 'seed' + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    ID2CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
