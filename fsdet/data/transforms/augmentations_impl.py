from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.augmentation_impl import RandomRotation, RandomFlip, ResizeShortestEdge
from detectron2.utils.registry import Registry

AUGMENTATIONS_REGISTRY = Registry("AUGMENTATIONS")


def build_augmentation(name, cfg):
    return AUGMENTATIONS_REGISTRY.get(name)(cfg)


class AugmentationWrapper(Augmentation):
    """
    Wrap a given augmentation and redirect calls to this encapsulated augmentation.
    This wrapper is necessary because directly inheriting from certain augmentations will break
    the '__repr__' method when the signature of a class' '__init__' method and the 'super()__init__' method
    differ!
    This way we can inherit from an 'Augmentation' class to offer Augmentation's methods and still have an informative
    class representation string.
    """
    def __init__(self, augmentation: Augmentation):
        self.augmentation = augmentation

    def get_transform(self, img):
        return self.augmentation.get_transform(img)

    def __repr__(self):
        return self.augmentation.__repr__()


@AUGMENTATIONS_REGISTRY.register()
class ResizeShortestEdgeLimitLongestEdge(AugmentationWrapper):
    def __init__(self, cfg):
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        super().__init__(ResizeShortestEdge(short_edge_length=min_size, max_size=max_size, sample_style=sample_style))


@AUGMENTATIONS_REGISTRY.register()
class RandomHFlip(AugmentationWrapper):
    def __init__(self, cfg):
        super().__init__(RandomFlip(prob=0.5, horizontal=True, vertical=False))


@AUGMENTATIONS_REGISTRY.register()
class RandomVFlip(AugmentationWrapper):
    def __init__(self, cfg):
        super().__init__(RandomFlip(prob=0.5, horizontal=False, vertical=True))


@AUGMENTATIONS_REGISTRY.register()
class RandomFourAngleRotation(AugmentationWrapper):
    def __init__(self, cfg):
        super().__init__(RandomRotation(angle=[90.0, 180.0, 270.0, 360.0], expand=True, sample_style="choice"))
