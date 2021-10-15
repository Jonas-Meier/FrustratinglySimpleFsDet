import numpy as np
from fvcore.transforms.transform import Transform
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.augmentation_impl import (
    RandomRotation, RandomFlip, ResizeShortestEdge, RandomBrightness, RandomContrast, RandomSaturation, RandomLighting
)
from detectron2.utils.registry import Registry

from albumentations.core.transforms_interface import BasicTransform, DualTransform
from albumentations.augmentations.transforms import (
    VerticalFlip,
    GaussNoise,
    ToGray,
    GaussianBlur,
)
from albumentations.augmentations.geometric.rotate import RandomRotate90, SafeRotate
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate

AUGMENTATIONS_REGISTRY = Registry("AUGMENTATIONS")


def build_augmentation(name, cfg):
    return AUGMENTATIONS_REGISTRY.get(name)(cfg)


class Detectron2AugmentationAdapter(Augmentation):
    """
    Wrap a given augmentation and redirect calls to this encapsulated augmentation.
    This wrapper is necessary because directly inheriting from certain augmentations will break
    the '__repr__' method when the signature of a class' '__init__' method and the 'super().__init__' method
    differ!
    This way, we can inherit from an 'Augmentation' class to offer Augmentation's methods and still have an informative
    class representation string.
    """
    def __init__(self, augmentation: Augmentation):
        self.augmentation = augmentation

    def get_transform(self, img):
        return self.augmentation.get_transform(img)

    def __repr__(self):
        return self.augmentation.__repr__()


class FvcoreTransformAdapter(Transform):
    def __init__(self, transform: BasicTransform, **kwargs):
        assert "image" in kwargs
        self.transform = transform
        self.params = self.transform.get_params()
        self._update_params(**kwargs)
        super().__init__()

    def _update_params(self, **kwargs):
        # Update the params with target-specific targets
        #   (see albumentations.core.transforms_interface:BasicTransform.__call__)
        # TODO: probably need to handle self.transform.{replay_mode|deterministic} as well!
        if self.transform.targets_as_params:
            targets_as_params = {k: kwargs[k] for k in self.transform.targets_as_params}
            params_dependent_on_targets = self.transform.get_params_dependent_on_targets(targets_as_params)
            self.params.update(params_dependent_on_targets)
        # params from albumentations.core.transforms_interface:BasicTransform.update_params
        # Note: We don't hard-code the params that come with the method 'update_params', since some transformations
        #  override this method!
        self.params.update(self.transform.update_params(self.params, **kwargs))

    def apply_image(self, img: np.ndarray):
        return self.transform.apply(img, **self.params)

    def apply_coords(self, coords: np.ndarray):
        raise NotImplementedError

    def inverse(self) -> Transform:
        # TODO: albumentations does not define inverse transformations!?
        # see: https://github.com/albumentations-team/albumentations/issues/322
        raise NotImplementedError

    def __repr__(self):
        return self.transform.__repr__()


class AlbumentationsBasicTransformAdapter(FvcoreTransformAdapter):
    def __init__(self, transform: BasicTransform, **kwargs):
        super().__init__(transform, **kwargs)

    def apply_coords(self, coords: np.ndarray):
        return coords


class AlbumentationsDualTransformAdapter(FvcoreTransformAdapter):
    def __init__(self, transform: DualTransform, **kwargs):
        super().__init__(transform, **kwargs)

    # Note: Detectron2 uses
    #  'apply_image' to transform images [1]
    #  'apply_segmentation' -> 'apply_image' to transform masks [1,2]
    #  'apply_box' -> 'apply_coords' to convert boxes [1,2]
    #  'apply_coords' to transform polygons [3,4,1]
    #  Sources:
    #    [1] 'detectron2/data/transforms/augmentation:StandardAugInput:transform'
    #    [2] 'fvcore/transforms/transform:Transform'
    #    [3] 'detectron2/data/dataset_mapper:DatasetMapper:__call__'
    #    [4] 'detectron2/data/detection_utils:transform_instance_annotations'
    # --> Since Albumentations defines its own methods for 'image', 'mask(s)', 'bboxes' and 'keypoints',
    #  we override the corresponding methods from fvcore.transforms.Transforms and disable the method 'apply_coords'

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return self.transform.apply_to_mask(segmentation, **self.params)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        # Note: the return type from albumentations seems to be a tuple, but detectron2 wants np.ndarray...
        return np.asarray(self.transform.apply_to_bboxes(box, **self.params))

    def apply_polygons(self, polygons: list) -> list:
        # TODO: unsure, if the return type from albumentations is correct...
        return self.transform.apply_to_keypoints(polygons, **self.params)

    def apply_coords(self, coords: np.ndarray):
        raise NotImplementedError


class AlbumentationsTransformAdapter(Augmentation):
    def __init__(self, albumentations_transform: BasicTransform):
        self.transform = albumentations_transform

    def get_transform(self, img) -> Transform:
        # force kwargs to have the field 'image' since it is always used by
        #   albumentations.core.transforms_interface:BasicTransform.update_params to extract the image's shape.
        # Note: it is necessary to yet set the 'image' field, because Detectron2 applies image and bbox transform
        #  separately but the bbox transform needs the image size. As a result, the image size has to be present before
        #  any transformation is applied! The behaviour is similar to
        #  'detectron2/data/transforms/augmentation_impl', the only difference is, that we don't set the arguments for
        #  the image size manually but rather let the method 'BasicTransform.update_params' do that for us.
        kwargs = {"image": img}
        if isinstance(self.transform, DualTransform):
            return AlbumentationsDualTransformAdapter(self.transform, **kwargs)
        else:
            assert isinstance(self.transform, BasicTransform)
            return AlbumentationsBasicTransformAdapter(self.transform, **kwargs)

    def __repr__(self):
        return self.transform.__repr__()


# -------------------------------------------------------------------------------------------------------------------- #
# Define your own Augmentations, using Instances of:
#  detectron2.data.transforms.Augmentation (wrap into an 'Detectron2AugmentationAdapter'), or
#  albumentations.core.transforms_interface.{BasicTransform|ImageOnlyTransform|DualTransform}
#    (wrap into 'AlbumentationsTransformAdapter')
# -------------------------------------------------------------------------------------------------------------------- #

@AUGMENTATIONS_REGISTRY.register()
class AlbumentationsVFlip(AlbumentationsTransformAdapter):
    def __init__(self, cfg):
        super().__init__(VerticalFlip())


@AUGMENTATIONS_REGISTRY.register()
class AlbumentationsRandom90degRotate(AlbumentationsTransformAdapter):
    def __init__(self, cfg):
        super().__init__(SafeRotate())


@AUGMENTATIONS_REGISTRY.register()
class AlbumentationsGaussNoise(AlbumentationsTransformAdapter):
    def __init__(self, cfg):
        super().__init__(GaussNoise())


@AUGMENTATIONS_REGISTRY.register()
class AlbumentationsGaussBlur(AlbumentationsTransformAdapter):
    def __init__(self, cfg):
        super().__init__(GaussianBlur())


@AUGMENTATIONS_REGISTRY.register()
class AlbumentationsToGrey(AlbumentationsTransformAdapter):
    def __init__(self, cfg):
        super().__init__(ToGray())


@AUGMENTATIONS_REGISTRY.register()
class ResizeShortestEdgeLimitLongestEdge(Detectron2AugmentationAdapter):
    def __init__(self, cfg):
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        super().__init__(ResizeShortestEdge(short_edge_length=min_size, max_size=max_size, sample_style=sample_style))


@AUGMENTATIONS_REGISTRY.register()
class RandomHFlip(Detectron2AugmentationAdapter):
    def __init__(self, cfg):
        super().__init__(RandomFlip(prob=0.5, horizontal=True, vertical=False))


@AUGMENTATIONS_REGISTRY.register()
class RandomVFlip(Detectron2AugmentationAdapter):
    def __init__(self, cfg):
        super().__init__(RandomFlip(prob=0.5, horizontal=False, vertical=True))


@AUGMENTATIONS_REGISTRY.register()
class RandomFourAngleRotation(Detectron2AugmentationAdapter):
    def __init__(self, cfg):
        super().__init__(RandomRotation(angle=[90.0, 180.0, 270.0, 360.0], expand=True, sample_style="choice"))


@AUGMENTATIONS_REGISTRY.register()
class Random50PercentContrast(Detectron2AugmentationAdapter):
    def __init__(self, cfg):
        super().__init__(RandomContrast(intensity_min=0.5, intensity_max=1.5))


@AUGMENTATIONS_REGISTRY.register()
class Random50PercentBrightness(Detectron2AugmentationAdapter):
    def __init__(self, cfg):
        super().__init__(RandomBrightness(intensity_min=0.5, intensity_max=1.5))


@AUGMENTATIONS_REGISTRY.register()
class Random50PercentSaturation(Detectron2AugmentationAdapter):
    def __init__(self, cfg):
        super().__init__(RandomSaturation(intensity_min=0.5, intensity_max=1.5))


@AUGMENTATIONS_REGISTRY.register()
class RandomAlexNetLighting(Detectron2AugmentationAdapter):
    def __init__(self, cfg):
        super().__init__(RandomLighting(scale=0.1))
