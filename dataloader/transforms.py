from enum import Enum, auto
from math import ceil, floor
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F
from torchvision import transforms

from .datasets import transforms3D

if TYPE_CHECKING:
    from utils.arguments import DataSet_Option
else:
    DataSet_Option = ()


class Transforms_Enum(Enum):
    random_crop = auto()
    resize = auto()
    RandomHorizontalFlip = auto()
    RandomVerticalFlip = auto()
    pad = auto()
    CenterCrop = auto()
    CenterCrop256 = auto()
    to_RGB = auto()


def get_transforms2D(opt: DataSet_Option, split):
    size = opt.img_size  # type: ignore
    if size is None:
        size: tuple[int, int] = (128, 128)
    return get_transforms(size, opt.transforms, split == "train")


def get_transforms(size: tuple[int, int], tf: list[Transforms_Enum] | None, train=False):
    if tf is None:
        tf = [Transforms_Enum.pad, Transforms_Enum.random_crop, Transforms_Enum.RandomHorizontalFlip]
    out: list = [transforms.ToTensor()]
    for t in tf:
        if isinstance(t, str):
            t = Transforms_Enum[t]
        if isinstance(t, int):
            t = Transforms_Enum(t)
        if t.value == Transforms_Enum.random_crop.value:
            out.append(transforms.RandomCrop(size)) if train else out.append(transforms.CenterCrop(size))
        elif t.value == Transforms_Enum.CenterCrop.value:
            out.append(transforms.CenterCrop(size))
        elif t.value == Transforms_Enum.CenterCrop256.value:
            out.append(transforms.CenterCrop(256))
        elif t.value == Transforms_Enum.resize.value:
            out.append(transforms.Resize(size))
        elif t.value == Transforms_Enum.RandomHorizontalFlip.value:
            out.append(transforms.RandomHorizontalFlip())
        elif t.value == Transforms_Enum.RandomVerticalFlip.value:
            out.append(transforms.RandomVerticalFlip())
        elif t.value == Transforms_Enum.pad.value:
            out.append(Pad(size))
        elif t.value == Transforms_Enum.to_RGB.value:
            out.append(to_RGB())
        else:
            raise NotImplementedError(t.name)
    out.append(transforms.Normalize(0.5, 0.5))
    return transforms.Compose(out)


class Pad:
    def __init__(self, size: tuple[int, int] | int) -> None:
        if isinstance(size, int):
            size = (size, size)
        if len(size) == 1:
            size = (size[0], size[0])
        assert len(size) == 2
        self.size = size

    def __call__(self, image):
        w, h = image.shape[-2], image.shape[-1]
        max_w, max_h = self.size
        hp = max((max_w - w) / 2, 0)
        vp = max((max_h - h) / 2, 0)
        padding = (int(floor(vp)), int(ceil(vp)), int(floor(hp)), int(ceil(hp)))
        # print(padding,w,h)
        x = F.pad(image, padding, value=0, mode="constant")
        # print(x.shape)
        return x


class to_RGB:
    def __call__(self, image: torch.Tensor):
        if len(image) == 2:
            image = image.unsqueeze(0)
        if image.shape[-3] == 3:
            return image
        return torch.cat([image, image, image], dim=-3)


class Paired_Transforms_Enum(Enum):
    default = auto()
    most = auto()
    superres = auto()
    ablation_base = 10


class Transforms_Enum_3D(Enum):
    RandomScale = auto()
    RandomRotate = auto()
    RandomQuadraticHistogramTransform = auto()
    RandomExponentialHistogramTransform = auto()
    RandomQuadraticHistogramTransform_only_lr = auto()  # 5
    RandomExponentialHistogramTransform_only_lr = auto()  # 6
    RandomBlur = auto()  # 7
    RandomNoise = auto()  # 8
    RandomNoiseMedium = auto()  # 9
    RandomNoiseStrong = auto()  # 10
    RandomBiasField = auto()  # 11
    ColorJitter3D = auto()  # 12
    RandomInterlaceMovementArtifact = auto()  # 13
    Fork_upscale_only = auto()  # 14
    Fork_smore_one_scale = auto()  # 15
    Fork_smore_many_scale = auto()  # 16
    Fork_smore_many_scale_stop_gap = auto()  # 17
    _RandomBlur = auto()
    _Noise = auto()

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Enum):
            return __value.value == self.value
        if isinstance(__value, str):
            return __value.lower() == self.name.lower()

        return False


def get_paired_transform_by_list(opt: DataSet_Option, train=False, linspace_name_addendum="", jpg=False):
    size = opt.shape
    tf = opt.transforms_3D
    if tf is None:
        return get_paired_transform(opt, size, Paired_Transforms_Enum.superres, train, linspace_name_addendum)
    return transforms.Compose(
        [a for a in _get_paired_transform_by_list(opt, size, tf, train, linspace_name_addendum, jpg) if a is not None]
    )


def _get_paired_transform_by_list(
    opt: DataSet_Option, size, tf: list[Transforms_Enum_3D], train=False, linspace_name_addendum="", jpg=False
):
    prob = opt.prob_transforms
    mask_keys = ["msk", "seg"]
    ignore = [*mask_keys, "linspace", "hq"]
    non_mri_keys = [*ignore, "ct"]

    out_list: list = [transforms3D.toTorch(mask_keys, jpg=jpg)]
    if train:
        if Transforms_Enum_3D.RandomScale in tf:
            out_list.append(transforms3D.RandomScale(min_scale=0.75, max_scale=1.5, prob=prob))
        if Transforms_Enum_3D.RandomRotate in tf:
            out_list.append(transforms3D.RandomRotate(prob=prob, angle=5))

    if (hasattr(opt, "super_res") and opt.super_res) or (hasattr(opt, "image_name") and opt.image_name == "img_lr"):  # type: ignore
        k = {"hq": "img_lr", "img": "img_lr"}
        source = [i / 4 for i in range(4 * 4, 9 * 4)] if opt.sr_source is None else opt.sr_source
        if Transforms_Enum_3D.Fork_upscale_only in tf:
            out_list.append(transforms3D.Fork_LR_Transform_Naive([(opt.sr_target, 5)], size, noise=True, key_map=k))
        elif Transforms_Enum_3D.Fork_smore_one_scale in tf:
            out_list.append(transforms3D.Fork_LR_Transform([(opt.sr_target, 5)], size, noise=True, key_map=k))
        elif Transforms_Enum_3D.Fork_smore_many_scale in tf:
            out_list.append(transforms3D.Fork_LR_Transform([(opt.sr_target, i) for i in source], size, noise=True, key_map=k))
        elif Transforms_Enum_3D.Fork_smore_many_scale_stop_gap in tf:
            out_list.append(transforms3D.Fork_LR_Transform_Stop_Gap([(opt.sr_target, i) for i in source], size, noise=True, key_map=k))
        else:
            out_list.append(transforms3D.Fork_LR_Transform([(opt.sr_target, i) for i in source], size, noise=True, key_map=k))
    if Transforms_Enum_3D.RandomInterlaceMovementArtifact in tf:
        out_list.append(
            transforms3D.RandomInterlaceMovementArtifact(prob=prob / opt.RandomInterlaceMovementArtifactFactor, ignore=ignore)
            if train
            else None
        )

    if opt.linspace:
        out_list.append(transforms3D.Linspace(linspace_name_addendum=linspace_name_addendum))

    out_list.append(transforms3D.Pad(size=size))
    out_list.append(transforms3D.Crop3D(size))
    ###
    if Transforms_Enum_3D.RandomQuadraticHistogramTransform in tf:
        out_list.append(transforms3D.RandomQuadraticHistogramTransform(prob=prob, ignore_keys=[*mask_keys, "linspace"]))
    if Transforms_Enum_3D.RandomExponentialHistogramTransform in tf:
        out_list.append(transforms3D.RandomExponentialHistogramTransform(prob=prob, ignore_keys=[*mask_keys, "linspace"]))
    if Transforms_Enum_3D.RandomQuadraticHistogramTransform_only_lr in tf:
        out_list.append(transforms3D.RandomQuadraticHistogramTransform(prob=prob, ignore_keys=ignore))
    if Transforms_Enum_3D.RandomExponentialHistogramTransform_only_lr in tf:
        out_list.append(transforms3D.RandomExponentialHistogramTransform(prob=prob, ignore_keys=ignore))
    #####
    if Transforms_Enum_3D._RandomBlur in tf:
        out_list.append(transforms3D.RandomNoise(prob=0.5, std=(0.0, 0.1), ignore_keys=ignore) if train else None)
    if Transforms_Enum_3D.RandomBlur in tf:
        out_list.append(transforms3D.RandomBlur(prob=prob, std=(0.5, 4), ignore_keys=ignore, kernel_size=9) if train else None)
    if Transforms_Enum_3D.RandomNoise in tf:
        out_list.append(transforms3D.RandomNoise(prob=prob, std=(0.0, 0.05), ignore_keys=ignore) if train else None)
    if Transforms_Enum_3D.RandomNoiseMedium in tf:
        out_list.append(transforms3D.RandomNoise(prob=prob, std=(0.0, 0.075), ignore_keys=ignore) if train else None)
    if Transforms_Enum_3D.RandomNoiseStrong in tf:
        out_list.append(transforms3D.RandomNoise(prob=prob, std=(0.0, 0.1), ignore_keys=ignore) if train else None)
    if Transforms_Enum_3D._Noise in tf:
        out_list.append(transforms3D.RandomNoise(prob=0.7, std=(0.0, 0.1), ignore_keys=ignore) if train else None)
    if Transforms_Enum_3D.RandomBiasField in tf:
        out_list.append(transforms3D.RandomBiasField(prob=prob, coefficients=(0.0, 0.5), ignore_keys=ignore) if train else None)
    if Transforms_Enum_3D.ColorJitter3D in tf:
        out_list.append(transforms3D.ColorJitter3D_(ignore_keys=non_mri_keys))
    out_list.append(transforms3D.lamdaTransform(lambda x: x * 2 - 1, ignore_keys=[]))
    return out_list


def get_paired_transform(opt, size, tf: Paired_Transforms_Enum | None, train=False, linspace_name_addendum=""):
    return transforms.Compose([a for a in _get_paired_transform(opt, size, tf, train, linspace_name_addendum) if a is not None])


def _get_paired_transform(opt, size, tf: Paired_Transforms_Enum | None, train=False, linspace_name_addendum=""):
    mask_keys = ["msk", "seg"]
    ignore = [*mask_keys, "linspace", "hq"]
    non_mri_keys = [*ignore, "ct"]
    if tf is None or Paired_Transforms_Enum.default.value == tf.value:
        return [
            transforms3D.toTorch(mask_keys),
            transforms3D.Linspace(linspace_name_addendum=linspace_name_addendum) if opt.linspace else None,
            transforms3D.Pad(size=size),
            transforms3D.Crop3D(size),
            transforms3D.ColorJitter3D_(),
            transforms3D.lamdaTransform(lambda x: x * 2 - 1),
        ]
    if Paired_Transforms_Enum.superres.value == tf.value:
        return [
            transforms3D.toTorch(mask_keys),
            # transforms3D.RandomScale(min_scale=0.5,max_scale=1.5,prob=1) if train else None,
            transforms3D.RandomRotate(prob=0.0, angle=15) if train else None,
            transforms3D.Fork_LR_Transform([(0.875, i) for i in range(5, 9)], size, noise=True, key_map={"hq": "img_lr", "img": "img_lr"}),
            transforms3D.RandomInterlaceMovementArtifact(prob=0.3, ignore=ignore),
            transforms3D.Linspace(linspace_name_addendum=linspace_name_addendum) if opt.linspace else None,
            transforms3D.Pad(size=size),
            transforms3D.Crop3D(size),
            # transforms3D.RandomQuadraticHistogramTransform(prob=0.3, ignore_keys=ignore),
            # transforms3D.RandomBlur(prob=0.5, std=(0.5, 4), ignore_keys=ignore, kernel_size=9) if train else None,
            # transforms3D.RandomNoise(prob=0.7, std=(0.0, 0.1), ignore_keys=ignore) if train else None,
            # transforms3D.RandomBiasField(prob=0.3, coefficients=(0.0, 0.5), ignore_keys=ignore) if train else None,
            transforms3D.ColorJitter3D_(ignore_keys=non_mri_keys),
            transforms3D.lamdaTransform(lambda x: x * 2 - 1, ignore_keys=[]),
        ]
    if Paired_Transforms_Enum.most.value == tf.value:
        return [
            transforms3D.toTorch(mask_keys),
            # transforms3D.RandomScale(prob=1) if train else None,
            transforms3D.RandomRotate(prob=0.4, angle=15) if train else None,
            transforms3D.Fork_LR_Transform([(0.875, 5)], size, noise=True, key_map={"hq": "xxx"}),
            transforms3D.Linspace(linspace_name_addendum=linspace_name_addendum) if opt.linspace else None,
            transforms3D.Pad(size=size),
            transforms3D.Crop3D(size),
            transforms3D.RandomBlur(prob=0.3, ignore_keys=ignore) if train else None,
            transforms3D.RandomNoise(prob=0.8, std=(0.0, 0.1), ignore_keys=ignore) if train else None,
            transforms3D.RandomBiasField(prob=0.5, ignore_keys=ignore) if train else None,
            transforms3D.ColorJitter3D_(ignore_keys=non_mri_keys),
            transforms3D.lamdaTransform(lambda x: x * 2 - 1, ignore_keys=[]),
        ]
    raise NotImplementedError()
    return None


# ablation_lvl
def ablation(lvl: int = 0):
    base = [
        Transforms_Enum_3D.ColorJitter3D,
    ]
    lvl = lvl % 100
    if lvl == 1:  # up/down-scaling only
        base.append(Transforms_Enum_3D.Fork_upscale_only)
    elif lvl == 2:
        base.append(Transforms_Enum_3D.Fork_smore_one_scale)
    elif lvl == 3:
        base.append(Transforms_Enum_3D.Fork_smore_many_scale)
    elif lvl >= 4:
        base.append(Transforms_Enum_3D.Fork_smore_many_scale_stop_gap)
    else:
        raise NotImplementedError()
    if lvl >= 5:
        base.append(Transforms_Enum_3D.RandomInterlaceMovementArtifact)
    if lvl >= 6:
        base.append(Transforms_Enum_3D.RandomScale)
    if lvl >= 7:
        base.append(Transforms_Enum_3D.RandomRotate)
    if lvl == 8:
        base.append(Transforms_Enum_3D.RandomQuadraticHistogramTransform)
    if lvl == 9:
        base.append(Transforms_Enum_3D.RandomExponentialHistogramTransform)
    if lvl >= 10:
        base.append(Transforms_Enum_3D.RandomBlur)
        base.append(Transforms_Enum_3D.RandomBiasField)
    if lvl >= 11:
        base.append(Transforms_Enum_3D.RandomNoise)
    if lvl >= 13:
        raise NotImplementedError()

    return base
