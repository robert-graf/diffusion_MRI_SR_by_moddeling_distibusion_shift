import numbers
import random
from collections.abc import Sequence
from math import ceil, floor

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from degrade.degrade import fwhm_needed, fwhm_units_to_voxel_space
from resize.pytorch import resize
from torch import Tensor
from torchvision.transforms.functional import rotate as rotate2D

from dataloader.datasets.dataset_utils import apply_target_pad_torch, parse_kernel, target_pad

_Shape = list[int] | torch.Size | tuple[int, ...]

mask_keys = ["msk", "seg"]
ignore_keys = [*mask_keys, "linspace"]

non_mri_keys = [*ignore_keys, "ct"]


class ColorJitter3D:
    """
    Randomly change the brightness and contrast an image.
    A grayscale tensor with values between 0-1 and shape BxCxHxWxD is expected.
    Args:
        brightness (float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    """

    def __init__(self, brightness_min_max: tuple, contrast_min_max: tuple) -> None:
        self.brightness_min_max = brightness_min_max
        self.contrast_min_max = contrast_min_max
        self.update()

    def update(self):
        if self.brightness_min_max:
            self.brightness = float(torch.empty(1).uniform_(self.brightness_min_max[0], self.brightness_min_max[1]))
        if self.contrast_min_max:
            self.contrast = float(torch.empty(1).uniform_(self.contrast_min_max[0], self.contrast_min_max[1]))

    def __call__(self, x: torch.Tensor, no_update=False) -> torch.Tensor:
        if not no_update:
            self.update()
        if self.brightness_min_max:
            x = (self.brightness * x).float().clamp(0, 1.0).to(x.dtype)
        if self.contrast_min_max:
            mean = torch.mean(x.float(), dim=list(range(-x.dim(), 0)), keepdim=True)
            x = (self.contrast * x + (1.0 - self.contrast) * mean).float().clamp(0, 1.0).to(x.dtype)
        return x


def pad(x, mod: int):
    padding = []
    for dim in reversed(x.shape[1:]):
        padding.extend([0, (mod - dim % mod) % mod])
    x = F.pad(x, padding)
    return x


def pad_size(x: Tensor, target_shape, mode="constant"):
    while 1.0 * target_shape[-1] / x.shape[-1] > 2:
        x = pad_size(x, target_shape=[min(2 * a, b) for a, b in zip(x.shape[-len(target_shape) :], target_shape, strict=True)])
    while 1.0 * target_shape[-2] / x.shape[-2] > 2:
        x = pad_size(x, target_shape=[min(2 * a, b) for a, b in zip(x.shape[-len(target_shape) :], target_shape, strict=True)])
    padding = []
    for in_size, out_size in zip(reversed(x.shape[-2:]), reversed(target_shape), strict=True):
        to_pad_size = max(0, out_size - in_size) / 2.0
        padding.extend([ceil(to_pad_size), floor(to_pad_size)])
    x_ = (
        F.pad(x.unsqueeze(0).unsqueeze(0), padding, mode=mode).squeeze(0).squeeze(0)
    )  # mode - 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
    return x_


def random_crop(target_shape: tuple[int, int], *arrs: torch.Tensor):
    sli = [slice(None), slice(None)]
    for i in range(2):
        z = max(0, arrs[0].shape[-i] - target_shape[-i])
        if z != 0:
            r = random.randint(0, z)
            r2 = r + target_shape[-i]
            sli[-i] = slice(r, r2 if r2 != arrs[0].shape[-i] else None)

    return tuple(a[..., sli[0], sli[1]] for a in arrs)


def calc_random_crop3D(target_shape: tuple[int, ...], shape: _Shape) -> tuple[slice, slice, slice]:
    sli = [slice(None), slice(None), slice(None)]
    for i in range(3):
        z = max(0, shape[-i] - target_shape[-i])
        if z != 0:
            r = random.randint(0, z)
            r2 = r + target_shape[-i]
            sli[-i] = slice(r, r2 if r2 != shape[-i] else None)

    return tuple(sli)  # type: ignore


def apply_random_crop3D(sli: tuple[slice, slice, slice], *arrs: torch.Tensor | None) -> tuple[Tensor, ...]:
    return tuple(a[..., sli[0], sli[1], sli[2]] if a is not None else None for a in arrs)  # type: ignore


def add_linspace_embedding(
    out: dict[str, Tensor],
    shape_in: _Shape,
    dim: int,
    crop: tuple[slice, slice, slice] | None = None,
    pad=None,
    name_addendum="",
    device=None,
):
    shape = shape_in[-3:]
    if dim == 3:
        l0 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], shape[2], 1))
        l1 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], shape[2], 1))
        l2 = np.tile(np.linspace(0, 1, shape[2]), (shape[0], shape[1], 1))
        l0 = Tensor(l0).permute(2, 0, 1)
        l1 = Tensor(l1).permute(0, 2, 1)
        l2 = Tensor(l2)
        assert l2.shape == l1.shape, (l2.shape, l1.shape)
    elif dim == 2:
        l0 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], 1))
        l1 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1))
        l0 = Tensor(l0).permute(1, 0)
        l1 = Tensor(l1)
        l2 = None
    else:
        raise NotImplementedError(dim)
    assert l0.shape == shape, (l0.shape, shape)
    assert l0.shape == l1.shape, (l0.shape, l1.shape)
    assert shape == l1.shape, (shape, l1.shape)
    l0 = l0.reshape(shape_in)
    l1 = l1.reshape(shape_in)
    l2 = l2.reshape(shape_in) if l2 is not None else None
    if pad is not None:
        l0 = apply_target_pad_torch(l0, pad, mode="reflect")
        l1 = apply_target_pad_torch(l1, pad, mode="reflect")
        l2 = apply_target_pad_torch(l2, pad, mode="reflect") if l2 is not None else None

    if crop is not None:
        l0, l1, l2 = apply_random_crop3D(crop, l0, l1, l2)
    if device is not None:
        l0 = l0.to(device)
        l1 = l1.to(device)
        l2 = l2.to(device) if l2 is not None else None
    out[f"linspace{name_addendum}0"] = l0.clone()
    out[f"linspace{name_addendum}1"] = l1.clone()
    if l2 is not None:
        out[f"linspace{name_addendum}2"] = l2.clone()


# https://github.com/Linus4world/3D-MRI-style-transfer/blob/master/data/data_augmentation_3D.py
# Author: https://github.com/Linus4world


# class SpatialRotation:
#    def __init__(self, dimensions: Sequence, k: Sequence = [3], auto_update=True):
#        self.dimensions = dimensions
#        self.k = k
#        self.args = None
#        self.auto_update = auto_update
#        self.update()#

#    def update(self):
#        self.args = [random.choice(self.k) for dim in self.dimensions]#

#    def __call__(self, x: torch.Tensor) -> torch.Tensor:
#        if self.auto_update:
#            self.update()
#        for k, dim in zip(self.args, self.dimensions):
#            x = torch.rot90(x, k, dim)
#        return x#


class SpatialFlip:
    def __init__(self, dims: Sequence, shape: tuple[int, ...], auto_update=True, ignore_keys=non_mri_keys, prob=0.5) -> None:
        self.dims = dims
        self.switches = []
        for e1, i in enumerate(shape):
            for e2, j in enumerate(shape):
                if i == j and e1 != e2:
                    self.switches.append((e1, e2))
        self.args = {}
        self.auto_update = auto_update
        self.update()
        self.ignore_keys = ignore_keys
        self.prob = prob

    def update(self):
        self.args = random.sample(self.switches, 1)

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        if self.auto_update:
            self.update()
        for key, x in input_tensor2.items():
            if any(a in key for a in self.ignore_keys):
                continue
            x = torch.flip(x, *self.args)
            input_tensor2[key] = x
        return input_tensor2


class ColorJitter3D_:
    """
    Randomly change the brightness and contrast an image.
    A grayscale tensor with values between 0-1 and shape BxCxHxWxD is expected.
    Args:
        brightness (float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    """

    def __init__(
        self, brightness_min_max: tuple = (0.8, 1.2), contrast_min_max: tuple = (0.8, 1.2), ignore_keys=non_mri_keys, prob=1.0
    ) -> None:
        self.brightness_min_max = brightness_min_max
        self.contrast_min_max = contrast_min_max
        self.ignore_keys = ignore_keys
        self.prob = prob

        self.update()

    def update(self):
        if self.brightness_min_max:
            self.brightness = float(torch.empty(1).uniform_(self.brightness_min_max[0], self.brightness_min_max[1]))
        if self.contrast_min_max:
            self.contrast = float(torch.empty(1).uniform_(self.contrast_min_max[0], self.contrast_min_max[1]))

    def __call__(self, input_tensor2: dict[str, torch.Tensor], no_update=False):
        if random.random() > self.prob:
            return input_tensor2

        for key, x in input_tensor2.items():
            if any(a in key for a in self.ignore_keys):
                continue
            if not no_update:
                self.update()
            if self.brightness_min_max:
                x = (self.brightness * x).float().clamp(0, 1.0).to(x.dtype)
            if self.contrast_min_max:
                mean = torch.mean(x.float(), dim=list(range(-x.dim(), 0)), keepdim=True)
                x = (self.contrast * x + (1.0 - self.contrast) * mean).float().clamp(0, 1.0).to(x.dtype)
            input_tensor2[key] = x
        return input_tensor2


class Linspace:
    """
    Randomly change the brightness and contrast an image.
    A grayscale tensor with values between 0-1 and shape BxCxHxWxD is expected.
    Args:
        brightness (float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    """

    def __init__(self, dims=3, linspace_name_addendum="") -> None:
        self.dims = dims
        self.linspace_name_addendum = linspace_name_addendum

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        example = next(iter(input_tensor2.values()))
        add_linspace_embedding(input_tensor2, example.shape, dim=self.dims, name_addendum=self.linspace_name_addendum)
        return input_tensor2


class Pad:
    def __init__(self, size: list[int]):
        self.mod = size
        # else:
        #    self.mod = 2**n_downsampling

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        for key, x in input_tensor2.items():
            padding = []
            for dim, mod in zip(reversed(x.shape[-len(self.mod) :]), reversed(self.mod), strict=True):
                a = max((mod - dim), 0) / 2
                padding.extend([floor(a), ceil(a)])
            input_tensor2[key] = F.pad(x, padding, mode="reflect")
        return input_tensor2

    def pad(self, x, n_downsampling: int = 1):
        mod = 2**n_downsampling
        padding = []
        for dim in reversed(x.shape[1:]):
            padding.extend([0, (mod - dim % mod) % mod])
        x = F.pad(x, padding)
        return x


class Crop3D:
    def __init__(self, size):
        self.size = size

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        example = next(iter(input_tensor2.values()))
        crop = calc_random_crop3D(self.size, example.shape[-3:])

        for key, x in input_tensor2.items():
            input_tensor2[key] = apply_random_crop3D(crop, x)[0]
        return input_tensor2

    def pad(self, x, n_downsampling: int = 1):
        mod = 2**n_downsampling
        padding = []
        for dim in reversed(x.shape[1:]):
            padding.extend([0, (mod - dim % mod) % mod])
        x = F.pad(x, padding)
        return x


class ColorJitterSphere3D:
    """
    Randomly change the brightness and contrast an image.
    A grayscale tensor with values between 0-1 and shape BxCxHxWxD is expected.
    Args:
        brightness (float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    """

    def __init__(self, brightness_min_max: tuple, contrast_min_max: tuple, sigma: float = 1.0, dims: int = 3) -> None:
        self.brightness_min_max = brightness_min_max
        self.contrast_min_max = contrast_min_max
        self.sigma = sigma
        self.dims = dims
        self.update()

    def update(self):
        if self.brightness_min_max:
            self.brightness = float(torch.empty(1).uniform_(self.brightness_min_max[0], self.brightness_min_max[1]))
        if self.contrast_min_max:
            self.contrast = float(torch.empty(1).uniform_(self.contrast_min_max[0], self.contrast_min_max[1]))
        self.ranges = []
        for _ in range(self.dims):
            r = torch.rand(2) * 10 - 5
            self.ranges.append((r.min().item(), r.max().item()))

    def __call__(self, x: torch.Tensor, no_update=False) -> torch.Tensor:
        if not no_update:
            self.update()

        jitterSphere = torch.zeros(1)
        for i, r in enumerate(self.ranges):
            jitterSphere_i = torch.linspace(*r, steps=x.shape[i + 1])
            jitterSphere_i = (1 / (self.sigma * 2.51)) * 2.71 ** (
                -0.5 * (jitterSphere_i / self.sigma) ** 2
            )  # Random section of a normal distribution between (-5,5)
            jitterSphere = jitterSphere.unsqueeze(-1) + jitterSphere_i.view(1, *[1] * i, -1)
        jitterSphere /= torch.max(jitterSphere)  # Random 3D section of a normal distribution sphere

        if self.brightness_min_max:
            brightness = (self.brightness - 1) * jitterSphere + 1
            x = (brightness * x).float().clamp(0, 1.0).to(x.dtype)
        if self.contrast_min_max:
            contrast = (self.contrast - 1) * jitterSphere + 1
            mean = x.float().mean()
            x = (contrast * x + (1.0 - self.contrast) * mean).float().clamp(0, 1.0).to(x.dtype)
        return x


class RandomRotate:
    def __init__(self, angle=15, seg_list=mask_keys, prob=0.3) -> None:
        self.angle = angle
        self.seg_list = seg_list
        self.prob = prob

    def rotation_matrix(self, axis, theta, device_="cpu"):
        """
        Generalized 3d rotation via Euler-Rodriguez formula, https://www.wikiwand.com/en/Euler%E2%80%93Rodrigues_formula
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = axis / torch.sqrt(torch.dot(axis, axis))
        a = torch.cos(theta / 2.0)
        b, c, d = -axis * torch.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return torch.tensor(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ],
            device=device_,
        )

    def get_3d_locations(self, d, h, w, device_):
        locations_x = torch.linspace(0, w - 1, w, device=device_).view(1, 1, 1, w).expand(1, d, h, w)
        locations_y = torch.linspace(0, h - 1, h, device=device_).view(1, 1, h, 1).expand(1, d, h, w)
        locations_z = torch.linspace(0, d - 1, d, device=device_).view(1, d, 1, 1).expand(1, d, h, w)
        # stack locations
        locations_3d = torch.stack([locations_x, locations_y, locations_z], dim=4).view(-1, 3, 1)
        return locations_3d

    def rotate(self, x: torch.Tensor, rotation_matrix: torch.Tensor, mode: str) -> torch.Tensor:
        device_ = x.device
        d, h, w = x.shape[-3:]
        input_tensor = x.unsqueeze(0)
        # get x,y,z indices of target 3d data
        locations_3d = self.get_3d_locations(d, h, w, device_)
        # rotate target positions to the source coordinate
        rotated_3d_positions = torch.bmm(rotation_matrix.view(1, 3, 3).expand(d * h * w, 3, 3), locations_3d).view(1, d, h, w, 3)
        rot_locs = torch.split(rotated_3d_positions, split_size_or_sections=1, dim=4)

        # change the range of x,y,z locations to [-1,1]
        def norm(x: torch.Tensor) -> torch.Tensor:
            x -= x.min()
            x -= x.max() / 2
            return x

        normalized_locs_x = (2.0 * rot_locs[0] - (w - 1)) / (w - 1)
        normalized_locs_y = (2.0 * rot_locs[1] - (h - 1)) / (h - 1)
        normalized_locs_z = (2.0 * rot_locs[2] - (d - 1)) / (d - 1)
        # Recenter grid into FOV
        normalized_locs_x = norm(normalized_locs_x)
        normalized_locs_y = norm(normalized_locs_y)
        normalized_locs_z = norm(normalized_locs_z)
        grid = (
            torch.stack([normalized_locs_x, normalized_locs_y, normalized_locs_z], dim=4).view(1, d, h, w, 3).to(dtype=input_tensor.dtype)
        )
        # here we use the destination voxel-positions and sample the input 3d data trilinear
        rotated_signal = F.grid_sample(input=input_tensor, grid=grid, align_corners=True, mode=mode)
        return rotated_signal.squeeze(0)

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        if random.random() > self.prob:
            return input_tensor2
        example = next(iter(input_tensor2.values()))
        dim = example[0].dim()
        if dim == 2:
            a = torch.FloatTensor(1).uniform_(-self.angle, self.angle)
            for key, x in input_tensor2.items():
                if key in self.seg_list:
                    continue
                input_tensor2[key] = rotate2D(x.float(), a.item()).to(dtype=x.dtype)
        else:
            a = torch.FloatTensor(3).uniform_(-self.angle, self.angle).deg2rad()
            rot = torch.eye(3, device=example.device)
            for i in range(3):
                axis = torch.tensor([float(i == j) for j in range(3)])
                rot = rot.matmul(self.rotation_matrix(axis, a[i], device_=example.device))  # type: ignore
            for key, x in input_tensor2.items():
                mode = "nearest" if key in self.seg_list else "bilinear"
                input_tensor2[key] = self.rotate(x, rot, mode)
        return input_tensor2


class toTorch:
    def __init__(self, no_norm=["seg"], jpg=False) -> None:
        self.no_norm = no_norm
        self.jpg = jpg

    def __call__(self, input_tensor2: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        for key, value in input_tensor2.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value.astype(np.float32))
            if key not in self.no_norm:
                if self.jpg:
                    value /= 255
                else:
                    value -= value.min()
                    value /= value.max()
            if value.shape[0] != 1:
                value = value.unsqueeze(0)
            input_tensor2[key] = value

        return input_tensor2


class RandomScale:
    def __init__(self, min_scale=1.0, max_scale=1.0, prob=0.3, seg_list=mask_keys) -> None:
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob
        self.seg_list = seg_list

    def __call__(self, input_tensor2: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if random.random() > self.prob:
            return input_tensor2
        example = next(iter(input_tensor2.values()))
        device_ = example.device
        dims = example.shape[1:]
        dim = len(dims)
        s = torch.FloatTensor(1).uniform_(self.min_scale, self.max_scale).item()
        locations = []
        for i, d in enumerate(dims):
            locations.append(torch.linspace(-1, 1, d, device=device_).view(*[1] * (i + 1), d, *[1] * (dim - i - 1)).expand(1, *dims))
        grid = torch.stack(list(reversed(locations)), dim=dim + 1).view(1, *dims, dim)
        grid *= s
        for key, value in input_tensor2.items():
            mode = "nearest" if key in self.seg_list else "bilinear"

            x_scaled = F.grid_sample(input=value.unsqueeze(0).float(), grid=grid, align_corners=True, mode=mode).squeeze(0)
            input_tensor2[key] = x_scaled.to(dtype=example.dtype)
        return input_tensor2


class RandomInterlaceMovementArtifact:
    def __init__(
        self,
        min_step_size=4,
        max_step_size: int = 10,
        min_strength=0.002,
        max_strength=0.02,
        prob=0.3,
        seg_list=mask_keys,
        ignore=ignore_keys,
    ) -> None:
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.max_strength = max_strength
        self.min_strength = min_strength
        self.prob = prob
        self.seg_list = seg_list
        self.ignore = ignore

    def __call__(self, input_tensor2: dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if random.random() > self.prob:
            return input_tensor2
        example = next(iter(input_tensor2.values()))
        device_ = example.device
        dims = example.shape[1:]
        dim = len(dims)
        s = int(torch.randint(self.min_step_size, self.max_step_size, (1,)).item())
        strength = torch.FloatTensor(1).uniform_(self.min_strength, self.max_strength).item()
        locations = []
        for i, d in enumerate(dims):
            locations.append(torch.linspace(-1, 1, d, device=device_).view(*[1] * (i + 1), d, *[1] * (dim - i - 1)).expand(1, *dims))
        grid = torch.stack(list(reversed(locations)), dim=dim + 1).view(1, *dims, dim)
        for j in range(s):
            grid[:, j :: 2 * s] += strength
        for key, value in input_tensor2.items():
            if key in self.ignore:
                continue
            mode = "nearest" if key in self.seg_list else "bilinear"

            x_scaled = F.grid_sample(input=value.unsqueeze(0).float(), grid=grid, align_corners=True, mode=mode).squeeze(0)
            input_tensor2[key] = x_scaled.to(dtype=example.dtype)
        return input_tensor2


class RandomBlur:
    r"""Blur an image using a random-sized Gaussian filter.

    Args:
        std: Tuple :math:`(a_1, b_1, a_2, b_2, a_3, b_3)` representing the
            ranges (in mm) of the standard deviations
            :math:`(\sigma_1, \sigma_2, \sigma_3)` of the Gaussian kernels used
            to blur the image along each axis, where
            :math:`\sigma_i \sim \mathcal{U}(a_i, b_i)`.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(a, b)`.
            If only one value :math:`x` is provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x)`.
            If three values :math:`(x_1, x_2, x_3)` are provided,
            then :math:`\sigma_i \sim \mathcal{U}(0, x_i)`.
    """

    def __init__(self, std: tuple[float, float] = (0.5, 1.15), prob=0.2, ignore_keys=ignore_keys, kernel_size=3):
        self.std_range = std
        self.prob = prob
        self.ignore_keys = ignore_keys
        self.kernel_size = kernel_size

    def createKernel(self, channels: int, sigma: float | list[float], kernel_size: int | list[int] = 3, dim=3):
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        sigma_ = [sigma] * dim if isinstance(sigma, numbers.Number) else sigma

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing="ij")
        for size, std, mgrid in zip(kernel_size, sigma_, meshgrids, strict=True):
            mean = (size - 1) / 2
            kernel *= 1.0 / (std * torch.tensor(2 * torch.pi).sqrt()) * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        return kernel

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        if random.random() > self.prob:
            return input_tensor2
        x = next(iter(input_tensor2.values()))
        std = torch.FloatTensor(1).uniform_(*self.std_range).item()
        dtype = x.dtype
        if x.device.type == "cpu":
            x = x.float()
        dim = x[0].dim()
        kernel = self.createKernel(x.shape[0], sigma=std, kernel_size=self.kernel_size, dim=dim).to(device=x.device, dtype=x.dtype)

        for key, v in input_tensor2.items():
            if any(a in key for a in self.ignore_keys):
                continue
            if dim == 2:
                v = F.conv2d(v.unsqueeze(0), weight=kernel, groups=v.shape[0]).squeeze(0)
            else:
                v = F.conv3d(v.unsqueeze(0), weight=kernel, groups=v.shape[0]).squeeze(0)
            v = F.pad(v, [self.kernel_size // 2, self.kernel_size // 2] * dim, mode="reflect")
            input_tensor2[key] = v.type(dtype)
        return input_tensor2


class lamdaTransform:
    def __init__(self, lamda, decisions_lamda=lambda: {}, ignore_keys=ignore_keys, prob=1.0):
        self.lamda = lamda
        self.decisions = decisions_lamda
        self.ignore_keys = ignore_keys
        self.prob = prob

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        if random.random() > self.prob:
            return input_tensor2
        args = self.decisions()
        for key, x in input_tensor2.items():
            if any(a in key for a in self.ignore_keys):
                continue
            input_tensor2[key] = self.lamda(x, **args)
        return input_tensor2


class RandomNoise:
    r"""Add Gaussian noise with random parameters.

    Add noise sampled from a normal distribution with random parameters.

    Args:
        mean: Mean :math:`\mu` of the Gaussian distribution
            from which the noise is sampled.
            If two values :math:`(a, b)` are provided,
            then :math:`\mu \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\mu \sim \mathcal{U}(-d, d)`.
        std: Standard deviation :math:`\sigma` of the Gaussian distribution
            from which the noise is sampled.
            If two values :math:`(a, b)` are provided,
            then :math:`\sigma \sim \mathcal{U}(a, b)`.
            If only one value :math:`d` is provided,
            :math:`\sigma \sim \mathcal{U}(0, d)`.
    """

    def __init__(self, mean: float = 0, std: tuple[float, float] = (0, 0.1), ignore_keys=ignore_keys, prob=0.2):
        self.mean = mean
        self.std_range = std
        self.ignore_keys = ignore_keys
        self.prob = prob

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        if random.random() > self.prob:
            return input_tensor2

        mean = self.mean
        std = torch.FloatTensor(1).uniform_(*self.std_range).item()
        for key, x in input_tensor2.items():
            if any(a in key for a in self.ignore_keys):
                continue
            noise = torch.randn(*x.shape, device=x.device) * std + mean
            input_tensor2[key] = (x + noise).clip(0, 1)
        return input_tensor2


class RandomBiasField:
    r"""Add random MRI bias field artifact.

    MRI magnetic field inhomogeneity creates intensity
    variations of very low frequency across the whole image.

    The bias field is modeled as a linear combination of
    polynomial basis functions, as in K. Van Leemput et al., 1999,
    *Automated model-based tissue classification of MR images of the brain*.

    It was implemented in NiftyNet by Carole Sudre and used in
    `Sudre et al., 2017, Longitudinal segmentation of age-related
    white matter hyperintensities
    <https://www.sciencedirect.com/science/article/pii/S1361841517300257?via%3Dihub>`_.

    Args:
        coefficients: Maximum magnitude :math:`n` of polynomial coefficients.
            If a tuple :math:`(a, b)` is specified, then
            :math:`n \sim \mathcal{U}(a, b)`.
        order: Order of the basis polynomial functions.
    """

    def __init__(self, coefficients: tuple[float, float] = (0.5, 0.5), order: int = 3, ignore_keys=ignore_keys, prob=0.3) -> None:
        self.coefficients = coefficients
        self.order = order
        self.ignore_keys = ignore_keys
        self.prob = prob

    def get_params(self, order: int, coefficients_range: tuple[float, float]) -> list[float]:
        # Sampling of the appropriate number of coefficients for the creation
        # of the bias field map
        random_coefficients = []
        for x_order in range(order + 1):
            for y_order in range(order + 1 - x_order):
                for _ in range(order + 1 - (x_order + y_order)):
                    number = torch.FloatTensor(1).uniform_(*coefficients_range)
                    random_coefficients.append(number.item())
        return random_coefficients

    @staticmethod
    def generate_bias_field(data: torch.Tensor, order: int, coefficients: list[float]) -> torch.Tensor:
        # Create the bias field map using a linear combination of polynomial
        # functions and the coefficients previously sampled
        shape = torch.tensor(data.shape[1:])  # first axis is channels
        half_shape = shape / 2

        ranges = [torch.arange(-n, n, device=data.device) + 0.5 for n in half_shape]  # type: ignore

        bias_field = torch.zeros(data.shape[1:], device=data.device)
        meshes = list(torch.meshgrid(*ranges, indexing="ij"))

        for i in range(len(meshes)):
            mesh_max = meshes[i].max()
            if mesh_max > 0:
                meshes[i] = meshes[i] / mesh_max
        if len(meshes) == 2:
            x_mesh, y_mesh = meshes
            i = 0
            for x_order in range(order + 1):
                for y_order in range(order + 1 - x_order):
                    coefficient = coefficients[i]
                    new_map = coefficient * x_mesh**x_order * y_mesh**y_order
                    bias_field += new_map
                    i += 1
        else:
            x_mesh, y_mesh, z_mesh = meshes
            i = 0
            for x_order in range(order + 1):
                for y_order in range(order + 1 - x_order):
                    for z_order in range(order + 1 - (x_order + y_order)):
                        coefficient = coefficients[i]
                        new_map = coefficient * x_mesh**x_order * y_mesh**y_order * z_mesh**z_order
                        bias_field += new_map
                        i += 1
        bias_field = 1.0 / torch.exp(bias_field)
        return bias_field

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        if random.random() > self.prob:
            return input_tensor2
        example = next(iter(input_tensor2.values()))
        dtype = example.dtype
        coefficients = self.get_params(self.order, self.coefficients)
        bias_field = self.generate_bias_field(example, self.order, coefficients)
        for key, x in input_tensor2.items():
            if any(a in key for a in self.ignore_keys):
                continue
            x = x * bias_field
            input_tensor2[key] = x.clip(0, 1).type(dtype)

        return input_tensor2


def lagrange3(x: torch.Tensor, a=(0.3, 0.7), b=(0.45, 0.65)):
    if isinstance(a, (tuple, list)):
        a = torch.FloatTensor(1).uniform_(*a).item()
    if isinstance(b, (tuple, list)):
        b = torch.FloatTensor(1).uniform_(*b).item()

    x = (x * (x - a)) / (1 - a) + (x * (x - 1)) / (a * a - a) * b
    # x = x*(a*a-a*x+b*(x-1))/((a-1)*a)
    return x


def lagrange3_parm(a=(0.3, 0.7), b=(0.45, 0.65)):
    a = torch.FloatTensor(1).uniform_(*a).item()
    b = torch.FloatTensor(1).uniform_(*b).item()
    return {"a": a, "b": b}


def exp_shift(x: torch.Tensor, a=(-1, 1.5)):
    a = torch.FloatTensor(1).uniform_(*a) if isinstance(a, (tuple, list)) else torch.FloatTensor(a)
    x = (torch.exp(a * x - 1) * x) / torch.exp(a - 1)
    return x


def exp_shift_parm(a=(-1, 1.5)):
    a = torch.FloatTensor(1).uniform_(*a)
    return {"a": a}


class RandomQuadraticHistogramTransform(lamdaTransform):
    def __init__(self, ignore_keys=ignore_keys, prob=1.0):
        super().__init__(lagrange3, lagrange3_parm, ignore_keys=ignore_keys, prob=prob)


class RandomExponentialHistogramTransform(lamdaTransform):
    def __init__(self, ignore_keys=ignore_keys, prob=1.0):
        super().__init__(exp_shift, exp_shift_parm, ignore_keys=ignore_keys, prob=prob)


class Fork_LR_Transform:
    """Add realistic Volume effects by adding the point-spread-function in fourier space.
    See SMORE: https://ieeexplore.ieee.org/document/9253710
    We upscale the image back to original unlike SMORE
    """

    def __init__(self, resolutions, patch_size, noise, key_map=None):
        if key_map is None:
            key_map = {"img": "img_lr"}
        self.key_map = key_map
        self.kernel = []
        self.noise = noise
        for hr, lr in resolutions:
            # Model the blurring-effect of a LR slice (lr x lr) compared to (hr,hr)
            # Axial images 0.5 mm x 0.5 mm x 5-6 mm
            # Sagittal (1.5 T) images 1.1 mm x 1.1 mm x 2-4 mm
            # Sagittal (3 T) images 0.95-0.8 mm x 0.95-0.8 mm x 2-4 mm
            blur_fwhm = fwhm_units_to_voxel_space(fwhm_needed(hr, lr), hr)
            slice_separation = float(lr / hr)
            self.kernel.append((parse_kernel(None, "rf-pulse-slr", blur_fwhm), slice_separation))
        self.patch_size = patch_size

    def __call__(self, input_tensor2: dict[str, torch.Tensor]):
        out = {}
        for key, x in input_tensor2.items():
            out[key] = x
            if key in self.key_map:
                x_new = self.down_res(x)
                # print(x.shape, x_new.shape)
                assert x_new.shape == x.shape, (x_new.shape, x.shape)
                out[self.key_map[key]] = x_new
        return out

    def resize(self, patch_lr, slice_separation, ext_patch_size):
        p_lr: torch.Tensor = resize(patch_lr, (slice_separation, 1), order=3)  # type: ignore
        if self.noise:
            p_lr += torch.rand_like(p_lr) * (random.randint(0, 15) / 300.0)
        p_lr: torch.Tensor = resize(p_lr, (1 / slice_separation, 1), order=3)  # type: ignore order=random.choice([0, 1, 3])
        patch_lr, pads = target_pad(p_lr, ext_patch_size, mode="reflect")  # type: ignore
        return patch_lr

    def down_res(self, patch_hr):
        # HQ image range [0,1]
        blur_kernel, slice_separation = self.kernel[random.randint(0, len(self.kernel) - 1)]
        if len(patch_hr.shape) == 2:
            patch_hr = patch_hr.unsqueeze(0)
        if len(patch_hr.shape) == 3:
            patch_hr = patch_hr.unsqueeze(1)
        if len(patch_hr.shape) == 4 and patch_hr.shape[2] != 1:
            patch_hr = patch_hr.swapaxes(0, 1)

        s = patch_hr.shape
        L = blur_kernel.shape[0]
        ext_patch_size = [p + ceil(L / 2) + 2 if p != 1 else p for p in patch_hr.shape]

        patch_hr, pads = target_pad(patch_hr, ext_patch_size, mode="reflect")
        patch_hr = torch.from_numpy(patch_hr)
        patch_lr = F.conv2d(patch_hr, blur_kernel, padding="same")
        patch_lr = self.resize(patch_lr, slice_separation, ext_patch_size)
        patch_lr = torch.from_numpy(patch_lr)
        crop = []
        for s1, s2 in zip(patch_lr.shape, s, strict=True):
            a = max((s1 - s2) / 2, 0)
            if a == 0 or s1 == 1:
                crop.append(slice(None))
            else:
                crop.append(slice(ceil(a), -floor(a)))
        patch_lr = patch_lr[crop]
        patch_lr = patch_lr.swapaxes(0, 1) if patch_lr.shape[0] != 1 else patch_lr.squeeze(0)
        return patch_lr


class Fork_LR_Transform_Naive(Fork_LR_Transform):
    def down_res(self, patch_hr):
        _, slice_separation = self.kernel[random.randint(0, len(self.kernel) - 1)]

        if len(patch_hr.shape) == 2:
            patch_hr = patch_hr.unsqueeze(0)
        if len(patch_hr.shape) == 3:
            patch_hr = patch_hr.unsqueeze(1)
        if len(patch_hr.shape) == 4 and patch_hr.shape[2] != 1:
            patch_hr = patch_hr.swapaxes(0, 1)

        s = patch_hr.shape
        ext_patch_size = [int(p + slice_separation) if p != 1 else p for p in patch_hr.shape]

        patch_hr, pads = target_pad(patch_hr, ext_patch_size, mode="reflect")
        patch_hr = torch.from_numpy(patch_hr)
        patch_lr = self.resize(patch_hr, slice_separation, ext_patch_size)
        patch_lr = torch.from_numpy(patch_lr)
        crop = []
        for s1, s2 in zip(patch_lr.shape, s, strict=True):
            a = max((s1 - s2) / 2, 0)
            if a == 0 or s1 == 1:
                crop.append(slice(None))
            else:
                crop.append(slice(ceil(a), -floor(a)))
        patch_lr = patch_lr[crop]
        patch_lr = patch_lr.swapaxes(0, 1) if patch_lr.shape[0] != 1 else patch_lr.squeeze(0)
        return patch_lr


class Fork_LR_Transform_Stop_Gap(Fork_LR_Transform):
    """Simulate random "Stop Gap". A slices have a measurement thickness, if this is smaller than the
    slice thickness than we call this a stop gap. This class simulates a random stop gab where we
    receive random signal strength form the layers in side the stopgap
    """

    def resize(self, patch_lr, slice_separation: float, ext_patch_size):
        stop_gap = int(random.random() * (slice_separation * 0.5) + slice_separation / 10) + 1
        step = 10_000_000
        arr = patch_lr[..., 0, :] * 0
        for i in range(patch_lr.shape[-2]):
            if step >= stop_gap:
                r = [random.random() * 0.9 + 0.1 for _ in range(int(stop_gap))]
                arr = patch_lr[..., i, :] * 0
                k = 0
                for j, rand in enumerate(r, i + int(slice_separation) // 2 - len(r) // 2):
                    if j >= patch_lr.shape[-2]:
                        r = r[:k]
                        break
                    k += 1
                    # print(j, patch_lr.shape)
                    arr += patch_lr[..., j, :] * rand
                if len(r) != 0:
                    arr /= max(sum(r), 0.001)
                # make arr
                step = 0
            patch_lr[..., i, :] = arr
            step += 1
        return super().resize(patch_lr, slice_separation, ext_patch_size)


def getBetterOrientation(nifti: nib.Nifti1Image, axisCode="IPL"):
    orig_ornt = nib.io_orientation(nifti.affine)
    targ_ornt = nib.orientations.axcodes2ornt(axisCode)
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    nifti = nifti.as_reoriented(transform)
    return nifti


def toGrayScale(x):
    x_min = np.amin(x)
    x_max = np.amax(x) - x_min
    x = (x - x_min) / x_max
    return x


def center(x, mean, std):
    return (x - mean) / std
