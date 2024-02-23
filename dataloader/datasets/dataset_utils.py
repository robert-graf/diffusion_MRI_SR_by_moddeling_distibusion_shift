from math import ceil

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor


def get_patch(img_rot, patch_center, patch_size, return_idx=False):
    """
    img_rot: np.array, the HR in-plane image at a single rotation
    patch_center: tuple of ints, center position of the patch
    patch_size: tuple of ints, the patch size in 3D. For 2D patches, supply (X, Y, 1).
    """

    # Get random rotation and center
    sts = [c - p // 2 if p != 1 else c for c, p in zip(patch_center, patch_size)]
    ens = [st + p for st, p in zip(sts, patch_size)]
    idx = tuple(slice(st, en) for st, en in zip(sts, ens))

    if return_idx:
        return idx

    return img_rot[idx].squeeze()


from tqdm import tqdm


def get_random_centers(imgs_rot, patch_size, n_patches, weighted=True):
    rot_choices = np.random.randint(0, len(imgs_rot), size=n_patches)
    centers = []

    for i, img_rot in tqdm(enumerate(imgs_rot), total=len(imgs_rot), desc="random_centers"):
        n_choices = int(np.sum(rot_choices == i))

        if weighted:
            smooth = gaussian_filter(img_rot, 1.0)
            grads = np.gradient(smooth)
            grad_mag = np.sum([np.sqrt(np.abs(grad)) for grad in grads], axis=0)

            # Set probability to zero at edges
            for p, axis in zip(patch_size, range(grad_mag.ndim)):
                if p > 1:
                    grad_mag = np.swapaxes(grad_mag, 0, axis)
                    grad_mag[: p // 2 + 1] = 0.0
                    grad_mag[-p // 2 - 1 :] = 0.0
                    grad_mag = np.swapaxes(grad_mag, axis, 0)

            # Normalize gradient magnitude to create probabilities
            grad_probs = grad_mag / grad_mag.sum()
            grad_probs = [
                grad_probs.sum(axis=tuple(k for k in range(grad_probs.ndim) if k != axis)) for axis in range(len(grad_probs.shape))
            ]
            # Re-normalize per axis to ensure probabilities sum to 1
            for axis in range(len(grad_probs)):
                grad_probs[axis] = grad_probs[axis] / grad_probs[axis].sum()

        else:
            grad_probs = [None for _ in img_rot.shape]

        # Generate random patch centers for each dimension
        random_indices = [
            np.random.choice(
                np.arange(0, img_dim),
                size=n_choices,
                p=grad_probs[axis],
            )
            for axis, img_dim in enumerate(img_rot.shape)
        ]
        # Combine random indices to form multi-dimensional patch centers
        centers.extend((i, tuple(coord)) for coord in zip(*random_indices))
    np.random.shuffle(centers)
    return centers


def get_pads(target_dim, d):
    if target_dim <= d:
        return 0, 0
    p = (target_dim - d) // 2
    if (p * 2 + d) % 2 != 0:
        return p, p + 1
    return p, p


def target_pad(img: Tensor | np.ndarray, target_dims, mode="reflect") -> tuple[np.ndarray, tuple]:
    pads = tuple(get_pads(t, d) for t, d in zip(target_dims, img.shape, strict=True))
    return np.pad(img, pads, mode=mode), pads  # type: ignore


def calc_target_pad(shape, target_dims) -> tuple[np.ndarray, tuple]:
    pads = tuple(get_pads(t, d) for t, d in zip(target_dims, shape))
    return pads  # type: ignore


def apply_target_pad(img, pads, mode="reflect") -> np.ndarray:
    return np.pad(img, pads, mode=mode)  # type: ignore


def apply_target_pad_torch(img: Tensor, pads, mode="reflect") -> Tensor:
    return Tensor(np.pad(img.numpy(), pads, mode=mode), device=img.device)  # type: ignore


def parse_kernel(blur_kernel_file, blur_kernel_type, blur_fwhm):
    from degrade.degrade import select_kernel

    if blur_kernel_file is not None:
        blur_kernel: np.ndarray = np.load(blur_kernel_file)
    else:
        window_size = int(2 * round(blur_fwhm) + 1)
        blur_kernel = select_kernel(window_size, blur_kernel_type, fwhm=blur_fwhm)  # type: ignore
    blur_kernel /= blur_kernel.sum()
    blur_kernel = blur_kernel.squeeze()[None, None, :, None]
    blur_kernel_t = torch.from_numpy(blur_kernel).float()

    return blur_kernel_t


def calc_extended_patch_size(blur_kernel, patch_size):
    """
    Calculate the extended patch size. This is necessary to remove all boundary
    effects which could occur when we apply the blur kernel. We will pull a patch
    which is the specified patch size plus half the size of the blur kernel. Then we later
    blur at test time, crop off this extended patch size, then downsample.
    """

    L = blur_kernel.shape[0]

    ext_patch_size = [p + 2 * ceil(L / 2) if p != 1 else p for p in patch_size]
    ext_patch_crop = [(e - p) // 2 for e, p in zip(ext_patch_size, patch_size)]
    ext_patch_crop = tuple([slice(d, -d) for d in ext_patch_crop if d != 0])

    return ext_patch_size, ext_patch_crop
