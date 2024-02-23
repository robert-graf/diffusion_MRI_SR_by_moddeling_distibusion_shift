from enum import Enum
from typing import Union

import torch
import torch.nn.functional as F


def extract_slices_from_volume(x3d: torch.Tensor, centre_of_mass: Union[torch.Tensor, None] = None) -> torch.Tensor:
    """
    Extracts slices from 3d volume.
    In: B x (C x) D x H x W, B x 3
    Out: (B*C) x 1 x H x W
    """

    # x3d = (x3d + 1) / 2
    num_channels = x3d.shape[-4]
    x3d_batch = x3d.view(-1, *x3d.shape[-3:])

    if centre_of_mass is None:
        centre_of_mass = calc_center_of_mass(x3d_batch)
    else:
        centre_of_mass = centre_of_mass.view(-1, 3).repeat_interleave(num_channels, dim=0)
    centre_of_mass = centre_of_mass.long()
    n = centre_of_mass.shape[0]

    mri_seq_slice_coronal = torch.stack([x3d_batch[i, ..., (centre_of_mass[i, 0]), :, :] for i in range(n)], dim=0)
    mri_seq_slice_sagittal = torch.stack([x3d_batch[i, ..., (centre_of_mass[i, 1]), :] for i in range(n)], dim=0)
    mri_seq_slice_axial = torch.stack([x3d_batch[i, ..., (centre_of_mass[i, 2])] for i in range(n)], dim=0)

    mri_seq_slice = [mri_seq_slice_coronal, mri_seq_slice_sagittal, mri_seq_slice_axial]

    mri_seq_slice = torch.stack(mri_seq_slice, dim=1).flatten(start_dim=0, end_dim=1).unsqueeze(1)

    x3d = torch.flatten(mri_seq_slice, start_dim=0, end_dim=1).unsqueeze(1)

    return x3d


def calc_center_of_mass(x3d: torch.Tensor) -> torch.Tensor:
    """
    Returns centre of mass for each 3d Volume in batch.
    In: B x (C x) D x H x W
    Out: B x 3
    """
    x3d = x3d.float()

    n_x, n_y, n_z = x3d.shape[-3:]
    ii, jj, kk = torch.meshgrid(
        torch.arange(n_x),
        torch.arange(n_y),
        torch.arange(n_z),
        indexing="ij",
    )
    coords = torch.stack([ii.flatten(), jj.flatten(), kk.flatten()], dim=-1).float().to(x3d.device)
    vmin = torch.min(x3d)
    vmax = torch.max(x3d)

    if vmax.allclose(vmin) and not vmax.allclose(torch.zeros(1)):
        # everything is tumor
        x3d_norm = torch.ones_like(x3d)
    else:
        x3d_norm = (x3d - vmin) / (vmax - vmin)

    if x3d_norm.dim() == 5:
        x3d_norm = (x3d_norm > 0).any(dim=1).float()

    assert x3d_norm.dim() == 4, "MRI must be 4D: BxDxHxW"

    x3d_list = torch.flatten(x3d_norm, start_dim=-3).unsqueeze(-1)

    brainmask_approx = (x3d_list > 0.0).all(dim=0).squeeze()
    coords = coords[brainmask_approx]
    x3d_list = x3d_list[:, brainmask_approx]

    total_mass = torch.sum(x3d_list, dim=1)
    centre_of_mass = torch.sum(x3d_list * coords, dim=1) / total_mass

    if torch.any(torch.isnan(centre_of_mass)):
        # backup method
        print("Centre of mass contains NaN. Using backup method. every entry is zero, this should not happen.")
        isna_mask = torch.isnan(centre_of_mass).any(dim=1)
        n_isna = isna_mask.sum()
        mean_coord = torch.tensor([n_x / 2, n_y / 2, n_z / 2], device=x3d.device, dtype=x3d.dtype)
        centre_of_mass[isna_mask] = mean_coord.unsqueeze(0).repeat(n_isna, 1)

    return centre_of_mass


# enum for loading mri as a whole or only the tumor
class MriCrop(int, Enum):
    WHOLE = 1
    TUMOR = 2
