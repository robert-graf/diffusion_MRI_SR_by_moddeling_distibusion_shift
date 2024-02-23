import gzip
import math
import time
from math import ceil, floor
from pathlib import Path

#!pip install numpy
import numpy as np
import torch
from TPTBox import (
    BIDS_FILE,
    NII,
    POI,
    BIDS_Global_info,
    Image_Reference,
    Location,
    Log_Type,
    No_Logger,
    calc_centroids_from_subreg_vert,
    to_nii,
)
from TPTBox.spine.snapshot2D.snapshot_templates import Snapshot_Frame, create_snapshot, mri_snapshot
from tqdm import tqdm

from dataloader.datasets.dataset_superres import target_pad
from pl_models.DEA import DAE_LitModel

model: DAE_LitModel = None  # type: ignore
model_ax: DAE_LitModel = None  # type: ignore
default_checkpoint = "lightning_logs_dae/my_dataset/version_0/checkpoints/last.ckpt"


def compute_crop_slice(
    array,
    minimum=0.0,
    dist=0,
    minimum_size: tuple[slice, ...] | int | tuple[int, ...] | None = None,
) -> tuple[slice, slice, slice]:
    shp = array.shape
    d = np.around(dist / np.array([1, 1])).astype(int)
    msk_bin = np.zeros(array.shape, dtype=bool)
    msk_bin[array > minimum] = 1
    msk_bin[np.isnan(msk_bin)] = 0
    cor_msk = np.where(msk_bin > 0)
    if cor_msk[0].shape[0] == 0:
        raise ValueError("Array would be reduced to zero size")
    c_min = [cor_msk[-2].min(), cor_msk[-1].min()]
    c_max = [cor_msk[-2].max(), cor_msk[-1].max()]
    x0 = c_min[0] - d[0] if (c_min[0] - d[0]) > 0 else 0
    y0 = c_min[1] - d[1] if (c_min[1] - d[1]) > 0 else 0
    x1 = c_max[0] + d[0] if (c_max[0] + d[0]) < shp[-2] else shp[-2]
    y1 = c_max[1] + d[1] if (c_max[1] + d[1]) < shp[-1] else shp[-1]
    ex_slice = [slice(None), slice(x0, x1 + 1), slice(y0, y1 + 1)]
    if minimum_size is not None:
        if isinstance(minimum_size, int):
            minimum_size = (minimum_size, minimum_size)
        for i, min_w in enumerate(minimum_size):
            if isinstance(min_w, slice):
                min_w = min_w.stop - min_w.start  # noqa: PLW2901
            curr_w = ex_slice[i].stop - ex_slice[i].start
            dif = min_w - curr_w
            if min_w > 0:
                new_start = ex_slice[i].start - math.floor(dif / 2)
                new_goal = ex_slice[i].stop + math.ceil(dif / 2)
                if new_goal > shp[i]:
                    new_start -= new_goal - shp[i]
                    new_goal = shp[i]
                if new_start < 0:  #
                    new_goal -= new_start
                    new_start = 0
                ex_slice[i] = slice(new_start, new_goal)

    return tuple(ex_slice)  # type: ignore


def upscale_nii(
    nii_org: NII,
    out_path: Path,
    batch_size=32,
    checkpoint_sag=default_checkpoint,
    checkpoint_ax=None,
    device=torch.device("cuda:0"),  # noqa: B008
    override_upscale=False,
):
    if not override_upscale and out_path.exists():
        print(f"{out_path} exist, did nothing")
        return NII.load(out_path, True)
    global model, model_ax  # noqa: PLW0603
    nii_org = nii_org.copy()
    nii = nii_org.reorient().rescale_((0.8571, 0.8571, 0.8571))
    arr_out = nii.get_array().astype(float)
    if model is None:
        model = DAE_LitModel.load_from_checkpoint(checkpoint_sag, strict=False)
    if model_ax is None and checkpoint_ax is not None:
        model_ax = DAE_LitModel.load_from_checkpoint(checkpoint_ax, strict=False)
    model.to(device)
    ###### SAGITAL #####
    size = (nii.shape[0] + 8 - nii.shape[0] % 8, nii.shape[1] + 8 - nii.shape[1] % 8)
    rand = torch.randn((size[1], size[0]))
    batch_size_ = batch_size
    for i in tqdm(range(0, nii.shape[-1], batch_size_), desc="sagittal super-resolution"):
        j = i + batch_size_  # min(i + batches, arr_out.shape[-1] - 1)
        if j > arr_out.shape[-1]:
            j = None
            batch_size_ = arr_out.shape[-1] - i
        arr_new = arr_out[:, :, i:j].copy().astype(float)
        arr, pads = target_pad(arr_new, [size[0], size[1], batch_size_])

        reversed_pad = tuple(slice(b, -a if a != 0 else None) for a, b in pads)

        with torch.no_grad():
            img_lr = torch.from_numpy(arr).permute((2, 1, 0)).unsqueeze_(1).to(device, torch.float32)
            img_lr /= img_lr.max()
            img_lr = img_lr * 2 - 1
            cond = model.encode(img_lr)
            noise = torch.stack([rand for _ in range(img_lr.shape[0])], 0).unsqueeze_(1).to(img_lr.device)
            pred2: torch.Tensor = model.render(noise, cond, T=20, palette_condition=[img_lr])
            pred2 = pred2.squeeze_(1).permute((2, 1, 0)).cpu().numpy()[reversed_pad]
            arr_out[:, :, i:j] = pred2
    ##### AXIAL #####
    checkpoint_ax = None
    if checkpoint_ax is not None:
        arr_out_phase_1 = np.nan_to_num(arr_out, nan=0)
        batches_ = batch_size
        # arr_out += 1

        arr_out = arr_out_phase_1
        arr_out = (arr_out + 1) / 2
        size = (nii.shape[0] + 8 - nii.shape[0] % 8, nii.shape[2] + 8 - nii.shape[2] % 8)
        rand = torch.randn((size[1], size[0]))
        for i in tqdm(range(0, arr_out.shape[-2], batches_), desc="axial smoothing and outpainting"):
            j = i + batches_  # min(i + batches, arr_out.shape[-1] - 1)
            if j > arr_out.shape[-2]:
                j = None
                batches_ = arr_out.shape[-2] - i
            arr_new = arr_out[:, i:j, :].copy().astype(float)
            arr, pads = target_pad(arr_new, [size[0], batches_, size[1]])
            reversed_pad = tuple(slice(b, -a if a != 0 else None) for a, b in pads)

            with torch.no_grad():
                # PIR -> IRP
                img_lr = torch.from_numpy(arr).permute((1, 2, 0)).unsqueeze_(1).to(device, torch.float32)
                img_lr /= img_lr.max()
                img_lr = img_lr * 2 - 1
                # mask = torch.zeros_like(img_lr)
                # for idx in range(img_lr.shape[0]):
                #    try:
                #        crop = compute_crop_slice(img_lr[idx].clone().cpu().numpy(), -0.5, dist=-10)
                #        mask[idx, :, crop[-2], crop[-1]] = 1
                #    except ValueError:
                #        mask[idx] = 1
                # mask *
                img_lr = img_lr + torch.rand_like(img_lr) * 0.4

                cond = model_ax.encode(img_lr)
                noise = torch.stack([rand for _ in range(img_lr.shape[0])], 0).unsqueeze_(1).to(img_lr.device)
                pred2: torch.Tensor = model_ax.render(noise, cond, T=20, palette_condition=[img_lr])  #
                # IRP -> PIR
                pred2 = pred2.squeeze_(1).permute((2, 0, 1)).cpu().numpy()[reversed_pad]
                arr_out[:, i:j, :] = pred2
        #################
    nii_out = nii.set_array((arr_out + 1) * 200)

    nii_out.save(out_path)
    return nii_out


def filter_segmentation(subreg_nii: NII) -> NII:
    No_Logger().print("filter_segmentation", subreg_nii, ltype=Log_Type.BOLD)

    ### Quick filter step ###
    ccs, stats = subreg_nii.clamp(0, 1).get_segmentation_connected_components(1)
    arr = ccs[1]
    for key, volume in enumerate(stats[1]["voxel_counts"], 1):
        if volume <= 1000:
            arr[arr == key] = 0
    spinal_channel2 = subreg_nii.set_array(arr).clamp(0, 1).dilate_msk(1)
    subreg_nii_full = subreg_nii * spinal_channel2
    crop = subreg_nii_full.compute_crop()
    subreg_nii = subreg_nii.apply_crop(crop)
    ### Slow filter step ###
    ccs, stats = subreg_nii.extract_label([60, 61]).get_segmentation_connected_components(1)
    arr = ccs[1]
    for key, volume in enumerate(stats[1]["voxel_counts"], 1):
        if volume <= 500:
            arr[arr == key] = 0
    spinal_channel2 = subreg_nii.set_array(arr).clamp(0, 1)
    stop = spinal_channel2.sum()
    spinal_channel2.dilate_msk_(int(18 / subreg_nii.zoom[-1]))
    for _ in range(10):
        arr = subreg_nii * spinal_channel2
        spinal_channel2 = arr.clamp(0, 1)
        stop_new = spinal_channel2.sum()
        if stop == stop_new:
            print("early dilate msk break")
            break
        spinal_channel2.dilate_msk_(int(12 / subreg_nii.zoom[-1]))
        stop = stop_new
    arr = subreg_nii * spinal_channel2

    a = subreg_nii_full.get_array() * 0
    a[crop] = arr.get_array()
    return subreg_nii_full.set_array_(a)


def reduce_nii_size(t2w: BIDS_FILE, t2w_nii: NII):
    if t2w_nii.dtype == np.float64:
        t2w_nii.round(decimals=1).save(t2w, dtype=np.float32)
        t2w_nii = t2w.open_nii()
    return t2w_nii
