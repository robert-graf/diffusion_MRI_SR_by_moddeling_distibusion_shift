import random
import time
from math import ceil, floor
from pathlib import Path

import numpy as np
import torch
from mri_segmentor.seg_run import ErrCode, output_paths_from_input, process_img_nii
from torch.nn import functional as F
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
from TPTBox.spine.snapshot2D.snapshot_templates import mri_snapshot
from TPTBox.stitching import stitching

from dataloader.datasets.dataset_superres import target_pad
from pl_models.cycleGAN import CycleGAN_LitModel
from pl_models.DEA import DAE_LitModel

model: DAE_LitModel | CycleGAN_LitModel = None  # type: ignore
model_ax: DAE_LitModel = None  # type: ignore


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
                min_w = min_w.stop - min_w.start
            curr_w = ex_slice[i].stop - ex_slice[i].start
            dif = min_w - curr_w
            if min_w > 0:
                new_start = ex_slice[i].start - floor(dif / 2)
                new_goal = ex_slice[i].stop + ceil(dif / 2)
                if new_goal > shp[i]:
                    new_start -= new_goal - shp[i]
                    new_goal = shp[i]
                if new_start < 0:  #
                    new_goal -= new_start
                    new_start = 0
                ex_slice[i] = slice(new_start, new_goal)

    return tuple(ex_slice)  # type: ignore


def padded_shape(shape):
    shape = list(shape)
    for i, j in enumerate(shape):
        shape[i] = j + 8 - (j % 8)
    return shape


def random_crop(target_shape: tuple[int, int, int], *arrs: torch.Tensor):
    sli = [slice(None), slice(None), slice(None)]
    for i in range(3):
        z = max(0, arrs[0].shape[-i] - target_shape[-i])
        if z != 0:
            r = random.randint(0, z)
            r2 = r + target_shape[-i]
            sli[-i] = slice(r, r2 if r2 != arrs[0].shape[-i] else None)

    return tuple(a[..., sli[0], sli[1], sli[2]] for a in arrs)


def pad_size(x: torch.Tensor, target_shape, mode="constant"):
    print(x.shape[-3:], target_shape)
    padding = []
    for in_size, out_size in zip(reversed(x.shape[-3:]), reversed(target_shape), strict=True):
        to_pad_size = max(0, out_size - in_size) / 2.0
        padding.extend([ceil(to_pad_size), floor(to_pad_size)])
    x_ = (
        F.pad(x.unsqueeze(0).unsqueeze(0), padding, mode=mode).squeeze(0).squeeze(0)
    )  # mode   'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
    # print("Padding", x_.shape, x.shape, padding)
    return x_


def transform_(items_in, target_shape, condition_types=("T2w"), padding="reflect", flip=False):
    # Transform to tensor
    items = map(torch.Tensor, items_in)
    assert len(target_shape) == 3, target_shape
    ## Padding
    items = [pad_size(x, target_shape, padding) for x in items]
    # Coordinate-encoding
    shape = items[0].shape
    l1 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], shape[2], 1))
    l2 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], shape[2], 1))
    l3 = np.tile(np.linspace(0, 1, shape[2]), (shape[0], shape[1], 1))
    l1 = torch.Tensor(l1).permute(2, 0, 1)
    l2 = torch.Tensor(l2).permute(0, 2, 1)
    l3 = torch.Tensor(l3)
    assert l1.shape == l2.shape, (l1.shape, l2.shape)
    assert l3.shape == l2.shape, (l3.shape, l2.shape)
    assert shape == l2.shape, (shape, l2.shape)
    items.append(l1)
    items.append(l2)
    items.append(l3)
    ## Random crop
    items = list(random_crop(target_shape, *items))

    for i, (x, y) in enumerate(zip(items, condition_types)):  # noqa: B905
        # if y in mri_names:
        #    items[i] = x  # mri_transform(x)
        # if y.upper() != "CT":
        items[i] = items[i] * 2 - 1
    # Random flipping
    out = tuple(a.to(torch.float32).unsqueeze_(0) for a in items)
    return torch.cat(out, 0)


def upscale_nii(
    nii_org_bids: BIDS_FILE,
    batch_size=64,
    checkpoint_sag="/media/data/robert/code/dae/lightning_logs_img2img/CycleGAN_pix2pix_super_paired_3D_neuropoly/version_9/checkpoints/last.ckpt",
    # checkpoint_sag="/media/data/robert/code/dae/lightning_logs_img2img/CUT_98_neuropoly/version_0/checkpoints/latest.ckpt",
    # checkpoint_sag="/media/data/robert/code/dae/lightning_logs/DAE_NAKO_160_palette_only_stop_gap/version_8/checkpoints/last.ckpt",  # "/media/data/robert/code/dae/lightning_logs_img2img/CycleGAN_pix2pix_super_sag_neuropoly/version_8/checkpoints/last.ckpt",  # "lightning_logs/DAE_sag_pallet_only/version_0/checkpoints/last.ckpt",  # "lightning_logs/DAE_sag_pallet_only/version_0/checkpoints/last.ckpt",
    checkpoint_ax=None,  # "lightning_logs/DAE_ax_pallet_only_outpainting/version_6/checkpoints/last.ckpt",
    device=torch.device("cuda:0"),  # noqa: B008
    parent="rawdata_upscale_paired",
    override_upscale=False,
):
    out_path = nii_org_bids.get_changed_path(info={"acq": "iso", "desc": "superres"}, parent=parent)
    if not override_upscale and out_path.exists():
        return NII.load(out_path, True), out_path
    global model, model_ax
    nii_org: BIDS_FILE = nii_org_bids.open_nii()
    nii = nii_org.reorient(("R", "I", "P")).rescale_((0.8571, 0.8571, 0.8571)).normalize()
    crop = nii.compute_crop(dist=-30)
    nii2 = nii.apply_crop(crop)
    arr_out = nii2.get_array().astype(float)
    arr_out /= np.max(arr_out)

    if model is None:
        if "CycleGAN" in checkpoint_sag or "CUT" in checkpoint_sag:
            model = CycleGAN_LitModel.load_from_checkpoint(checkpoint_sag, strict=False)

        else:
            model = DAE_LitModel.load_from_checkpoint(checkpoint_sag, strict=False)
    model.to(device)
    if model_ax is None and checkpoint_ax is not None:
        model_ax = DAE_LitModel.load_from_checkpoint(checkpoint_ax, strict=False)
    ###### SAGITAL #####
    size = (nii2.shape[0] + 8 - nii2.shape[0] % 8, nii2.shape[1] + 8 - nii2.shape[1] % 8, nii2.shape[2] + 8 - nii2.shape[2] % 8)
    rand = None  # torch.randn((size[1], size[0]))
    # batch_size_ = batch_size
    # for i in tqdm(range(0, nii.shape[-1], batch_size_), desc="sagittal super-resolution"):
    #    j = i + batch_size_  # min(i + batches, arr_out.shape[-1] - 1)
    #    if j > arr_out.shape[-1]:
    #        j = None
    #        batch_size_ = arr_out.shape[-1] - i
    #        if batch_size_ % 8 != 0:
    #            batch_size_ = batch_size_ + 8 - batch_size_ % 8
    arr_new = arr_out.copy().astype(float)
    arr, pads = target_pad(arr_new, [size[0], size[1], size[2]])

    reversed_pad = tuple(slice(b, -a if a != 0 else None) for a, b in pads)

    with torch.no_grad():
        img_lr = torch.from_numpy(arr)  # .permute((2, 1, 0))
        img_lr /= img_lr.max()
        img_lr = img_lr * 2 - 1
        # if model.opt.dims == 3:
        if model.opt.linspace:
            img_lr = transform_([img_lr], img_lr.shape).to(device, torch.float32).unsqueeze(0)
        else:
            img_lr = img_lr.to(device, torch.float32).unsqueeze(0).unsqueeze(0)
        print(img_lr.shape)
        # img_lr = img_lr.swapaxes(0, 1)
        # out = {"x": img_lr}
        # transforms3D.add_linspace_embedding(out, img_lr.shape, dim=3, crop=None, pad=None, name_addendum="", device=img_lr.device)
        # img_lr = torch.stack(list(out.values()), 1).to(img_lr.device)
        # img_lr = img_lr * 2 - 1
        # cond = model.encode(img_lr)

        # noise = torch.stack([rand for _ in range(img_lr.shape[0])], 0).unsqueeze_(1).to(img_lr.device)
        # noise = None
        # pred2: torch.Tensor = model.render(noise, cond, T=20, palette_condition=[img_lr])
        print(img_lr.shape, img_lr.min(), img_lr.max())
        pred2: torch.Tensor = model.forward(img_lr)
        # if model.opt.dims == 3:
        pred2 = pred2.squeeze_()
        print(pred2.shape, img_lr.shape)
        pred2 = pred2.cpu().numpy()[reversed_pad]
        arr_out = pred2

        #################
    a = nii.get_array()
    a[crop] = (arr_out + 1) * 200
    nii_out = nii.set_array(a)

    nii_out.save(out_path)
    return nii_out, out_path


def filter_segmentation(subreg_nii: NII) -> NII:
    No_Logger().print("filter_segmentation", subreg_nii, type=Log_Type.BOLD)

    arr = subreg_nii.extract_label([61, 60]).dilate_msk_(int(5 / subreg_nii.zoom[-1]))
    spinal_channel2 = arr.get_largest_k_segmentation_connected_components(1, 1, return_original_labels=False)
    stop = spinal_channel2.sum()
    for _ in range(10):
        spinal_channel2.dilate_msk_(int(4 / subreg_nii.zoom[-1]))
        arr = subreg_nii * spinal_channel2
        spinal_channel2 = arr.clamp(0, 1)
        stop_new = spinal_channel2.sum()
        if stop == stop_new:
            print("early dilate msk break")
            break
        stop = stop_new
    # spinal_channel = spinal_channel2.dilate_msk(int(20 / subreg_nii.zoom[-1]))
    arr = subreg_nii * spinal_channel2
    # arr += spinal_channel2
    return arr  # .set_array(arr.get_array() + spinal_channel.get_array())


def scale_back_seg(ax_file: BIDS_FILE, subreg_nii_r: Image_Reference, vert_nii_r: Image_Reference, poi: POI):
    snapshot_copy_folder = ax_file.dataset / "derivatives_ax" / "snap"
    output_paths = output_paths_from_input(ax_file, "derivatives_ax", snapshot_copy_folder)
    vert_nii = to_nii(vert_nii_r, True)
    vert_nii = vert_nii.map_labels({k: k % 100 + 100 for k in range(101, 301)}, verbose=False)
    vert_nii.resample_from_to(ax_file).save(output_paths["out_vert"])
    subreg_nii = to_nii(subreg_nii_r, True).map_labels({62: 100}, verbose=False)
    subreg_nii.resample_from_to(ax_file).save(output_paths["out_spine"])
    poi = poi.to_global().to_other_nii(ax_file)  # .filter_points_inside_shape()
    poi.save(output_paths["out_ctd"])
    out_snap = output_paths["out_snap"]
    out_snap2 = output_paths["out_snap"]
    # Snapshot
    if not Path(out_snap).exists():
        # make only snapshot
        if snapshot_copy_folder is not None:
            out_snap = [out_snap, out_snap2]
        poi = poi.extract_subregion(Location.Ligament_Attachment_Point_Posterior_Longitudinal_Inferior_Median)

        mri_snapshot(ax_file, output_paths["out_vert"], poi, subreg_msk=output_paths["out_spine"], out_path=out_snap)


#
# poi_ax = poi.to_global().to_other_nii(nii_org)
## TODO Save


def run(
    in_ds: Path,
    raw="rawdata",
    der="derivatives_paired",
    iso="rawdata_upscale_paired",
    iso2="rawdata_upscale",
    override_upscale=False,
    batch_size=64,
    stitching_only=False,
    sort=True,
):
    # INPUT
    in_ds = Path(in_ds)
    head_logger = No_Logger()  # (in_ds, log_filename="source-convert-to-unet-train", default_verbose=True)

    block = ""  # put i.e. 101 in here for block
    parent_raw = str(Path(raw).joinpath(str(block)))
    parent_der = str(Path(der).joinpath(str(block)))

    from mri_segmentor import get_model_spine, get_model_vert

    model_subreg = get_model_spine("T2w_Segmentor").load()
    model_vert = get_model_vert("Vertebra_Highres").load()

    BIDS_Global_info.remove_splitting_key("chunk")
    bids_ds = BIDS_Global_info(datasets=[in_ds], parents=[parent_raw, parent_der], verbose=False)

    execution_times = []

    for name, subject in bids_ds.enumerate_subjects(sort=True):
        logger = head_logger.add_sub_logger(name=name)
        q = subject.new_query()
        q.flatten()
        q.filter("part", "inphase", required=False)
        q.filter("acq", "ax")
        q.filter("seg", lambda x: x != "manual", required=False)
        q.filter("lesions", lambda x: x != "manual", required=False)
        q.filter("desc", lambda x: False, required=False)
        q.unflatten()
        q.filter_format("T2w")
        q.filter_filetype("nii.gz")
        families = q.loop_dict(sort=sort)
        for f in families:
            try:
                fid = f.family_id

                if "T2w" not in f:
                    logger.print(f"{fid}: T2w without part- not found, skip")
                    continue
                stitched = None
                stitched = f["T2w"][0].get_changed_bids(info={"desc": "stitched", "chunk": None}, parent=iso2, make_parent=True)
                if stitched is not None and not stitched.exists():
                    stitching(*f["T2w"], out=stitched, verbose_stitching=True, bias_field=True)
                if stitching_only:
                    continue
                start_time = time.perf_counter()
                # ref_t2w: BIDS_FILE = BIDS_FILE(stitched, f["T2w"][0].dataset)
                logger.print("Upscale ", fid)
                loop = [stitched]
                if stitched is None:
                    loop = f["T2w"]
                for t2w in loop:
                    nii_iso, path_iso = upscale_nii(nii_org_bids=t2w, parent=iso, override_upscale=override_upscale, batch_size=batch_size)
                    # Call to the pipeline
                    output_paths, errcode = process_img_nii(
                        img_ref=BIDS_FILE(path_iso, t2w.dataset),
                        derivative_name=der,
                        model_subreg=model_subreg,
                        model_vert=model_vert,
                        override_subreg=False,
                        override_vert=False,
                        lambda_subreg=filter_segmentation,
                        save_debug_data=False,
                        verbose=False,
                    )
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    logger.print(f"Inference time is: {execution_time}")
                    execution_times.append(execution_time)
                    if errcode == ErrCode.UNKNOWN:
                        continue
                    if errcode not in [ErrCode.OK, ErrCode.ALL_DONE]:
                        logger.print(f"{fid}: Pipeline threw error code {errcode}")
                        # TODO continue? assert?

                    # Load Outputs
                    img_nii = nii_iso
                    seg_nii = NII.load(output_paths["out_spine"], seg=True)  # subregion mask
                    vert_nii = NII.load(output_paths["out_vert"], seg=True)  # subregion mask
                    poi = calc_centroids_from_subreg_vert(
                        vert_nii,
                        seg_nii,
                        subreg_id=[
                            *list(range(40, 51)),
                            100,
                            # Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Median.value,
                            # Location.Vertebra_Disc_Posterior.value,
                        ],
                        buffer_file=output_paths["out_ctd"],
                        save_buffer_file=True,
                    )
                    # for ax_file in f["T2w"]:
                    #    scale_back_seg(ax_file, seg_nii, vert_nii, poi)

                # TODO do something with outputs, potentially saving them again to the output paths

            except Exception:
                logger.print_error()

    if len(execution_times) > 0:
        head_logger.print(
            f"\nExecution times:\n{execution_times}\nRange:{min(execution_times)}, {max(execution_times)}\nAvg {np.average(execution_times)}"
        )


if __name__ == "__main__":
    ds = "/media/data/robert/datasets/dataset-neuropoly/"
    ds = "/media/data/robert/test_hendrik/dataset-neuropoly/"
    # ds = "/media/data/robert/test_hendrik/dataset-reg/"

    # Parallel(n_jobs=3)(
    #    [delayed(run)(ds, stitching_only=True, sort=True), delayed(run)(ds, stitching_only=True, sort=False), delayed(run)(ds)]
    # )
    run(Path(ds))

    # pip install 'resize @ git+https://gitlab.com/iacl/resize@v0.3.0'
    # pip install 'degrade @ git+https://gitlab.com/iacl/degrade@v0.4.0'
    # pip install openpyxl
    # pip install tensorboardX
    # pip install protobuf==3.20
    # pip install pillow==9.0.1
    # pip install simplification
    # pip install monai
