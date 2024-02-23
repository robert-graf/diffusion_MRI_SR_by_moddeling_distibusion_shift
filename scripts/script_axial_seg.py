import time
from math import ceil, floor
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from mri_segmentor.seg_run import ErrCode, output_paths_from_input, process_img_nii
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
from TPTBox.stitching.stitching_tools import stitching
from tqdm import tqdm

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


if Path("/DATA/NAS/ongoing_projects/robert/code/dae/lightning_logs_msfu/").exists():
    default_checkpoint = "/DATA/NAS/ongoing_projects/robert/code/dae/lightning_logs_msfu/DAE_sag_pallet_v2/version_1/checkpoints/last.ckpt"
else:
    default_checkpoint = "/media/data/robert/code/dae/lightning_logs/DAE_sag_pallet_v2/version_1/checkpoints/last.ckpt"
    # default_checkpoint = "/media/data/robert/code/dae/lightning_logs_abl/DAE_abl_rcan/version_10/checkpoints/last.ckpt"


def upscale_nii(
    nii_org_bids: BIDS_FILE,
    batch_size=32,
    checkpoint_sag=default_checkpoint,
    # checkpoint_sag="/media/data/robert/code/dae/lightning_logs/DAE_sag_pallet_v2/version_1/checkpoints/last.ckpt",
    # checkpoint_sag="/media/data/robert/code/dae/lightning_logs_img2img/CUT_98_neuropoly/version_0/checkpoints/latest.ckpt",
    # checkpoint_sag="/media/data/robert/code/dae/lightning_logs/DAE_NAKO_160_palette_only_stop_gap/version_8/checkpoints/last.ckpt",  # "/media/data/robert/code/dae/lightning_logs_img2img/CycleGAN_pix2pix_super_sag_neuropoly/version_8/checkpoints/last.ckpt",  # "lightning_logs/DAE_sag_pallet_only/version_0/checkpoints/last.ckpt",  # "lightning_logs/DAE_sag_pallet_only/version_0/checkpoints/last.ckpt",
    checkpoint_ax=None,  # "lightning_logs/DAE_ax_pallet_only_outpainting/version_6/checkpoints/last.ckpt",
    device=torch.device("cuda:0"),  # noqa: B008
    parent="rawdata_upscale",
    override_upscale=False,
    reset_model=False,
):
    out_path = nii_org_bids.get_changed_path(info={"acq": "iso", "desc": "superres"}, parent=parent)
    global model, model_ax
    if reset_model:
        del model
        del model_ax
        torch.cuda.empty_cache()
        model = None  # type: ignore
        model_ax = None  # type: ignore

    if not override_upscale and out_path.exists():
        return NII.load(out_path, True), out_path

    nii_org: BIDS_FILE = nii_org_bids.open_nii()
    nii = nii_org.reorient().rescale_((0.8571, 0.8571, 0.8571))
    arr_out = nii.get_array().astype(float)
    if model is None:
        if "CycleGAN" in checkpoint_sag:
            model = CycleGAN_LitModel.load_from_checkpoint(checkpoint_sag, strict=False, map_location=device)
        else:
            model = DAE_LitModel.load_from_checkpoint(checkpoint_sag, strict=False, map_location=device)
        model.to(device)
    if model_ax is None and checkpoint_ax is not None:
        model_ax = DAE_LitModel.load_from_checkpoint(checkpoint_ax, strict=False)
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
    try:
        nii_out = nii_out.nan_to_num()
    except AttributeError:
        pass

    nii_out.save(out_path)
    return nii_out, out_path


def filter_segmentation_old(subreg_nii: NII) -> NII:
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
    vert_nii.resample_from_to(ax_file).save(output_paths["out_vert"], make_parents=True)
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


def run(
    in_ds: Path,
    raw="rawdata",
    der="derivatives",
    iso="rawdata_upscale",
    override_upscale=False,
    batch_size=32,
    stitching_only=False,
    sort=True,
    override_subreg=False,
    override_vert=False,
    scale_back=True,
):
    # INPUT
    in_ds = Path(in_ds)
    head_logger = No_Logger()  # (in_ds, log_filename="source-convert-to-unet-train", default_verbose=True)

    block = ""  # put i.e. 101 in here for block
    parent_raw = str(Path(raw).joinpath(str(block)))
    parent_der = str(Path(der).joinpath(str(block)))
    from mri_segmentor import get_model_spine, get_model_vert

    # check available models
    # modelid2folder_subreg, modelid2folder_vert = check_available_models(model_dir, verbose=True)
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
                stitched = f["T2w"][0].get_changed_path(info={"desc": "stitched", "chunk": None}, parent=iso, make_parent=True)
                if not stitched.exists():
                    if len(f["T2w"]) == 1:
                        f["T2w"][0].open_nii().save(stitched)
                    else:
                        stitching(*f["T2w"], out=stitched, verbose_stitching=True, bias_field=True)
                if stitching_only:
                    continue
                start_time = time.perf_counter()
                ref_t2w: BIDS_FILE = BIDS_FILE(stitched, f["T2w"][0].dataset)
                logger.print("Upscale ", fid)
                nii_iso, path_iso = upscale_nii(nii_org_bids=ref_t2w, parent=iso, override_upscale=override_upscale, batch_size=batch_size)
                # Call to the pipeline
                output_paths, errcode = process_img_nii(
                    img_ref=BIDS_FILE(path_iso, ref_t2w.dataset),
                    derivative_name=der,
                    model_subreg=model_subreg,
                    model_vert=model_vert,
                    override_subreg=override_subreg,
                    override_vert=override_vert,
                    override_ctd=override_vert,
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
                if scale_back:
                    poi = calc_centroids_from_subreg_vert(
                        vert_nii,
                        seg_nii,
                        subreg_id=[
                            50,
                            100,
                            Location.Ligament_Attachment_Point_Anterior_Longitudinal_Inferior_Median.value,
                            # Location.Vertebra_Disc_Posterior.value,
                        ],
                        buffer_file=output_paths["out_ctd"],
                        save_buffer_file=True,
                    )

                    for ax_file in f["T2w"]:
                        scale_back_seg(ax_file, seg_nii, vert_nii, poi)

                # TODO do something with outputs, potentially saving them again to the output paths

            except Exception:
                logger.print_error()

    if len(execution_times) > 0:
        head_logger.print(
            f"\nExecution times:\n{execution_times}\nRange:{min(execution_times)}, {max(execution_times)}\nAvg {np.average(execution_times)}"
        )


if __name__ == "__main__":
    ds = "/media/data/robert/datasets/dataset-neuropoly/"
    ds = "/media/data/robert/test_hendrik/dataset-neuropoly2/"
    ds = "/media/data/robert/test_paper4/"
    # Parallel(n_jobs=3)(
    #    [delayed(run)(ds, stitching_only=True, sort=True), delayed(run)(ds, stitching_only=True, sort=False), delayed(run)(ds)]
    # )
    # run(Path(ds))
    from script_axial_seg_no_stiching import run as run_sag
    from script_axial_seg_no_stiching import validate

    Parallel(n_jobs=4)(
        [
            delayed(run)(Path(ds), iso="rawdata", scale_back=False, sort=True),
            # delayed(run)(Path(ds), iso="rawdata", scale_back=False, sort=False),
            delayed(run_sag)(Path(ds), der="derivatives", raw="rawdata", sort=False, sag_only=True),
            delayed(run_sag)(Path(ds), der="derivatives", raw="rawdata", sort=True, sag_only=True),
            delayed(validate)(Path(ds), sort=True),
        ]
    )
    # run(Path(ds), iso="rawdata", override_vert=True, override_subreg=True, scale_back=False)
    # run_sag(Path(ds), der="derivatives", out_par="rawdata")

    validate(Path(ds))
    # pip install 'resize @ git+https://gitlab.com/iacl/resize@v0.3.0'
    # pip install 'degrade @ git+https://gitlab.com/iacl/degrade@v0.4.0'
    # pip install openpyxl
    # pip install tensorboardX
    # pip install protobuf==3.20
    # pip install pillow==9.0.1
    # pip install simplification
    # pip install monai
