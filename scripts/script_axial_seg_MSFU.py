import gzip
import time
from math import ceil, floor
from pathlib import Path

import numpy as np
import torch
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
from TPTBox.spine.snapshot2D.snapshot_templates import Snapshot_Frame, create_snapshot, mri_snapshot
from tqdm import tqdm

from dataloader.datasets.dataset_superres import target_pad
from pl_models.DEA import DAE_LitModel

model: DAE_LitModel = None  # type: ignore
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


def upscale_nii(
    nii_org_bids: BIDS_FILE,
    batch_size=32,
    checkpoint_sag=default_checkpoint,
    checkpoint_ax=None,
    device=torch.device("cuda:0"),  # noqa: B008
    parent="rawdata_upscale",
    override_upscale=False,
):
    out_path = nii_org_bids.get_changed_path(info={"acq": "iso", "desc": "superres"}, parent=parent)
    if not override_upscale and out_path.exists():
        return NII.load(out_path, True), out_path
    global model, model_ax
    nii_org: BIDS_FILE = nii_org_bids.open_nii()
    nii = nii_org.reorient().rescale_((0.8571, 0.8571, 0.8571))
    arr_out = nii.get_array().astype(float)
    if model is None:
        if "CycleGAN" in checkpoint_sag:
            model = CycleGAN_LitModel.load_from_checkpoint(checkpoint_sag, strict=False)
        else:
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
    return nii_out, out_path


def filter_segmentation(subreg_nii: NII) -> NII:
    No_Logger().print("filter_segmentation", subreg_nii, type=Log_Type.BOLD)

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


def reduce_nii_size(t2w: BIDS_FILE, t2w_nii: NII):
    if t2w_nii.dtype == np.float64:
        t2w_nii.round(decimals=1).save(t2w, dtype=np.float32)
        t2w_nii = t2w.open_nii()
    return t2w_nii


def run(
    in_ds: Path,
    raw="rawdata",
    der="derivatives",
    override_upscale=False,
    batch_size=32,
    sort=True,
    device=torch.device("cuda:0"),  # noqa: B008
    resample_back=True,
    sag_only=False,
):
    iso = raw
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
    try:
        model_vert = get_model_vert("vert_highres").load()
    except KeyError:
        model_vert = get_model_vert("Vertebra_Highres").load()

    BIDS_Global_info.remove_splitting_key("chunk")
    bids_ds = BIDS_Global_info(datasets=[in_ds], parents=[parent_raw, parent_der], verbose=False)

    execution_times = []
    for name, subject in bids_ds.enumerate_subjects(sort=True):
        logger = head_logger.add_sub_logger(name=name)
        q = subject.new_query()
        q.flatten()
        q.filter("part", "inphase", required=False)
        # q.filter("acq", "ax")
        q.filter("seg", lambda x: x != "manual", required=False)
        q.filter("lesions", lambda x: x != "manual", required=False)
        # q.filter("desc", lambda _: False, required=False)
        q.unflatten()
        q.filter_format("T2w")
        q.filter_filetype("nii.gz")
        families = q.loop_dict(sort=sort, key_addendum=["acq"])
        for f in families:
            print(f)
            try:
                fid = f.family_id
                if "T2w_acq-sag" in f:
                    for t2w in f["T2w_acq-sag"]:
                        start_time = time.perf_counter()
                        reduce_nii_size(t2w, t2w.open_nii())
                        # Call to the pipeline
                        output_paths, errcode = process_img_nii(
                            img_ref=t2w,
                            derivative_name=der,
                            model_subreg=model_subreg,
                            model_vert=model_vert,
                            override_subreg=False,
                            override_vert=False,
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
                if "T2w_acq-ax" not in f or sag_only:
                    # logger.print(f"{fid}: T2w without part- not found, skip")
                    continue
                for t2w in f["T2w_acq-ax"]:
                    start_time = time.perf_counter()
                    t2w_nii = t2w.open_nii()
                    t2w_nii = reduce_nii_size(t2w, t2w_nii)
                    nii_iso, path_iso = upscale_nii(
                        nii_org_bids=t2w, parent=iso, override_upscale=override_upscale, batch_size=batch_size, device=device
                    )
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
                    # Load Outputs
                    seg_nii = NII.load(output_paths["out_spine"], seg=True)  # subregion mask
                    vert_nii = NII.load(output_paths["out_vert"], seg=True)  # subregion mask
                    poi = calc_centroids_from_subreg_vert(
                        vert_nii,
                        seg_nii,
                        subreg_id=[50, 100, Location.Spinal_Canal_ivd_lvl.value, Location.Spinal_Canal.value],
                        buffer_file=output_paths["out_ctd"],
                        save_buffer_file=True,
                    )
                    ### RESCALE BACK ###
                    if resample_back:
                        output_paths = output_paths_from_input(t2w, der, None)
                        out_spine = output_paths["out_spine"]
                        out_vert = output_paths["out_vert"]
                        out_ctd = output_paths["out_ctd"]
                        if not out_spine.exists() or override_upscale:
                            seg = seg_nii.map_labels({Location.Endplate.value: Location.Vertebra_Disc.value})
                            seg.resample_from_to_(t2w_nii).save(out_spine)
                        if not out_vert.exists() or override_upscale:
                            seg = vert_nii.map_labels({i: i % 100 + 100 for i in range(200, 500)})
                            seg.resample_from_to_(t2w_nii).save(out_vert)
                        if not out_ctd.exists() or override_upscale:
                            poi_lr = poi.resample_from_to(t2w_nii)
                            poi_lr.save(out_ctd)
                        # vertebra_level = BIDS_FILE(output_paths["out_spine"], t2w.dataset).get_changed_path(info={"seg": "vertebra-level"})

                        # poi_lr.extract_subregion(Location.Spinal_Canal_ivd_lvl.value).make_point_cloud_nii(s=4)[0].save(vertebra_level)
            except Exception:
                logger.print_error()
    if len(execution_times) > 0:
        head_logger.print(
            f"\nExecution times:\n{execution_times}\nRange:{min(execution_times)}, {max(execution_times)}\nAvg {np.average(execution_times)}"
        )


def validate(
    in_ds: Path,
    raw="rawdata",
    der="derivatives",
    override_upscale=False,
    sort=True,
):
    # INPUT
    in_ds = Path(in_ds)
    head_logger = No_Logger()  # (in_ds, log_filename="source-convert-to-unet-train", default_verbose=True)

    block = ""  # put i.e. 101 in here for block
    parent_raw = str(Path(raw).joinpath(str(block)))
    parent_der = str(Path(der).joinpath(str(block)))

    BIDS_Global_info.remove_splitting_key("chunk")
    bids_ds = BIDS_Global_info(datasets=[in_ds], parents=[parent_raw, parent_der], verbose=False)

    execution_times = []
    for name, subject in bids_ds.enumerate_subjects(sort=True):
        logger = head_logger.add_sub_logger(name=name)
        q = subject.new_query()
        q.flatten()
        q.filter("part", "inphase", required=False)
        q.filter("seg", lambda x: x != "manual", required=False)
        q.filter("lesions", lambda x: x != "manual", required=False)
        q.unflatten()
        q.filter_format("T2w")
        q.filter_filetype("nii.gz")
        q_sag = q.copy()
        q_sag.filter("acq", "sag")
        # q_sag.filter("desc", lambda _: False, required=False)

        q_ax = q.copy()
        q_ax.filter("acq", "iso")
        q_ax.filter("desc", "superres")

        sag = q_sag.loop_dict(sort=sort)
        ax = {next(iter(a.values()))[0].get("ses"): a for a in q_ax.loop_dict(sort=sort)}
        if len(ax) == 0:
            continue
        for f in sag:
            try:
                ses = next(iter(f.values()))[0].get("ses")
                ax_t2w = ax[ses]["T2w"][0]
                ax_subreg = ax[ses]["msk_seg-spine"][0]
                ax_vert = ax[ses]["msk_seg-vert"][0]
                poi_ax = ax[ses]["ctd_seg-spine"][0]

                sag_t2w = f["T2w"][0]
                sag_subreg = f["msk_seg-spine"][0]
                sag_vert = f["msk_seg-vert"][0]
                poi_sag = f["ctd_seg-spine"][0]
                snps = list(get_snap_name(ax_t2w, der))
                snps[0].parent.mkdir(exist_ok=True)
                snps[1].parent.mkdir(exist_ok=True)
                if snps[1].exists():
                    continue
                poi_ax = poi_ax.open_poi()
                poi_ax_new, mapping = remap_centroids(poi_ax, poi_sag.open_poi())
                sag_subreg.get_changed_path(info={"seg": "vertebra-level"}).unlink(missing_ok=True)
                vertebra_level = ax_subreg.get_changed_path(info={"seg": "vertebra-level"})
                if not vertebra_level.exists() or override_upscale:
                    p = poi_ax_new.extract_subregion(Location.Spinal_Canal_ivd_lvl.value)
                    p.make_point_cloud_nii(sphere=True, s=6)[0].save(vertebra_level)

                mapping = {k: v for k, v in mapping.items() if k != v}
                ## frame_0 = Snapshot_Frame(ax_t2w, segmentation=vertebra_level, centroids=cord_poi, hide_centroids=True, mode="MRI")
                # frame_1 = Snapshot_Frame(ax_t2w, segmentation=ax_vert, centroids=poi_ax_new.extract_subregion(50), coronal=True, mode="MRI")
                # frame_2 = Snapshot_Frame(ax_t2w, segmentation=ax_subreg, centroids=poi_ax.extract_subregion(50), coronal=True, mode="MRI")
                # frame_4 = Snapshot_Frame(sag_t2w, segmentation=sag_vert, centroids=poi_sag, coronal=True, mode="MRI")
                # f = [frame_1, frame_2, frame_4]
                # cord_poi = poi_ax_new.extract_subregion(Location.Spinal_Canal_ivd_lvl)
                # if len(cord_poi) >= 2:
                #    frame_3 = Snapshot_Frame(ax_t2w, centroids=cord_poi, coronal=True, mode="MRI")
                #    f.append(frame_3)
                # create_snapshot(snps, f)
                cord_poi = poi_ax_new.extract_subregion(Location.Spinal_Canal_ivd_lvl)
                # frame_0 = Snapshot_Frame(ax_t2w, segmentation=vertebra_level, centroids=cord_poi, hide_centroids=True, mode="MRI")
                frame_0 = Snapshot_Frame(sag_t2w, segmentation=sag_vert, centroids=poi_sag.open_poi().extract_subregion(50), mode="MRI")
                frame_1 = Snapshot_Frame(ax_t2w, segmentation=ax_vert, centroids=poi_ax_new.extract_subregion(50), coronal=True, mode="MRI")
                frame_2 = Snapshot_Frame(ax_t2w, segmentation=ax_subreg, centroids=poi_ax.extract_subregion(50), mode="MRI")
                frame_3 = Snapshot_Frame(ax_t2w, centroids=cord_poi, coronal=True, mode="MRI")
                create_snapshot(snps, [frame_0, frame_1, frame_2, frame_3])
                vertebra_level_snap = ax_t2w.get_changed_path(
                    "json", format="poi", info={"seg": "vertebra-level"}, parent="derivatives", make_parent=False
                )
                cord_poi.save(vertebra_level_snap)
                snap = ax_t2w.dataset / "derivatives" / "poi" / vertebra_level_snap.name
                cord_poi.save(snap, make_parents=True)
            except KeyError as e:
                logger.print("File Not Found:", f.family_id, e.args[0], Log_Type.FAIL)
            except gzip.BadGzipFile:
                logger.print("BadGzipFile:", f.family_id, Log_Type.FAIL)
                logger.print_error()
                # exit()
        # for f in families:
        #    try:
        #        pass
        #    except Exception:
        #        logger.print_error()
    if len(execution_times) > 0:
        head_logger.print(
            f"\nExecution times:\n{execution_times}\nRange:{min(execution_times)}, {max(execution_times)}\nAvg {np.average(execution_times)}"
        )


def get_snap_name(ax_stitched_og: BIDS_FILE, parent):
    vertebra_level_snap = ax_stitched_og.get_changed_bids(
        "jpg", format="snp", info={"seg": "vertebra-level"}, parent=parent, make_parent=False
    )
    snap = vertebra_level_snap.dataset / parent / "snapshots" / str(vertebra_level_snap.file["jpg"].name)
    return vertebra_level_snap.file["jpg"], snap


def remap_centroids(changing_poi: POI, fixed_poi: POI):
    ### rematch pois ###
    glob1 = changing_poi.to_global().extract_subregion(50)
    glob2 = fixed_poi.to_global()
    best = {}
    for key in glob2.keys_region():
        if (key, 50) not in glob2:
            continue
        stats = glob1.calculate_distances_cord(glob2[key, 50])
        min_key = min(stats, key=lambda key: stats[key])
        k, _ = min_key
        # print(k, key, stats[min_key])
        if k not in best or best[k][0] > stats[min_key]:
            best[k] = (stats[min_key], key)
    # glob1 to glob2
    mapping = dict(((k, best[k][1]) if k in best else (k, None)) for k in glob1.keys_region())
    for i in changing_poi.keys_region():
        if i >= 50:
            mapping[i] = None
    mapping[1] = min([a for a in mapping.values() if a is not None]) - 1

    # print(mapping)
    return changing_poi.map_labels(label_map_region=mapping), mapping


if __name__ == "__main__":
    ds = Path("/DATA/NAS/datasets_processed/MRI_spine/dataset-MSFU/")
    if ds.exists():
        run(ds)
        validate(Path(ds))

    else:
        ds = "/media/data/robert/datasets/dataset-neuropoly-test/"
        run(Path(ds), resample_back=True)
        validate(Path(ds))
    # pip install 'resize @ git+https://gitlab.com/iacl/resize@v0.3.0'
    # pip install 'degrade @ git+https://gitlab.com/iacl/degrade@v0.4.0'
    # pip install openpyxl
    # pip install tensorboardX
    # pip install protobuf==3.20
    # pip install pillow==9.0.1
    # pip install simplification
    # pip install monai
