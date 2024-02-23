### Vertebra dtr↑ Betti b0 er↓ Betti b1 er↓ Betti b2 er on stitched images ###
### Fast preprocessing by given mask
### save result in individual derivative folder.

from functools import partial
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
from mri_segmentor import get_model_spine, get_model_vert
from script_axial_seg import upscale_nii
from TPTBox import NII, POI, BIDS_Family, Log_Type, No_Logger, to_nii
from TPTBox.core.np_utils import np_dice

from scripts.loops import get_sessions


def make_fast_preprocessing(fam: BIDS_Family, x: NII):
    # if "msk_seg-postprocessing" in fam:
    #    return fam["msk_seg-postprocessing"][0].open_nii_reorient()
    # else:
    vert_bids = fam["msk_seg-vert"][0]
    subreg_bids = fam["msk_seg-spine"][0]
    subreg = subreg_bids.open_nii_reorient()
    vert_nii = vert_bids.open_nii_reorient() + subreg.extract_label([60, 61, 62, 42])
    arr = vert_nii.get_array()
    mask = vert_nii.get_array()
    vert_nii.compute_crop()
    for i in range(arr.shape[1]):
        array = arr[:, i]
        shp = vert_nii.shape
        d = 5
        msk_bin = np.zeros(array.shape, dtype=bool)
        # bool_arr[array<minimum] = 0
        msk_bin[array >= 1] = 1
        # msk_bin = np.asanyarray(bool_arr, dtype=bool)
        msk_bin[np.isnan(msk_bin)] = 0
        cor_msk = np.where(msk_bin > 0)
        if cor_msk[0].shape[0] == 0:
            continue
        c_min = [cor_msk[0].min(), cor_msk[1].min()]
        c_max = [cor_msk[0].max(), cor_msk[1].max()]
        x0 = c_min[0] - d if (c_min[0] - d) > 0 else 0
        y0 = c_min[1] - d if (c_min[1] - d) > 0 else 0
        x1 = c_max[0] + d if (c_max[0] + d) < shp[0] else shp[0]
        y1 = c_max[1] + d if (c_max[1] + d) < shp[1] else shp[1]
        ex_slice = [slice(x0, x1 + 1), slice(i, i + 1), slice(y0, y1 + 1)]
        mask[tuple(ex_slice)] = 1
    mask = vert_nii.set_array(mask).resample_from_to_(x).clamp(0, 1)

    mask.save(vert_bids.get_changed_path(info={"seg": "postprocessing"}))
    return mask


def filter_segmentation(x: NII, fam: BIDS_Family):
    mask = make_fast_preprocessing(fam, x)

    return x * mask


old = False
# Load segmentation model
if old:
    model_subreg = get_model_spine("T2w_Segmentor_old").load()
else:
    model_subreg = get_model_spine("T2w_Segmentor").load()
try:
    model_vert = get_model_vert("vert_highres").load()
except KeyError:
    model_vert = get_model_vert("Vertebra_Highres").load()
# Load SR model TODO ADD a list of Paths (set model_name and upscale_nii checkpoint_sag)


def get_latest_checkpoint(path: Path) -> str | None:
    import glob
    import os

    checkpoints = sorted(
        glob.glob(f"{path!s}/version_*/checkpoints/last.ckpt"),
        key=os.path.getmtime,
    )
    return None if len(checkpoints) == 0 else str(checkpoints[-1])


models_paths = {s.name: get_latest_checkpoint(s) for s in Path("lightning_logs_abl").iterdir()}
models_paths["upscale-only"] = "upscale-only"
print(models_paths)
ds = "/media/data/robert/test_paper4_2/"
# ds = "/media/data/robert/test_paper4_fig/"
summary = {}
for max_limit in [5000]:
    for model_name, model_path in models_paths.items():
        try:
            limit = max_limit
            if model_path is None:
                continue
            reset_model = True
            parent = f"eval_{model_name}"
            # Pandas dict
            pd_dict = {"name": [], "dice": [], "ssim": [], "psnr": [], "precision": []}

            for ses in get_sessions(ds, require_sag=False, require_stiched=False):
                limit -= 1
                if limit == 0:
                    break

                for chunk in ses.axial_chunks:
                    fam = ses.get_seg(chunk)
                    if "msk_seg-vertslice" not in fam:
                        assert "vertslice" in fam, fam
                        p = fam["vertslice"][0]
                        # seg-vert_
                        if p.exists():
                            p_new = Path(str(p.file["nii.gz"]).replace("_T2w_vertslice.nii.gz", "_seg-vertslice_msk.nii.gz"))
                            p.file["nii.gz"].rename(p_new)
                            print(p, p_new)
                            continue
                        else:
                            print(chunk, fam)
                            exit()
                            continue
                    gt = fam["msk_seg-vertslice"][0]
                    # Translate
                    if model_name.lower() == "upscale-only":
                        out_path = chunk.get_changed_bids(info={"acq": "iso", "desc": "superres"}, parent=parent)
                        T2w_iso = out_path
                        if not out_path.exists():
                            nii_org = chunk.open_nii()
                            nii = nii_org.reorient().rescale_((0.8571, 0.8571, 0.8571))
                            nii.save(out_path)
                    else:
                        T2w_iso = chunk.get_changed_bids(info={"acq": "iso", "desc": "superres"}, parent=parent)
                        if old and not T2w_iso.exists():
                            No_Logger().print("Missing SKIP (old seg)", Log_Type.FAIL)
                            continue
                        print(chunk)
                        upscale_nii(
                            chunk,
                            parent=parent,
                            reset_model=reset_model,
                            checkpoint_sag=model_path,
                            batch_size=32 if "conv" in model_name.lower() else 48,
                        )
                        T2w_iso = chunk.get_changed_bids(info={"acq": "iso", "desc": "superres"}, parent=parent)
                        reset_model = False
                        # Segment
                    from mri_segmentor.seg_run import process_img_nii

                    a = partial(filter_segmentation, fam=fam)
                    output_paths, errcode = process_img_nii(
                        img_ref=T2w_iso,
                        derivative_name=parent if not old else parent + "/old_seg",
                        model_subreg=model_subreg,
                        model_vert=model_vert,
                        override_subreg=False,
                        override_vert=False,
                        lambda_subreg=a,  # type: ignore
                        save_debug_data=False,
                        verbose=False,
                        proc_n4correction=False,
                    )
                    # Load Outputs
                    try:
                        seg_nii = NII.load(output_paths["out_spine"], seg=True)  # subregion mask
                    except FileNotFoundError:
                        seg_nii = to_nii(T2w_iso).nan_to_num() * 0
                        seg_nii.seg = True
                    try:
                        vert_nii = NII.load(output_paths["out_vert"], seg=True)  # vert mask
                        poi = POI.load(output_paths["out_ctd"])
                    except FileNotFoundError:
                        vert_nii = seg_nii * 0
                        poi = POI({}, **vert_nii._extract_affine())
                    # Measure detection rate
                    if "dice" in poi.info and "precision" in poi.info:
                        pass
                    else:
                        try:
                            gt = gt.open_nii()
                        except nibabel.filebasedimages.ImageFileError:
                            a = Path(str(gt.file["nii.gz"]).replace(".nii.gz", ".nii"))
                            gt.file["nii.gz"].rename(a)
                            gt_nii = to_nii(a)
                            # gt_nii.save(str(gt.file["nii.gz"]))
                            # a.unlink()
                            gt = gt.open_nii()
                        gt_np = gt.get_array()
                        # DICE
                        height = np.unique(np.where(gt_np)[-1])

                        assert len(height) == 1, (height, gt, gt.file["nii.gz"])
                        height = height[0]

                        pred_np = seg_nii.extract_label(list(range(40, 51))).resample_from_to(gt).get_array()[..., height]
                        gt_slice = gt_np[..., height]
                        dice = np_dice(pred_np, gt_slice)
                        poi.info["dice"] = dice
                        precision = float(np.sum(pred_np * gt_slice) / np.sum(gt_slice))
                        poi.info["precision"] = precision
                        poi.save(output_paths["out_ctd"])

                    if "ssim" not in poi.info:
                        chunk_nii = chunk.open_nii()
                        iso_nii = T2w_iso.open_nii()
                        iso_nii.seg = True  # Nearest Neighbor
                        iso_nii.resample_from_to_(chunk_nii)
                        iso_nii.seg = False
                        iso_nii.normalize_(clamp_lower=0)
                        chunk_nii.normalize_(clamp_lower=0)
                        # iso_nii = iso_nii / iso_nii.get_array().mean()
                        # chunk_nii = chunk_nii / chunk_nii.get_array().mean()
                        psnr = iso_nii.psnr(chunk_nii)
                        ssim = iso_nii.ssim(chunk_nii)
                        poi.info["psnr"] = psnr
                        poi.info["ssim"] = ssim
                        poi.save(output_paths["out_ctd"])
                    for k in pd_dict:
                        if k == "name":
                            pd_dict[k].append(str(chunk))
                        else:
                            pd_dict[k].append(poi.info[k])
                    summary[model_name] = {}
                    for k, v in pd_dict.items():
                        if k == "name":
                            continue
                        summary[model_name][k] = round(sum(v) / max(1, len(v)), 4)
                        summary[model_name]["len"] = len(v)
                        print(k, summary[model_name][k], len(v))

                p = Path(ds, parent, "dice.xlsx")
                p.parent.mkdir(exist_ok=True)
                pd.DataFrame(pd_dict).to_excel(p)

            print("############################################################")
            print("######################## summary ###########################")
            for i in sorted(summary.items(), key=lambda x: -x[1]["psnr"]):
                print(i[0], ":", i[1])
            print("############################################################")
            print("############################################################")

        except Exception:
            No_Logger().print_error()


print("############################################################")
print("######################## summary ###########################")
for i in sorted(summary.items(), key=lambda x: -x[1]["dice"]):
    print(i[0], ":", i[1])
print("############################################################")
print("############################################################")

print("############################################################")
print("######################## summary ###########################")
for i in sorted(summary.items(), key=lambda x: -x[1]["psnr"]):
    print(i[0], ":", i[1])
print("############################################################")
print("############################################################")


print("############################################################")
print("######################## summary ###########################")
for i in sorted(summary.items(), key=lambda x: -x[1]["ssim"]):
    print(i[0], ":", i[1])
print("############################################################")
print("############################################################")
# python train_DAE.py --config config/neuropoly/super_ax/03_abliation_cubic.conf
# /home/robert/anaconda3/envs/mri_spine_seg/bin/python /media/data/robert/code/dae/test_scripts.py
# python train_DAE.py --config config/neuropoly/super_ax/04_abliation_cubic.conf
# /home/robert/anaconda3/envs/mri_spine_seg/bin/python /media/data/robert/code/dae/test_scripts.py
