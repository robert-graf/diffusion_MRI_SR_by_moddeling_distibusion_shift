### Vertebra dtr↑ Betti b0 er↓ Betti b1 er↓ Betti b2 er on stitched images ###
### Fast preprocessing by given mask
### save result in individual derivative folder.

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mri_segmentor import get_model_spine, get_model_vert
from script_axial_seg import upscale_nii
from TPTBox import NII, POI, No_Logger, to_nii

from scripts.loops import get_sessions


def make_fast_preprocessing(ses):
    fam = ses.get_iso_stiched_seg()
    if "msk_seg-postprocessing" in fam:
        pass
    else:
        vert = ses.get_iso_stiched_seg()["msk_seg-vert"][0]
        subreg = ses.get_iso_stiched_seg()["msk_seg-spine"][0].open_nii_reorient()
        vert_nii = vert.open_nii_reorient() + subreg.extract_label([60, 61, 62, 42])
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
        mask = vert_nii.set_array(mask)

        mask.save(vert.get_changed_path(info={"seg": "postprocessing"}))


def filter_segmentation(x: NII, mask: NII):
    print(x, mask)
    return x * mask


# Load segmentation model
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
skip_list = [
    "eval_DAE_7_abl_ConvEmbpallet",
    "eval_DAE_6_abliation_06_lr_histogram_shifts",
    "eval_DAE_11_abliation_11_other_ds",
    "eval_DAE_?_abl_pallet",
    "eval_DAE_12_abliation_12_long",
    "eval_DAE_11_abliation_11_long",
    # "eval_DAE_9_abliation_09",
    # "eval_DAE_8_abliation_08",
    # "eval_DAE_7_abliation_07",
    "eval_DAE_111_abliation_114_p_2",
    "eval_DAE_111_abliation_114",
    # "eval_DAE_10_abliation_11",
]
models_paths["upscale-only"] = "upscale-only"
print(models_paths)
ds = "/media/data/robert/test_paper4/"
if Path("/DATA/NAS/ongoing_projects/robert/paper/paper4/").exists():
    ds = "/DATA/NAS/ongoing_projects/robert/paper/paper4/test_paper4"
summary = {}
device = torch.device("cuda:0")
args = sys.argv

for max_limit in [45, 50, 60, 70, 80, 90, 5000]:
    if len(args) == 1:
        it = models_paths.items()
    elif args[1] == "0":
        it = reversed(models_paths.items())
    else:
        print(args)
        j = int(args[1])
        it = reversed([k for i, k in enumerate(models_paths.items()) if (i + j // 2) % j == 0])
    for model_name, model_path in it:
        try:
            limit = max_limit
            if model_path is None:
                continue
            reset_model = True
            parent = f"eval_{model_name}"
            if parent in skip_list:
                print(parent, " skipped. (In skip list)")
                continue
            if "old" not in parent and "RCAN" not in parent and "ESRGAN" not in parent:
                continue

            # Pandas dict
            pd_dict = {"vertebra_hit_rate": [], "b0_er": [], "b1_er": [], "b2_er": []}

            for ses in get_sessions(ds):
                limit -= 1
                if limit == 0:
                    break
                make_fast_preprocessing(ses)
                fam = ses.get_iso_stiched_seg()
                vert = fam["msk_seg-vert"][0]
                post = fam["msk_seg-postprocessing"][0].open_nii_reorient()
                gt_nii = vert.open_nii_reorient().clamp(max=59).remove_labels_(59)
                # Translate
                if model_name.lower() == "upscale-only":
                    out_path = ses.axial_stiched.get_changed_bids(info={"acq": "iso", "desc": "superres"}, parent=parent)
                    T2w_iso = out_path
                    if not out_path.exists():
                        nii_org = ses.axial_stiched.open_nii()
                        nii = nii_org.reorient().rescale_((0.8571, 0.8571, 0.8571))
                        nii.save(out_path)
                else:
                    print(ses.axial_stiched)
                    upscale_nii(
                        ses.axial_stiched,
                        parent=parent,
                        reset_model=reset_model,
                        checkpoint_sag=model_path,
                        batch_size=32 if "conv" in model_name.lower() else 48,
                        device=device,
                    )
                    T2w_iso = ses.axial_stiched.get_changed_bids(info={"acq": "iso", "desc": "superres"}, parent=parent)
                    reset_model = False
                # Segment
                from mri_segmentor.seg_run import process_img_nii

                a = partial(filter_segmentation, mask=post)
                output_paths, errcode = process_img_nii(
                    img_ref=T2w_iso,
                    derivative_name=parent,
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
                if "vertebra_hit_rate" in poi.info and "betti" in poi.info:
                    pass
                else:
                    labels = gt_nii.unique()
                    checklist = {label: 0 for label in labels}
                    valid_reg = []
                    merges = 0
                    for reg, _, (x, y, z) in poi.items():
                        idx = gt_nii.get_array()[int(round(x)), int(round(y)), int(round(z))]  # type: ignore
                        if idx in checklist:
                            if checklist[idx] == 1:
                                merges += 1
                            checklist[idx] = 1
                            valid_reg.append(reg)
                    poi.info["vertebra_hit_rate"] = 1.0 * sum(checklist.values()) / len(checklist)
                    poi.info["vertebra_hit_count"] = sum(checklist.values())
                    poi.info["vertebra_missed_count"] = len(checklist) - sum(checklist.values())
                    # Measure Betti error rate
                    ### add missing vertebra as error
                    ### go over all valid betti numbers
                    poi.info["betti"] = vert_nii.clamp(max=59).remove_labels_(59).betti_numbers()
                    b0 = len(checklist) - sum(checklist.values())
                    b1 = b0
                    b2 = b0
                    betti_total = b0
                    for region in valid_reg:
                        betti = poi.info["betti"][region]
                        if betti[0] != 1:
                            b0 += 1
                        if (betti[1] != 1 and region > 7) or (betti[1] not in [1, 2, 3] and region <= 7):
                            b1 += 1
                        if betti[2] != 0:
                            b2 += 1
                        betti_total += 1
                        # print(b0, b1, b2, betti_total, betti)
                    poi.info["betti_0_er"] = b0 / betti_total
                    poi.info["betti_1_er"] = b1 / betti_total
                    poi.info["betti_2_er"] = b2 / betti_total
                    poi.info["betti_total"] = betti_total
                    poi.save(output_paths["out_ctd"])
                pd_dict["vertebra_hit_rate"].append(poi.info["vertebra_hit_rate"])
                pd_dict["b0_er"].append(poi.info["betti_0_er"])
                pd_dict["b1_er"].append(poi.info["betti_1_er"])
                pd_dict["b2_er"].append(poi.info["betti_2_er"])
                summary[model_name] = {}
                for k, v in pd_dict.items():
                    summary[model_name][k] = round(sum(v) / max(1, len(v)), 4)
                    summary[model_name]["len"] = len(v)
                    print(k, summary[model_name][k], len(v))

            print("############################################################")
            print("######################## summary ###########################")
            for i in sorted(summary.items(), key=lambda x: -x[1]["vertebra_hit_rate"]):
                print(i[0], ":", i[1])
            print("############################################################")
            print("############################################################")

            p = Path(ds, parent, "betti.xlsx")
            p.parent.mkdir(exist_ok=True)
            pd.DataFrame(pd_dict).to_excel(p)
        except Exception:
            No_Logger().print_error()
# python train_DAE.py --config config/neuropoly/super_ax/03_abliation_cubic.conf
# /home/robert/anaconda3/envs/mri_spine_seg/bin/python /media/data/robert/code/dae/test_scripts.py
# python train_DAE.py --config config/neuropoly/super_ax/04_abliation_cubic.conf
# /home/robert/anaconda3/envs/mri_spine_seg/bin/python /media/data/robert/code/dae/test_scripts.py
