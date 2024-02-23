import random
from pathlib import Path

import numpy as np
import pandas as pd
from BIDS import BIDS_Global_info, to_nii
from BIDS.core.np_utils import np_dice
from dataset_generation import make_nii_to_slice
from numpy import ndarray
from tqdm import tqdm


def generate_paired(
    ds="/media/data/robert/datasets/dataset-neuropoly/",
    parents=("registrate", "derivatives_ax", "derivatives_seg"),
    dice_mode=True,
    sort=True,
):
    bgi = BIDS_Global_info([ds], parents)
    out_dict = {"name": [], "dice": [], "mean_dist": []}
    out_dict_2 = {"subject": [], "dice": [], "mean_dist": [], "Split": [], "file_path": [], "zoom": []}
    for _, sequ in tqdm(bgi.iter_subjects(sort=sort)):
        split_sub = "train"
        if random.random() <= 0.1:  # noqa: PLR2004
            split_sub = "test"
        q = sequ.new_query()
        q.filter("parent", "registrate")
        q2 = sequ.new_query()
        q2.filter("parent", "derivatives_seg")
        q2.filter("desc", "stitched")
        q2.filter_format("ctd")

        q3 = sequ.new_query()
        q3.filter("parent", "derivatives_ax")
        q3.filter_format("ctd")

        chunks_sag = {s["ctd_seg-spine"][0].get("ses"): s["ctd_seg-spine"][0] for s in q2.loop_dict()}
        chunks_ax = {
            f'{v["ctd_seg-spine"][0].get("ses")}_{v["ctd_seg-spine"][0].get("chunk")}': v["ctd_seg-spine"][0] for v in list(q3.loop_dict())
        }
        for fam in q.loop_dict(key_addendum=["desc"]):
            split = split_sub
            ax = fam["T2w_desc-registrate-ax"][0]
            sag = fam["T2w_desc-registrate-sag"][0]
            spine_ax = fam["msk_seg-spine_desc-registrate-ax"][0]
            spine_sag = fam["msk_seg-spine_desc-registrate-sag"][0]
            old_poi_sag = chunks_sag[f'{ax.get("ses")}'].open_ctd().to_global()
            old_poi_ax_loc = chunks_ax[f'{ax.get("ses")}_{ax.get("chunk")}'].open_ctd().filter_points_inside_shape()
            old_poi_ax = old_poi_ax_loc.to_global()
            dist = old_poi_sag.calculate_distances_poi(old_poi_ax)
            mean_dist = np.mean(np.array(list(dist.values()))) if len(dist) != 0 else 10**6

            # dict_keys(['msk_seg-vert_desc-registrate-ax', 'T2w_desc-registrate-ax', 'T2w_desc-registrate-sag', 'snp_desc-registrate-ax',
            # 'msk_seg-spine_desc-registrate-ax', 'msk_seg-vert_desc-registrate-sag', 'msk_seg-spine_desc-registrate-sag'])
            spine_ax = to_nii(spine_ax).extract_label([60, 61])
            spine_sag_cord = to_nii(spine_sag).extract_label([60, 61])
            out_dict["name"].append(fam.family_id)
            dice = np_dice(spine_ax.get_array(), spine_sag_cord.get_array())
            out_dict["dice"].append(dice)
            out_dict["mean_dist"].append(mean_dist)
            if dice_mode:
                continue
            if mean_dist > 10:  # noqa: PLR2004
                split = "val"
                continue
            orientation = ("R", "I", "P")
            # crop = (slice(0, -1), slice(None), slice(None))
            zoom = (-1, -1, -1)
            _ax = ax.open_nii().reorient_(orientation)
            if _ax.zoom[-1] <= 0.5:  # noqa: PLR2004
                zoom = (-1, _ax.zoom[-1] * 2, _ax.zoom[-1] * 2)
            niis = {
                "ax": _ax.rescale_(zoom).normalize(),
                "sag": sag.open_nii().reorient_(orientation).rescale_(zoom).normalize(),
                "seg": spine_sag.open_nii().reorient_(orientation).rescale_(zoom),
                # "seg": to_nii(spine_sag).reorient_(orientation).apply_crop_(crop).normalize(),
            }

            def filter_slice(c: dict[str, ndarray]) -> bool:
                return c["sag"].sum() > 3000  # noqa: PLR2004

            files = make_nii_to_slice(
                False,
                niis,
                Path(f"/media/data/robert/datasets/dataset-neuropoly/training_img/paired/{split}"),
                prefix=fam.family_id,
                deform=False,
                filter_slice=filter_slice,
                # sub_folders={"ax": "ax", "sag": "sag"},
            )

            # niis["seg"] = to_nii(spine_sag).reorient_(orientation).normalize()
            #
            # make_nii_to_slice(
            #    True,
            #    niis,
            #    Path(f"/media/data/robert/datasets/dataset-neuropoly/training_img/paired/{split}"),
            #    prefix=fam.family_id,
            #    deform=False,
            #    filter_slice=filter_slice,
            #    # sub_folders={"ax": "ax", "sag": "sag"},
            # )

            assert files is not None
            for file in files:
                out_dict_2["subject"].append(fam.family_id)
                out_dict_2["dice"].append(dice)
                out_dict_2["Split"].append(split)
                out_dict_2["file_path"].append(file)
                out_dict_2["mean_dist"].append(mean_dist)
                out_dict_2["zoom"].append(str(niis["ax"].zoom))
            # break
        if len(out_dict["mean_dist"]) == 50:  # noqa: PLR2004
            break

    df_reg = pd.DataFrame(out_dict)
    df_reg.to_excel("/media/data/robert/datasets/dataset-neuropoly/code/registration.xlsx")
    df_reg = pd.DataFrame(out_dict_2)
    df_reg.to_excel("/media/data/robert/datasets/dataset-neuropoly/training_img/paired/train.xlsx")


def generate_paired_3D(
    ds="/media/data/robert/datasets/dataset-neuropoly/",
    parents=("registrate", "derivatives_ax", "derivatives_seg"),
    dice_mode=True,
    sort=True,
):
    bgi = BIDS_Global_info([ds], parents)
    out_dict = {"name": [], "dice": [], "mean_dist": [], "Split": [], "hq": [], "lq": []}
    j = 0
    for _, sequ in tqdm(bgi.iter_subjects(sort=sort)):
        split_sub = "train"
        if random.random() <= 0.05:  # noqa: PLR2004
            split_sub = "val"
        if random.random() <= 0.1:  # noqa: PLR2004
            split_sub = "test"
        q = sequ.new_query()
        q.filter("parent", "registrate")
        q2 = sequ.new_query()
        q2.filter("parent", "derivatives_seg")
        q2.filter("desc", "stitched")
        q2.filter_format("ctd")

        q3 = sequ.new_query()
        q3.filter("parent", "derivatives_ax")
        q3.filter_format("ctd")

        chunks_sag = {s["ctd_seg-spine"][0].get("ses"): s["ctd_seg-spine"][0] for s in q2.loop_dict()}
        chunks_ax = {
            f'{v["ctd_seg-spine"][0].get("ses")}_{v["ctd_seg-spine"][0].get("chunk")}': v["ctd_seg-spine"][0] for v in list(q3.loop_dict())
        }
        for fam in q.loop_dict(key_addendum=["desc"]):
            split = split_sub
            ax = fam["T2w_desc-registrate-ax"][0]
            sag = fam["T2w_desc-registrate-sag"][0]
            spine_ax = fam["msk_seg-spine_desc-registrate-ax"][0]
            spine_sag = fam["msk_seg-spine_desc-registrate-sag"][0]
            old_poi_sag = chunks_sag[f'{ax.get("ses")}'].open_ctd().to_global()
            old_poi_ax_loc = chunks_ax[f'{ax.get("ses")}_{ax.get("chunk")}'].open_ctd().filter_points_inside_shape()
            old_poi_ax = old_poi_ax_loc.to_global()
            dist = old_poi_sag.calculate_distances_poi(old_poi_ax)
            mean_dist = int(np.mean(np.array(list(dist.values())))) if len(dist) != 0 else 10**6

            # dict_keys(['msk_seg-vert_desc-registrate-ax', 'T2w_desc-registrate-ax', 'T2w_desc-registrate-sag', 'snp_desc-registrate-ax',
            # 'msk_seg-spine_desc-registrate-ax', 'msk_seg-vert_desc-registrate-sag', 'msk_seg-spine_desc-registrate-sag'])
            spine_ax = to_nii(spine_ax).extract_label([60, 61])
            spine_sag_cord = to_nii(spine_sag).extract_label([60, 61])
            dice = np_dice(spine_ax.get_array(), spine_sag_cord.get_array())
            if dice_mode:
                continue
            if mean_dist > 10:  # noqa: PLR2004
                continue

            orientation = ("R", "I", "P")
            crop = (slice(1, -1), slice(1, -1), slice(1, -1))
            zoom = (0.8571, 0.8571, 0.8571)
            ax_out = ax.open_nii().reorient_(orientation).rescale_(zoom).normalize().apply_crop_(crop)
            sag_out = sag.open_nii().reorient_(orientation).rescale_(zoom).normalize().apply_crop_(crop)
            folder = Path(f"/media/data/robert/datasets/dataset-neuropoly/training_img/paired3D/{split}/{j//100}")
            folder.mkdir(exist_ok=True, parents=True)
            ax_path = Path(folder, f"{fam.family_id}_ax.npy")
            sag_path = Path(folder, f"{fam.family_id}_sag.npy")
            np.save(ax_path, ax_out.get_array())
            np.save(sag_path, sag_out.get_array())
            # ax_out.save(ax_path)
            # sag_out.save(sag_path)
            out_dict["name"].append(fam.family_id)
            out_dict["dice"].append(dice)
            out_dict["mean_dist"].append(mean_dist)
            out_dict["Split"].append(split)
            out_dict["hq"].append(str(sag_path))
            out_dict["lq"].append(str(ax_path))

            j += 1

    df_reg = pd.DataFrame(out_dict)
    df_reg.to_excel("/media/data/robert/datasets/dataset-neuropoly/training_img/paired3D/train.xlsx")


if __name__ == "__main__":
    # generate_paired(dice_mode=False)
    generate_paired_3D(dice_mode=False)
