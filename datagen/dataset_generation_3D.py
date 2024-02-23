import random
from pathlib import Path

import fastnumpyio
import numpy as np
import pandas as pd
from BIDS import BIDS_Global_info, to_nii
from BIDS.core.np_utils import np_dice
from tqdm import tqdm


def generate_paired_3D(
    ds="/media/data/robert/datasets/dataset-neuropoly/",
    parents=("registrate_2", "derivatives_ax", "derivatives_seg"),
    dice_mode=True,
    sort=True,
):
    bgi = BIDS_Global_info([ds], parents)
    out_dict = {"name": [], "dice": [], "mean_dist": [], "Split": [], "hq": [], "lq": []}
    out_dict = {"train": out_dict.copy(), "val": out_dict.copy(), "test": out_dict.copy()}
    j = 0
    for _, sequ in tqdm(bgi.iter_subjects(sort=sort)):
        split_sub = "train"
        if random.random() <= 0.05:
            split_sub = "val"
        if random.random() <= 0.1:
            split_sub = "test"
        q = sequ.new_query()
        q.filter("parent", "registrate_2")
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
            if mean_dist > 10:
                continue

            orientation = ("R", "I", "P")
            zoom = (0.8571, 0.8571, 0.8571)
            ax_out = ax.open_nii().reorient_(orientation).rescale_(zoom).normalize()
            sag_out = sag.open_nii().reorient_(orientation).rescale_(zoom).normalize()
            if ax_out.shape != sag_out.shape:
                continue
            folder = Path(f"/media/data/robert/datasets/dataset-neuropoly/training_img/paired3D_fnio/{split}")
            folder.mkdir(exist_ok=True, parents=True)
            ax_path = Path(folder, "lq", f"{j // 50}", f"{fam.family_id}_ax.fnio")
            sag_path = Path(folder, "hq", f"{j // 50}", f"{fam.family_id}_sag.fnio")
            ax_path.parent.mkdir(exist_ok=True, parents=True)
            sag_path.parent.mkdir(exist_ok=True, parents=True)

            fastnumpyio.save(ax_path, ax_out.get_array())
            fastnumpyio.save(sag_path, sag_out.get_array())

            # ax_out.save(ax_path)
            # sag_out.save(sag_path)
            out_dict[split]["name"].append(fam.family_id)
            out_dict[split]["dice"].append(dice)
            out_dict[split]["mean_dist"].append(mean_dist)
            out_dict[split]["Split"].append(split)
            out_dict[split]["hq"].append(str(sag_path))
            out_dict[split]["lq"].append(str(ax_path))
            j += 1

    df_reg = pd.DataFrame({**out_dict["train"], **out_dict["test"], **out_dict["val"]})
    df_reg.to_excel("/media/data/robert/datasets/dataset-neuropoly/training_img/paired3D_fnio/train.xlsx")
    a = ["hq", "lq"]
    for split, dict_ in out_dict.items():
        for image_type in a:
            d = dict_.copy()
            d["file_path"] = d[image_type]
            df_reg = pd.DataFrame(d)
            print(df_reg)
            df_reg = df_reg.drop(a, axis=1)
            print(a)
            df_reg.to_excel(f"/media/data/robert/datasets/dataset-neuropoly/training_img/paired3D_fnio/{split}/_{image_type}_train.xlsx")


if __name__ == "__main__":
    # generate_paired(dice_mode=False)
    generate_paired_3D(dice_mode=False)
