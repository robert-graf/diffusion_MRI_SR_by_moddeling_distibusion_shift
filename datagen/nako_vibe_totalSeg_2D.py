import gc
import os
import random
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from BIDS import NII, BIDS_Family
from BIDS.logger.log_file import Log_Type, Reflection_Logger
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).parent.parent))
from datagen.dataset_generation import make_nii_to_slice
from datagen.nako_vibe_totalSeg_3D import run_vibe
from datagen.reorder_total_seg import get_all_files

random.seed(6456)
logger = Reflection_Logger()
logger.override_prefix = "Data GEN"


def make_axial_total_seg(
    acq: Literal["total", "vibe"] = "total",
    orientation=("I", "P", "R"),  # first will be cut
    zoom=None,
    crop3d=True,
    name_folder=None,
    out="/DATA/NAS/ongoing_projects/robert/train_dataset/totalSeg2/",
):
    if name_folder is None:
        name_folder = acq
    if acq == "total":
        files = {a: list(iter_total_seg(a)) for a in ("train", "test", "val")}
    elif acq == "vibe":
        files = {"train": [], "test": [], "val": []}
        for _, i in get_split_vibe().iterrows():
            try:
                v = {"vibe": i.path_vibe, "body_comp": i["bodycomp"]}
                files[i.split].append(v)
            except Exception:
                print(i.path_vibe)
                continue
    else:
        raise NotImplementedError()
    print(files)

    def slice_total_seg_(i, nii_path, prefix_, phase, zoom, name_folder):
        gc.collect()
        try:
            mr = NII.load(nii_path["CT" if acq == "total" else "vibe"], False).reorient_(orientation)
            seg = NII.load(nii_path["body_comp"], True).reorient_(orientation)
            if zoom is not None:
                mr.rescale_(zoom)
                seg.rescale_(zoom)
            if mr.shape != seg.shape:
                return
        except EOFError:
            return
        except FileNotFoundError:
            return
        mr = mr.normalize(clamp_lower=-1024) if acq.lower() == "total" else mr.normalize(clamp_lower=0)
        seg *= 75
        prefix = f"{prefix_}_{i:05}"
        out_path = Path(out, phase, name_folder, f"{prefix}_{name_folder}_{mr.shape[0]-10}.png")

        if out_path.exists():
            logger.print("Skip", out_path, "exists")
            return
        logger.print("Start", i, type=Log_Type.ITALICS)

        niis = {prefix_: mr, prefix_ + "_seg": seg}
        zoom = mr.zoom
        sub_folders = {prefix_: prefix_, prefix_ + "_seg": prefix_ + "_seg"}
        make_nii_to_slice(
            True,
            niis,
            out_path=Path(out, phase),
            prefix=prefix,
            deform=False,
            single_png=False,
            crop3D=prefix_ if crop3d else None,
            sub_folders=sub_folders,
        )
        logger.print(i, mr.zoom, zoom, f"finished{' ':90}", type=Log_Type.SAVE)
        gc.collect()

    tasks = []
    a = os.cpu_count() // 2
    print()
    print()
    for trainings_phase, list_nii in files.items():
        for i, ct_path in enumerate(list_nii):
            tasks.append(delayed(slice_total_seg_)(i, ct_path, name_folder, trainings_phase, zoom, name_folder))
    print("Start n =", len(tasks))
    Parallel(n_jobs=a)(tasks)


def iter_total_seg(split: Literal["train", "val", "test"]):
    root = "/DATA/NAS/datasets_processed/CT_fullbody/dataset-CACTS/ct_ready_1mm/"
    expect_find_all = True
    out_folder = "/DATA/NAS/datasets_processed/CT_fullbody/dataset-CACTS/reduced"

    if not Path(root).exists():
        root = "D:/data/totalseg/ct_ready_1mm"
        out_folder = "D:/data/totalseg/ct_ready_1mm/reduced"

        expect_find_all = False
    files, files_seg = get_all_files(split=split, root=root, expect_find_all=expect_find_all)

    for i, (ct_path, seg_path_root) in enumerate(zip(files, files_seg, strict=True)):
        print(i, ct_path)
        try:
            ct_path = Path(ct_path)
            out_path = Path(out_folder, *list(Path(ct_path).parts[-3:]))
            out_name = out_path.name
            out_path = out_path.parent
            out_final = Path(out_path, out_name.replace("0000.nii.gz", "all.nii.gz"))
            out_body_comp = Path(out_path, out_name.replace("0000.nii.gz", "body-comp.nii.gz"))
            out_reduced = Path(out_path, out_name.replace("0000.nii.gz", "reduced.nii.gz"))
            yield {"CT": ct_path, "all": out_final, "body_comp": out_body_comp, "reduced": out_reduced, "seg_path_root": seg_path_root}
        except Exception as e:
            print(e)


def get_split_vibe():
    filename = Path("/DATA/NAS/datasets_processed/NAKO/notes/nako_split_10k.xlsx")
    if filename.exists():
        return pd.read_excel(filename)

    random.seed(3069304830)

    out = {"name": [], "split_continuous": [], "split": [], "path_vibe": [], "bodycomp": []}

    def add_fun(fam: BIDS_Family):
        # print(fam["vibe"][0].open_nii().zoom)
        print(fam)
        vibes = fam["vibe"]

        out["name"].append(vibes[0].get("sub"))
        r = random.random()
        out["split_continuous"].append(r)
        if r <= 0.8:
            s = "train"
        elif r <= 0.9:
            s = "val"
        else:
            s = "test"
        out["split"].append(s)
        out["bodycomp"].append(str(fam["msk_seg-body-composition"][0].file["nii.gz"]))
        s = ",".join([str(f.file["nii.gz"]) for f in vibes])
        out["path_vibe"].append(s)

    # run_vibe(add_fun, in_ds=Path("D:/data"), raw="dataset-bodycomp", fuse_chunk=True, stitched=True)
    run_vibe(
        add_fun,
        in_ds=Path("/DATA/NAS/datasets_processed/NAKO/MRT"),
        der="derivatives_Abdominal-Segmentation",
        fuse_chunk=False,
        stitched=True,
    )
    df_ = pd.DataFrame(out)
    try:
        df_.to_excel(filename)
    except Exception:
        pass
    return df_


if __name__ == "__main__":
    # make_axial_total_seg()
    make_axial_total_seg("vibe")
