import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import fastnumpyio
import numpy as np
import pandas as pd
from BIDS import BIDS_FILE, NII, BIDS_Family, BIDS_Global_info, No_Logger
from tqdm import tqdm


def run_vibe(
    callback: Callable[[BIDS_Family], None],
    in_ds: Path = Path("/DATA/NAS/datasets_processed/NAKO/dataset-nako"),
    raw="rawdata",
    der="derivatives",
    block="",  # "105",
    fuse_chunk=False,
    stitched=False,
):
    # INPUT
    in_ds = Path(in_ds)
    head_logger = No_Logger()  # (in_ds, log_filename="source-convert-to-unet-train", default_verbose=True)
    parent_raw = str(Path(raw).joinpath(str(block)))
    parent_der = str(Path(der).joinpath(str(block)))
    if fuse_chunk:
        BIDS_Global_info.remove_splitting_key("chunk")
    bids_ds = BIDS_Global_info(datasets=[in_ds], parents=[parent_raw, parent_der], verbose=False)

    for name, subject in bids_ds.enumerate_subjects(sort=True):
        logger = head_logger.add_sub_logger(name=name)
        q = subject.new_query()
        q.flatten()
        q.filter("part", "outphase", required=False)
        q.filter("acq", "ax")
        q.filter("seg", lambda x: x != "manual", required=False)
        q.filter("lesions", lambda x: x != "manual", required=False)
        q.filter("desc", lambda _: False, required=False)
        if stitched:
            q.filter("sequ", "stitched")
            q.filter("chunk", lambda _: False, required=False)
        q.unflatten()
        q.filter_format("vibe")
        q.filter_filetype("nii.gz")
        families = q.loop_dict(sort=True)
        for f in families:
            try:
                callback(f)
            except Exception:
                logger.print_error()


def get_split():
    filename = Path("/DATA/NAS/datasets_processed/NAKO/notes/nako_split.xlsx")
    if filename.exists():
        return pd.read_excel(filename)

    random.seed(3069304830)

    out = {"name": [], "split_continuous": [], "split": [], "path_vibe": []}

    def add_fun(fam: BIDS_Family):
        # print(fam["vibe"][0].open_nii().zoom)
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
        s = ",".join([str(f.file["nii.gz"]) for f in vibes])
        out["path_vibe"].append(s)

    run_vibe(add_fun, fuse_chunk=True, block="105")
    df_ = pd.DataFrame(out)
    df_.to_excel(filename)
    return df_


class ROW:
    name: str
    split_continuous: float
    split: Literal["train", "val", "test"]
    path_vibe: str


def random_crop_vibe():
    out = {"file_path": [], "split": []}
    out_path = Path("/DATA/NAS/ongoing_projects/robert/train_dataset/3D/nako-vibe/")

    # batch_size:  192 x 192 x 64
    for _, row in tqdm(get_split().iterrows(), total=len(get_split())):  # type: ignore
        try:
            row: ROW
            r = row.path_vibe.split(",")
            vibe_path = r[random.randint(0, len(r) - 1)]
            bf = BIDS_FILE(vibe_path, vibe_path.split("rawdata")[0])
            nii = NII.load(vibe_path, False).reorient_(("L", "A", "S"))
            c1, c2, c3 = nii.compute_crop_slice(minimum=20, dist=3)
            if c1.stop - c1.start > 192:
                c = random.randint(c1.start, c1.stop - 192)
                c1 = slice(c, c + 192)
            if c2.stop - c2.start > 192:
                c = random.randint(c2.start, c2.stop - 192)
                c2 = slice(c, c + 192)
            if c3.stop - c3.start > 64:
                c = random.randint(c3.start, c3.stop - 64)
                c3 = slice(c, c + 64)
            nii.apply_crop_slice_((c1, c2, c3))
            # nii.save("/DATA/NAS/datasets_processed/NAKO/dataset-nako/test.nii.gz")
            out_path2 = out_path / f"{bf.get('sub')[:4]}"
            out_path2.mkdir(exist_ok=True)
            file_path = out_path2 / f"{bf.get('sub')}-{bf.get('chunk')}.fnio"
            fastnumpyio.save(file_path, nii.get_array().astype(np.int16))
            out["file_path"].append(str(file_path))
            out["split"].append(row.split)
        except Exception:
            No_Logger().print_error()

    df_ = pd.DataFrame(out)
    df_.to_excel(out_path.parent / "nako-vibe.xlsx")


def random_crop_ct():
    from reorder_total_seg import get_all_files

    out = {"file_path": [], "Split": []}
    out_path = Path("/DATA/NAS/ongoing_projects/robert/train_dataset/3D/total-ct/")
    # out_path2 = Path("/DATA/NAS/ongoing_projects/robert/train_dataset/3D/total-seg/")

    # batch_size:  192 x 192 x 64

    for split in ["train", "val", "test"]:
        cts, segs = get_all_files(split=split)
        for vibe_path in tqdm(cts, total=len(cts)):  # type: ignore
            vibe_path = Path(vibe_path)
            try:
                out_path2 = out_path / vibe_path.parent.parent.name
                file_path = out_path2 / f"{vibe_path.name}.fnio"
                if not file_path.exists():
                    nii = NII.load(vibe_path, False).reorient_(("L", "A", "S")).rescale((-1, -1, 3))
                    c1, c2, c3 = nii.compute_crop_slice(minimum=-900, dist=3)
                    if c1.stop - c1.start > 192:
                        c = random.randint(c1.start, c1.stop - 192)
                        c1 = slice(c, c + 192)
                    if c2.stop - c2.start > 192:
                        c = random.randint(c2.start, c2.stop - 192)
                        c2 = slice(c, c + 192)
                    if c3.stop - c3.start > 64:
                        c = random.randint(c3.start, c3.stop - 64)
                        c3 = slice(c, c + 64)
                    nii.apply_crop_slice_((c1, c2, c3))
                    # nii.save("/DATA/NAS/datasets_processed/NAKO/dataset-nako/test.nii.gz")
                    out_path2.mkdir(exist_ok=True, parents=True)

                    fastnumpyio.save(file_path, nii.get_array().astype(np.float16))
                out["file_path"].append(str(file_path))
                out["Split"].append(split)
            except Exception:
                No_Logger().print_error()
    df_ = pd.DataFrame(out)
    df_.to_excel(out_path.parent / "total-ct.xlsx")


if __name__ == "__main__":
    # get_split()
    # random_crop_vibe()
    random_crop_ct()
