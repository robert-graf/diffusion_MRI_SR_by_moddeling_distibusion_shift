import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from mri_segmentor.seg_run import ErrCode, process_img_nii
from TPTBox import BIDS_FILE, NII, POI, BIDS_Global_info, Log_Type, No_Logger, calc_centroids_from_subreg_vert
from TPTBox.registration.ridged_points import ridged_points_from_poi
from TPTBox.snapshot2D import Snapshot_Frame, create_snapshot
from TPTBox.stitching.stitching_tools import stitching

# from pl_models.cycleGAN import CycleGAN_LitModel
from pl_models.DEA import DAE_LitModel

model: DAE_LitModel = None  # type: ignore
model_ax: DAE_LitModel = None  # type: ignore


def run_ax(in_ds: Path, raw="rawdata", der="derivatives_seg", out_par="rawdata_upscale", stitching_only=False, sort=True):
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
        q.filter("acq", ["sag", "iso"])
        q.filter("seg", lambda x: x != "manual", required=False)
        q.filter("lesions", lambda x: x != "manual", required=False)
        q.filter("desc", lambda _: False, required=False)
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
                files = [f for f in f["T2w"] if "stitched" not in str(f.file["nii.gz"])]
                stitched = f["T2w"][0].get_changed_path(info={"desc": "stitched", "chunk": None}, parent=out_par, make_parent=True)
                if not stitched.exists():
                    stitching(*files, out=stitched, verbose_stitching=True, bias_field=True)

                if stitching_only:
                    continue
                start_time = time.perf_counter()
                ref_t2w: BIDS_FILE = BIDS_FILE(stitched, f["T2w"][0].dataset)
                logger.print("Upscale ", fid)
                # Call to the pipeline
                output_paths, errcode = process_img_nii(
                    img_ref=BIDS_FILE(stitched, ref_t2w.dataset),
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

            except Exception:
                logger.print_error()

    if len(execution_times) > 0:
        head_logger.print(
            f"\nExecution times:\n{execution_times}\nRange:{min(execution_times)}, {max(execution_times)}\nAvg {np.average(execution_times)}"
        )


def register_ax_sag(axial_img: BIDS_FILE, sagittal_img: BIDS_FILE, log: No_Logger):
    chunk = axial_img.get("chunk")
    axial_out = axial_img.get_changed_bids(
        info={"acq": None, "desc": "registrate-ax", "res": f"sag-{chunk}"},
        parent="registrate",
        make_parent=False,
        additional_folder=f"{chunk}",
    )
    snap = axial_out.dataset / "registrate/snapshots" / str(axial_out.file["nii.gz"].name).replace("_T2w.nii.gz", "_snp.jpg")
    snap1 = str(axial_out.file["nii.gz"]).replace("_T2w.nii.gz", "_snp.jpg")
    if snap.exists():
        log.print(axial_out, "exists. Skip", Log_Type.OK)
        return
    # def to_poi_file(p: Path | str):
    #    return Path(str(p).replace("rawdata", "derivatives").replace("_seg-vert_msk.nii.gz", "_seg-spine_ctd.json"))
    fam = axial_img.get_sequence_files(key_addendum=["acq", "desc", "chunk"])
    ## LOADING ###
    try:
        img_sag_nii = sagittal_img.open_nii()
        subreg_sag_nii = fam["msk_seg-spine_acq-sag_desc-stitched"][0].open_nii()
        vert_sag_nii = fam["msk_seg-vert_acq-sag_desc-stitched"][0].open_nii()
        poi_sag = fam["ctd_seg-spine_acq-sag_desc-stitched"][0]
        img_ax_nii = axial_img.open_nii()
        subreg_ax_nii = fam[f"msk_seg-spine_acq-ax_chunk-{chunk}"][0].open_nii()
        vert_ax_nii = fam[f"msk_seg-vert_acq-ax_chunk-{chunk}"][0].open_nii()
        poi_ax = fam[f"ctd_seg-spine_acq-ax_chunk-{chunk}"][0]
    except FileNotFoundError as e:
        log.print("FileNotFoundError:", e.filename, Log_Type.FAIL)
        return
    if len(vert_ax_nii.unique()) <= 4:
        No_Logger().print("SKIP. Not enough vertebra", type=Log_Type.NEUTRAL)
        return

    assert img_ax_nii.zoom == vert_ax_nii.zoom, (img_ax_nii, vert_ax_nii, subreg_ax_nii)
    assert img_ax_nii.zoom == subreg_ax_nii.zoom, (img_ax_nii, vert_ax_nii, subreg_ax_nii)
    assert img_ax_nii.shape == vert_ax_nii.shape, (img_ax_nii, vert_ax_nii, subreg_ax_nii)
    assert img_ax_nii.shape == subreg_ax_nii.shape, (img_ax_nii, vert_ax_nii, subreg_ax_nii)
    poi_file = None
    print(poi_ax)
    try:
        poi = POI.load(poi_sag)
        if poi.origin is None:
            poi_sag = None
        else:
            poi_file = poi_sag.file["json"]

    except FileNotFoundError:
        pass
    subreg_id = [50, 61]

    ax_poi = calc_centroids_from_subreg_vert(vert_ax_nii, subreg_ax_nii, subreg_id=subreg_id, buffer_file=poi_ax.file["json"])
    sag_poi = calc_centroids_from_subreg_vert(vert_sag_nii, subreg_sag_nii, subreg_id=subreg_id, buffer_file=poi_file)
    sag_poi.shape = img_sag_nii.shape

    #### CALC resample filter ###
    resample_filter = ridged_points_from_poi(sag_poi, ax_poi, c_val=0)

    ## target res ##
    fixed_img = img_sag_nii  # .reorient(orientation).rescale_(zoom)
    fixed_vert = vert_sag_nii  # .reorient(orientation).rescale_(zoom)
    fixed_subreg = subreg_sag_nii
    fixed_poi = sag_poi
    ## registrate ax to sag
    moved_img: NII = resample_filter.transform_nii(img_ax_nii)
    moved_vert: NII = resample_filter.transform_nii(vert_ax_nii)
    moved_subreg: NII = resample_filter.transform_nii(subreg_ax_nii)
    moved_poi = resample_filter.transform_poi(ax_poi.extract_subregion(*subreg_id))
    ## calc crop ##
    assert moved_img.shape == fixed_img.shape, (moved_img, fixed_img)
    crop = moved_img.compute_crop()
    crop = fixed_img.compute_crop(other_crop=crop)
    fixed_img.apply_crop_(crop)
    fixed_vert.apply_crop_(crop)
    fixed_subreg.apply_crop_(crop)
    fixed_poi.apply_crop_(crop)
    moved_img.apply_crop_(crop)
    moved_vert.apply_crop_(crop)
    moved_subreg.apply_crop_(crop)
    moved_poi.apply_crop_(crop)
    sagittal_out = axial_img.get_changed_bids(
        info={"acq": None, "desc": "registrate-sag", "res": f"sag-{chunk}"},
        parent="registrate",
        make_parent=True,
        additional_folder=f"{chunk}",
    )

    moved_img.save(axial_out)
    moved_vert.save(axial_out.get_changed_path(parent="registrate", format="msk", info={"seg": "vert"}))
    moved_subreg.save(axial_out.get_changed_path(parent="registrate", format="msk", info={"seg": "spine"}))
    fixed_vert.save(sagittal_out.get_changed_path(parent="registrate", format="msk", info={"seg": "vert"}))
    fixed_subreg.save(sagittal_out.get_changed_path(parent="registrate", format="msk", info={"seg": "spine"}))
    fixed_img.save(sagittal_out)
    frame_1 = Snapshot_Frame(moved_img, segmentation=moved_subreg, centroids=moved_poi)
    frame_2 = Snapshot_Frame(moved_img, segmentation=fixed_subreg, centroids=fixed_poi)
    frame_3 = Snapshot_Frame(fixed_img, segmentation=moved_subreg, centroids=moved_poi)
    frame_4 = Snapshot_Frame(fixed_img, segmentation=fixed_subreg, centroids=fixed_poi)

    create_snapshot([snap, snap1], [frame_1, frame_2, frame_3, frame_4])


def run_registration(
    in_ds: Path, raw="rawdata", der="derivatives_seg", out_par="rawdata_upscale", sort=True, register_call_back=register_ax_sag
):
    # INPUT
    in_ds = Path(in_ds)
    head_logger = No_Logger()
    block = ""  # put i.e. 101 in here for block
    parent_raw = str(Path(raw).joinpath(str(block)))
    parent_der = str(Path(der).joinpath(str(block)))
    # check available models
    BIDS_Global_info.remove_splitting_key("chunk")
    BIDS_Global_info.remove_splitting_key("acq")
    bids_ds = BIDS_Global_info(
        datasets=[in_ds], parents=[parent_raw, parent_der, der, out_par, "derivatives_ax", "derivatives"], verbose=False
    )

    execution_times = []

    for name, subject in bids_ds.enumerate_subjects(sort=True):
        logger = head_logger.add_sub_logger(name=name)
        q = subject.new_query()
        q.flatten()
        q.filter("part", "inphase", required=False)
        q.filter("seg", lambda x: x != "manual", required=False)
        q.filter("lesions", lambda x: x != "manual", required=False)
        # q.filter("desc", "stitched")
        q.unflatten()
        q.filter_format("T2w")
        q.filter_filetype("nii.gz")
        families = q.loop_dict(sort=sort, key_addendum=["acq", "desc"])
        for f in families:
            try:
                ax = f["T2w_acq-ax"]
                sag = f["T2w_acq-sag_desc-stitched"][0]
                for axial_img in ax:
                    register_call_back(axial_img, sag, log=head_logger)
            except Exception:
                logger.print_error()

    if len(execution_times) > 0:
        head_logger.print(
            f"\nExecution times:\n{execution_times}\nRange:{min(execution_times)}, {max(execution_times)}\nAvg {np.average(execution_times)}"
        )


if __name__ == "__main__":
    ds = Path("/media/data/robert/datasets/dataset-neuropoly/")

    # class TIME(object):
    #    def __init__(self, name):
    #        self.name = name
    #
    #    def __enter__(self):
    #        self.start = time.time()
    #
    #    def __exit__(self, *args):
    #        print(f"{self.name} took {round(time.time()-self.start,4)} seconds")
    #
    # with TIME("####Name####") as xfile:
    #    # CODE DEN DU Timen möchtes
    #    list(range(100000000))
    Parallel(n_jobs=3)([delayed(run_registration)(ds), delayed(run_registration)(ds, sort=False)])
    # register_ax_sag()
    # run_sag(ds, stitching_only=True)
    # Parallel(n_jobs=3)(
    #    [delayed(run_sag)(ds, stitching_only=True, sort=True), delayed(run_sag)(ds, stitching_only=True, sort=False), delayed(run_sag)(ds)]
    # )
    # run_sag(Path(ds))

    # pip install 'resize @ git+https://gitlab.com/iacl/resize@v0.3.0'
    # pip install 'degrade @ git+https://gitlab.com/iacl/degrade@v0.4.0'
    # pip install openpyxl
    # pip install tensorboardX
    # pip install protobuf==3.20
    # pip install pillow==9.0.1
    # pip install simplification
    # pip install monai
