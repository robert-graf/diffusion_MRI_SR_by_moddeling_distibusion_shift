from pathlib import Path

from script_sag_seg import run_registration
from script_sag_seg_v2 import remap_centroids, vertebra_level
from TPTBox import BIDS_FILE, NII, Location, Log_Type, No_Logger, calc_centroids_from_subreg_vert
from TPTBox.registration.ridged_points import ridged_points_from_poi
from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot

from pl_models.cycleGAN import CycleGAN_LitModel
from pl_models.DEA import DAE_LitModel

model: DAE_LitModel | CycleGAN_LitModel = None  # type: ignore
model_ax: DAE_LitModel = None  # type: ignore
affine_matrixes = {}


def get_snap_name(ax_stitched_og: BIDS_FILE, parent):
    vertebra_level_snap = ax_stitched_og.get_changed_bids(
        "jpg", format="snp", info={"seg": "vertebra-level"}, parent=parent, make_parent=False
    )
    snap = vertebra_level_snap.dataset / parent / "snapshots" / str(vertebra_level_snap.file["jpg"].name)
    return vertebra_level_snap.file["jpg"], snap


# def _register_chunk(img_sag_nii: NII, vert_sag_nii: NII, subreg_sag_nii: NII, poi_sag: POI, vertebra_level_path: BIDS_FILE, snap: Path):


def register_sag_to_ax_chunks(axial_img: BIDS_FILE, sagittal_img: BIDS_FILE, log: No_Logger, override=False, parent="derivative_ax_reg"):
    fam = axial_img.get_sequence_files(key_addendum=["acq", "desc", "lesions"])
    chunks = fam["T2w_acq-ax"]
    fam = axial_img.get_sequence_files(key_addendum=["acq", "desc", "chunk", "lesions"])

    if all(get_snap_name(a, parent)[1].exists() for a in chunks) and not override:
        log.print(chunks[0], "exists. Skip", Log_Type.OK)
    try:
        img_sag_nii = sagittal_img.open_nii()
        vert_sag_nii = fam["msk_seg-vert_acq-sag_desc-stitched"][0].open_nii()
        vert_sag_nii.affine = img_sag_nii.affine
        subreg_sag_nii = fam["msk_seg-spine_acq-sag_desc-stitched"][0].open_nii()
        subreg_sag_nii.affine = img_sag_nii.affine
        poi_sag = fam["ctd_seg-spine_acq-sag_desc-stitched"][0].open_poi()
    except FileNotFoundError as e:
        log.print("Sagital FileNotFoundError:", e.filename, Log_Type.FAIL)
        return

    ### CALC POI for SAG ###
    vertebra_level_nii_sag, poi_sag = vertebra_level(vert_sag_nii, subreg_sag_nii.copy(), poi_sag)
    info = {"seg": "vertebra-level", "acq": "sag"}
    vertebra_level_path = sagittal_img.get_changed_bids(format="msk", info=info, parent=parent, make_parent=False)
    vertebra_level_nii_sag.save(vertebra_level_path, make_parents=True)
    for chunk in chunks:
        try:
            snap1, snap = get_snap_name(chunk, parent)
            snap.parent.mkdir(exist_ok=True, parents=True)
            if snap.exists():
                continue
            # FIND axial scans
            fam = axial_img.get_sequence_files(key_addendum=["acq", "desc", "chunk", "lesions"])
            q = fam.new_query()
            q.flatten()
            q.filter("chunk", chunk.get("chunk"))
            q.unflatten()
            a = next(iter(q.loop_dict(key_addendum=["desc"])))
            ax_nii = chunk.open_nii()
            #### CALC POIS for registaton ####

            subreg_id = [
                Location.Vertebra_Disc.value,
                Location.Spinal_Canal_ivd_lvl.value,
                50,
                # *range(40, 50),
                Location.Spinal_Canal.value,
            ]
            others = [50, 100]
            ax_vert_nii = a["msk_seg-vert"][0].open_nii().resample_from_to(ax_nii)
            ax_subreg_nii = a["msk_seg-spine"][0].open_nii().resample_from_to(ax_nii)
            poi_ax = a["ctd_seg-spine"][0].open_poi().resample_from_to(ax_nii)
            poi_ax = calc_centroids_from_subreg_vert(ax_vert_nii, ax_subreg_nii, subreg_id=[*subreg_id, *others], extend_to=poi_ax)
            poi_sag = calc_centroids_from_subreg_vert(vert_sag_nii, subreg_sag_nii, subreg_id=[*subreg_id, *others], extend_to=poi_sag)
            poi_sag.shape = img_sag_nii.shape
            poi_ax = remap_centroids(poi_ax, poi_sag)
            #### CALC resample filter ###
            resample_filter = ridged_points_from_poi(poi_ax.extract_subregion(*subreg_id), poi_sag, c_val=0, leave_worst_percent_out=0.1)
            affine_matrixes[fam.family_id + f"_chunk-{chunk.get('chunk')}"] = resample_filter.get_affine().reshape(-1).tolist()
            ## target res ##
            fixed_img = ax_nii
            fixed_vert = ax_vert_nii
            fixed_subreg = ax_subreg_nii
            fixed_poi = poi_ax
            ## registrate sag to ax
            moved_img: NII = resample_filter.transform_nii(img_sag_nii)
            moved_vert: NII = resample_filter.transform_nii(vert_sag_nii.remove_labels(list(range(99, 400))))
            moved_subreg: NII = resample_filter.transform_nii(subreg_sag_nii.remove_labels(100, 62, 63))
            moved_ivd: NII = resample_filter.transform_nii(subreg_sag_nii.extract_label(100))
            moved_vertebra_level: NII = resample_filter.transform_nii(vertebra_level_nii_sag)
            moved_poi = resample_filter.transform_poi(poi_sag)
            frame_2 = Snapshot_Frame(fixed_img, segmentation=moved_vert, centroids=moved_poi.extract_subregion(50))
            frame_3 = Snapshot_Frame(fixed_img, segmentation=moved_subreg.copy(), centroids=moved_poi.extract_subregion(subreg_id[-1]))
            frame_4 = Snapshot_Frame(fixed_img, segmentation=moved_vertebra_level, centroids=moved_poi.extract_subregion(100))
            # frame_5 = Snapshot_Frame(fixed_img, centroids=moved_poi)
            frame_2_f = Snapshot_Frame(fixed_img, segmentation=fixed_vert, centroids=fixed_poi.extract_subregion(50))
            frame_3_f = Snapshot_Frame(fixed_img, segmentation=fixed_subreg, centroids=fixed_poi.extract_subregion(subreg_id[-1]))

            frame_6 = Snapshot_Frame(moved_img, segmentation=fixed_subreg.extract_label(100), centroids=fixed_poi.extract_subregion(50))
            frame_6_f = Snapshot_Frame(fixed_img, segmentation=moved_ivd, centroids=fixed_poi.extract_subregion(50))

            create_snapshot([snap, snap1], [frame_2, frame_2_f, frame_3, frame_3_f, frame_4, frame_6, frame_6_f])
            moved_vert.save(chunk.get_changed_path(format="msk", info={"seg": "vert"}, parent=parent))
            moved_subreg.save(chunk.get_changed_path(format="msk", info={"seg": "spine"}, parent=parent))
            moved_vertebra_level.save(vertebra_level_path)
            moved_poi.save(chunk.get_changed_path(format="poi", info={"seg": "spine"}, parent=parent, file_type="json"))
        except Exception:
            No_Logger().print_error()
    with open(ds / "affines2.json", "w") as outfile:
        json_object = json.dumps(affine_matrixes, indent=4)
        outfile.write(json_object)


# axial = fam["T2w_acq-ax_desc-stitched"]
# exit()
if __name__ == "__main__":
    import json

    # ds = Path("/media/data/robert/datasets/dataset-neuropoly/")
    ds = Path("/media/data/robert/test_hendrik/dataset-neuropoly/")
    buffer = ds / "affines2.json"
    if buffer.exists():
        with open(buffer) as outfile:
            json_object = json.load(outfile)

    # run_registration(ds, register_call_back=register_ax_sag)
    # Parallel(n_jobs=3)([delayed(run_sag)(ds), delayed(run_sag)(ds, sort=False)])
    run_registration(ds, register_call_back=register_sag_to_ax_chunks)

    # Serializing json

    with open(buffer, "w") as outfile:
        json_object = json.dumps(affine_matrixes, indent=4)
        outfile.write(json_object)
    #
    # Parallel(n_jobs=3)(
    #    [
    #        delayed(run_registration)(ds, register_call_back=register_ax_sag),
    #        delayed(run_registration)(ds, sort=False, register_call_back=register_ax_sag),
    #    ]
    # )
