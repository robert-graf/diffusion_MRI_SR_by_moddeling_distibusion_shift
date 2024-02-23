from pathlib import Path

from script_sag_seg import run_registration
from TPTBox import BIDS_FILE, NII, POI, Location, Log_Type, No_Logger, calc_centroids_from_subreg_vert
from TPTBox.registration.ridged_points import ridged_points_from_poi
from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot

from pl_models.cycleGAN import CycleGAN_LitModel
from pl_models.DEA import DAE_LitModel

model: DAE_LitModel | CycleGAN_LitModel = None  # type: ignore
model_ax: DAE_LitModel = None  # type: ignore


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
    # print(mapping)
    return changing_poi.map_labels(label_map_region=mapping)


def vertebra_level(vert: NII, subreg: NII, poi: POI | None):
    from TPTBox import Location
    from TPTBox.core.vertebra_pois_non_centroids import calc_center_spinal_cord

    sag_poi = calc_centroids_from_subreg_vert(vert, subreg, subreg_id=[Location.Vertebra_Corpus, Location.Vertebra_Disc], extend_to=poi)
    back = subreg.copy()
    calc_center_spinal_cord(sag_poi, subreg, _fill_inplace=back, source_subreg_point_id=Location.Vertebra_Disc)
    return back.resample_from_to_(subreg), sag_poi


def register_ax_sag(axial_img: BIDS_FILE, sagittal_img: BIDS_FILE, log: No_Logger, override=False, parent="registrate_2"):
    # if axial_img.get("sub") != "m002886":
    #    return
    ## FILE STUFF ##
    chunk = axial_img.get("chunk")
    axial_out = axial_img.get_changed_bids(
        info={"acq": None, "desc": "registrate-ax", "res": f"sag-{chunk}"},
        parent=parent,
        make_parent=False,
        additional_folder=f"{chunk}",
    )
    snap = axial_out.dataset / parent / "snapshots" / str(axial_out.file["nii.gz"].name).replace("_T2w.nii.gz", "_snp.jpg")
    snap1 = str(axial_out.file["nii.gz"]).replace("_T2w.nii.gz", "_snp.jpg")
    # if override:
    #    snap.unlink(missing_ok=True)
    if snap.exists() and not override:
        log.print(axial_out, "exists. Skip", Log_Type.OK)
        return
    fam = axial_img.get_sequence_files(key_addendum=["acq", "desc", "chunk"])
    ######################

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
    try:
        poi = POI.load(poi_sag)
        if poi.origin is None:
            poi_sag = None
        else:
            poi_file = poi_sag.file["json"]

    except FileNotFoundError:
        pass

    ###### ALL FILES RELOADED ###

    #### CALC POIS for registaton ####
    subreg_id = [50, 61]
    ax_poi = calc_centroids_from_subreg_vert(vert_ax_nii, subreg_ax_nii, subreg_id=subreg_id, buffer_file=poi_ax.file["json"])
    sag_poi = calc_centroids_from_subreg_vert(vert_sag_nii, subreg_sag_nii, subreg_id=subreg_id, buffer_file=poi_file)
    sag_poi.shape = img_sag_nii.shape

    ax_poi = remap_centroids(ax_poi, sag_poi)
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
        info={"acq": None, "desc": "registrate-sag", "res": f"sag-{chunk}"}, parent=parent, make_parent=True, additional_folder=f"{chunk}"
    )

    moved_img.save(axial_out, make_parents=True)
    moved_vert.save(axial_out.get_changed_path(parent=parent, format="msk", info={"seg": "vert"}))
    moved_subreg.save(axial_out.get_changed_path(parent=parent, format="msk", info={"seg": "spine"}))
    fixed_vert.save(sagittal_out.get_changed_path(parent=parent, format="msk", info={"seg": "vert"}))
    fixed_subreg.save(sagittal_out.get_changed_path(parent=parent, format="msk", info={"seg": "spine"}))
    fixed_img.save(sagittal_out)
    frame_1 = Snapshot_Frame(moved_img, segmentation=moved_subreg, centroids=moved_poi)
    frame_2 = Snapshot_Frame(moved_img, segmentation=fixed_subreg, centroids=fixed_poi)
    frame_3 = Snapshot_Frame(fixed_img, segmentation=moved_subreg, centroids=moved_poi)
    frame_4 = Snapshot_Frame(fixed_img, segmentation=fixed_subreg, centroids=fixed_poi)

    create_snapshot([snap, snap1], [frame_1, frame_2, frame_3, frame_4])


affine_matrixes = {}


def register_sag_to_ax_stitched(
    axial_img: BIDS_FILE, sagittal_img: BIDS_FILE, log: No_Logger, override=False, parent="derivative_ax_stitched"
):
    fam = axial_img.get_sequence_files(key_addendum=["acq", "desc", "chunk"])
    ax_stitched_og = fam["T2w_acq-ax_desc-stitched"][0]
    vertebra_level_path = ax_stitched_og.get_changed_bids(format="msk", info={"seg": "vertebra-level"}, parent=parent, make_parent=False)
    snap = (
        vertebra_level_path.dataset / parent / "snapshots" / str(vertebra_level_path.file["nii.gz"].name).replace("_msk.nii.gz", "_snp.jpg")
    )
    if snap.exists() and not override:
        log.print(vertebra_level_path, "exists. Skip", Log_Type.OK)
        return
    try:
        img_sag_nii = sagittal_img.open_nii()
        vert_sag_nii = fam["msk_seg-vert_acq-sag_desc-stitched"][0].open_nii()
        vert_sag_nii.affine = img_sag_nii.affine
        subreg_sag_nii = fam["msk_seg-spine_acq-sag_desc-stitched"][0].open_nii()
        subreg_sag_nii.affine = img_sag_nii.affine
        poi_sag_bids = fam["ctd_seg-spine_acq-sag_desc-stitched"][0]
        poi_sag = fam["ctd_seg-spine_acq-sag_desc-stitched"][0].open_poi()
        # poi_ax = fam[f"ctd_seg-spine_acq-ax_chunk-{chunk}"][0]  # Contains all POIS
        ax_iso_subreg = fam["msk_seg-spine_acq-iso_desc-superres"][0]
        ax_iso_vert = fam["msk_seg-vert_acq-iso_desc-superres"][0]
        ax_iso_poi = fam["ctd_seg-spine_acq-iso_desc-superres"][0]
        # fam["msk_seg-vert_acq-ax_desc-stitched"]
    except FileNotFoundError as e:
        log.print("FileNotFoundError:", e.filename, Log_Type.FAIL)
        return

    # if axial_img.get("sub") != "m002886":
    #    return
    ## FILE STUFF - END##

    vertebra_level_nii_sag, poi_sag = vertebra_level(vert_sag_nii, subreg_sag_nii.copy(), poi_sag)
    vertebra_level_path2 = ax_stitched_og.get_changed_bids(
        format="msk", info={"seg": "vertebra-level", "acq": "sag"}, parent=parent, make_parent=False
    )

    vertebra_level_nii_sag.save(vertebra_level_path2, make_parents=True)

    ax_stitched_og_nii = ax_stitched_og.open_nii()
    snap.parent.mkdir(exist_ok=True, parents=True)
    snap1 = str(vertebra_level_path.file["nii.gz"]).replace("_msk.nii.gz", "_snp.jpg")

    #### CALC POIS for registaton ####
    subreg_id = [50, 100, 61, Location.Spinal_Canal_ivd_lvl.value]
    ax_vert_nii = ax_iso_vert.open_nii().resample_from_to(ax_stitched_og_nii)
    ax_subreg_nii = ax_iso_subreg.open_nii().resample_from_to(ax_stitched_og_nii)
    poi_ax = ax_iso_poi.open_poi().resample_from_to(ax_stitched_og_nii)
    # vertebra_level_path = ax_stitched_og.get_changed_bids(format="msk", info={"seg": "vertebra-level"}, parent=parent, make_parent=False)

    poi_ax = calc_centroids_from_subreg_vert(ax_vert_nii, ax_subreg_nii, subreg_id=subreg_id, extend_to=poi_ax)
    poi_sag = calc_centroids_from_subreg_vert(vert_sag_nii, subreg_sag_nii, subreg_id=subreg_id, extend_to=poi_sag)
    poi_sag.shape = img_sag_nii.shape
    poi_ax = remap_centroids(poi_ax, poi_sag)
    #### CALC resample filter ###
    resample_filter = ridged_points_from_poi(poi_ax.extract_subregion(*subreg_id), poi_sag, c_val=0, leave_worst_percent_out=0.1)

    affine_matrixes[fam.family_id] = resample_filter.get_affine().reshape(-1).tolist()
    ## target res ##
    fixed_img = ax_stitched_og_nii
    fixed_vert = ax_vert_nii
    fixed_subreg = ax_subreg_nii
    fixed_poi = poi_ax
    ## registrate sag to ax
    moved_img: NII = resample_filter.transform_nii(img_sag_nii)
    moved_vert: NII = resample_filter.transform_nii(vert_sag_nii.remove_labels(list(range(99, 400))))
    moved_subreg: NII = resample_filter.transform_nii(subreg_sag_nii.remove_labels(100, 63, 62))
    moved_ivd: NII = resample_filter.transform_nii(subreg_sag_nii.extract_label(100))

    moved_vertebra_level: NII = resample_filter.transform_nii(vertebra_level_nii_sag)
    moved_poi = resample_filter.transform_poi(poi_sag)

    crop = img_sag_nii.compute_crop(10)
    frame_1 = Snapshot_Frame(
        img_sag_nii.apply_crop(crop), segmentation=vertebra_level_nii_sag.apply_crop(crop), centroids=poi_sag.apply_crop(crop)
    )
    # frame_2 = Snapshot_Frame(fixed_img, segmentation=moved_vert, centroids=moved_poi.extract_subregion(50))
    # frame_3 = Snapshot_Frame(fixed_img, segmentation=moved_subreg.copy(), centroids=moved_poi.extract_subregion(61))
    # frame_4 = Snapshot_Frame(fixed_img, segmentation=moved_vertebra_level, centroids=moved_poi.extract_subregion(100))
    # frame_5 = Snapshot_Frame(fixed_img, centroids=moved_poi)
    # frame_6 = Snapshot_Frame(moved_img, centroids=moved_poi)
    # create_snapshot([snap, snap1], [frame_1, frame_2, frame_3, frame_4, frame_5, frame_6])
    frame_2 = Snapshot_Frame(fixed_img, segmentation=moved_vert, centroids=moved_poi.extract_subregion(50))
    frame_3 = Snapshot_Frame(fixed_img, segmentation=moved_subreg.copy(), centroids=moved_poi.extract_subregion(subreg_id[-1]))
    frame_4 = Snapshot_Frame(fixed_img, segmentation=moved_vertebra_level, centroids=moved_poi.extract_subregion(100))
    frame_2_f = Snapshot_Frame(fixed_img, segmentation=fixed_vert, centroids=fixed_poi.extract_subregion(50))
    frame_3_f = Snapshot_Frame(fixed_img, segmentation=fixed_subreg, centroids=fixed_poi.extract_subregion(subreg_id[-1]))

    frame_6 = Snapshot_Frame(moved_img, segmentation=fixed_subreg.extract_label(100), centroids=fixed_poi.extract_subregion(50))
    frame_6_f = Snapshot_Frame(fixed_img, segmentation=moved_ivd, centroids=fixed_poi.extract_subregion(50))
    create_snapshot([snap, snap1], [frame_1, frame_2, frame_2_f, frame_3, frame_3_f, frame_4, frame_6, frame_6_f])
    moved_vert.save(ax_stitched_og.get_changed_path(format="msk", info={"seg": "vert"}, parent=parent))
    moved_subreg.save(ax_stitched_og.get_changed_path(format="msk", info={"seg": "spine"}, parent=parent))
    moved_vertebra_level.save(vertebra_level_path)
    moved_poi.save(ax_stitched_og.get_changed_path(format="poi", info={"seg": "spine"}, parent=parent, file_type="json"))
    with open(ds / "affines.json", "w") as outfile:
        json_object = json.dumps(affine_matrixes, indent=4)
        outfile.write(json_object)


# axial = fam["T2w_acq-ax_desc-stitched"]
# exit()
if __name__ == "__main__":
    import json

    # ds = Path("/media/data/robert/datasets/dataset-neuropoly/")
    ds = Path("/media/data/robert/test_hendrik/dataset-neuropoly/")
    buffer = ds / "affines.json"
    if buffer.exists():
        with open(buffer) as outfile:
            json_object = json.load(outfile, indent=4)

    # run_registration(ds, register_call_back=register_ax_sag)
    # Parallel(n_jobs=3)([delayed(run_sag)(ds), delayed(run_sag)(ds, sort=False)])
    run_registration(ds, register_call_back=register_sag_to_ax_stitched)

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
