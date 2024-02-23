from pathlib import Path

from TPTBox import Location, calc_centroids_from_subreg_vert
from TPTBox.spine.snapshot2D import Snapshot_Frame, create_snapshot

vert_path = "/DATA/NAS/ongoing_projects/robert/paper/paper4/test_paper4/eval_DAE_10_abliation_10/sub-m967205/ses-20190708/anat/sub-m967205_ses-20190708_acq-iso_mod-T2w_seg-vert_desc-superres_msk.nii.gz"
subreg_path = "/DATA/NAS/ongoing_projects/robert/paper/paper4/test_paper4/eval_DAE_10_abliation_10/sub-m967205/ses-20190708/anat/sub-m967205_ses-20190708_acq-iso_mod-T2w_seg-spine_desc-superres_msk.nii.gz"
mr_path = "/DATA/NAS/ongoing_projects/robert/paper/paper4/test_paper4/eval_DAE_10_abliation_10/sub-m967205/ses-20190708/anat/sub-m967205_ses-20190708_acq-iso_desc-superres_T2w.nii.gz"
poi_ax_new = calc_centroids_from_subreg_vert(
    vert_path, subreg_path, subreg_id=[50, 100, Location.Spinal_Canal_ivd_lvl.value, Location.Spinal_Canal.value]
)
print(poi_ax_new)
cord_poi = poi_ax_new.extract_subregion(Location.Spinal_Canal_ivd_lvl, 100).map_labels(
    label_map_subregion={Location.Spinal_Canal_ivd_lvl.value: 50}
)
print()
frame_3 = Snapshot_Frame(mr_path, centroids=cord_poi, mode="MRI")
create_snapshot(Path(Path(subreg_path).parent, Path(subreg_path).name.replace("_msk.nii.gz", "_poi.jpg")), [frame_3])
p = poi_ax_new.extract_subregion(Location.Spinal_Canal_ivd_lvl.value)
