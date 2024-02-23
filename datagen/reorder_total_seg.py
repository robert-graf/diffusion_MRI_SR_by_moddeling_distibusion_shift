from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd
from BIDS import NII


class Region(Enum):
    subcutaneous_tissue = auto()
    # Organs (no fatty)
    spleen = auto()
    kidney_right = auto()
    kidney_left = kidney_right  # auto()
    gallbladder = auto()
    pancreas = auto()
    liver = auto()
    OAR_Eye_AL = auto()
    OAR_Eye_AR = OAR_Eye_AL  # auto()
    OAR_Eye_PL = auto()
    OAR_Eye_PR = OAR_Eye_PL  # auto()
    # Nerfs
    OAR_OpticChiasm = auto()
    OAR_OpticNrv_L = auto()
    OAR_OpticNrv_R = OAR_OpticNrv_L  # auto()
    OAR_Glottis = auto()

    # eye_balls = auto()
    larynx = auto()
    urinary_bladder = auto()

    prostate = auto()
    # heart
    heart_myocardium = auto()
    heart_atrium_left = auto()
    heart_ventricle_left = auto()
    heart_atrium_right = auto()
    heart_ventricle_right = auto()
    pericardium = auto()

    # arteries/veins
    aorta = auto()
    brain_arteries_and_veins = auto()
    inferior_vena_cava = auto()
    portal_vein_and_splenic_vein = auto()
    adrenal_gland_right = auto()
    adrenal_gland_left = adrenal_gland_right  # auto()
    pulmonary_artery = auto()
    OAR_A_Carotid_R = auto()
    OAR_A_Carotid_L = OAR_A_Carotid_R  # auto()
    iliac_artery_left = auto()
    iliac_artery_right = iliac_artery_left  # auto()
    iliac_vena_left = auto()
    iliac_vena_right = iliac_vena_left  # auto()
    # OAR_Arytenoid = auto()

    # Lung
    lung_upper_lobe_left = auto()
    lung_lower_lobe_left = lung_upper_lobe_left  # auto()
    lung_upper_lobe_right = lung_upper_lobe_left  # auto()  # noqa: PIE796
    lung_middle_lobe_right = lung_upper_lobe_left  # auto() # noqa: PIE796
    lung_lower_lobe_right = lung_upper_lobe_left  # auto() # noqa: PIE796
    # thoracic_cavity = auto()
    # lung = auto()
    trachea = auto()
    mediastinum = auto()
    # brain
    brain_head = auto()
    brain = auto()
    brainstem = auto()
    white_matter = auto()
    gray_matter = auto()
    OAR_Pituitary = auto()
    csf = auto()
    # brain = auto()
    # Spinalcord
    spinal_canal = auto()
    spinal_cord = auto()

    # digestion
    esophagus = auto()
    stomach = auto()
    small_bowel = auto()
    duodenum = auto()
    colon = auto()
    sigmoid = auto()
    rectum = auto()
    # Bone
    head_bone = auto()
    compact_bone = auto()
    spongy_bone = auto()
    OAR_Bone_Mandible = auto()

    # Bone CHEST
    sternum = auto()
    rib = auto()
    vertebrae = auto()
    sacrum = auto()
    # Bone Lumbar
    humerus_left = auto()
    humerus_right = humerus_left  # auto()
    scapula_left = auto()
    scapula_right = scapula_left  # auto()
    clavicula_left = auto()
    clavicula_right = clavicula_left  # auto()
    femur_left = auto()
    femur_right = femur_left  # auto()
    hip_left = auto()
    hip_right = hip_left  # auto()
    # bones = auto()
    # muscle
    gluteus_maximus_left = auto()
    gluteus_maximus_right = gluteus_maximus_left  # auto()
    gluteus_medius_left = auto()
    gluteus_medius_right = gluteus_medius_left  # auto()
    gluteus_minimus_left = auto()
    gluteus_minimus_right = gluteus_minimus_left  # auto()
    autochthon_left = auto()
    autochthon_right = autochthon_left  # auto()
    iliopsoas_left = auto()
    iliopsoas_right = iliopsoas_left  # auto()
    right_psoas_major = auto()
    left_psoas_major = right_psoas_major  # auto()
    right_rectus_abdominis = auto()
    left_ectus_abdominis = right_rectus_abdominis  # auto()
    head_muscles = auto()
    OAR_Cricopharyngeus = auto()
    # muscle = auto()
    # glands
    thyroid_gland = auto()
    right_adrenal_gland = auto()
    left_adrenal_gland = right_adrenal_gland  # auto()
    left_parotid_gland = auto()
    right_parotid_gland = left_parotid_gland  # auto()
    left_submandibular_gland = auto()
    right_submandibular_gland = left_submandibular_gland  # auto()
    left_mammary_gland = auto()
    right_mammary_gland = left_mammary_gland  # auto()
    OAR_Glnd_Lacrimal_L = auto()
    OAR_Glnd_Lacrimal_R = OAR_Glnd_Lacrimal_L  # auto()
    OAR_Glnd_Submand_L = auto()
    OAR_Glnd_Submand_R = OAR_Glnd_Submand_L  # auto()
    OAR_Glnd_Thyroid = auto()
    OAR_Parotid_L = auto()
    OAR_Parotid_R = OAR_Parotid_L  # auto()
    seminal_vasicle = auto()
    # glands = auto()

    # Others
    OAR_Cavity_Oral = auto()
    OAR_BuccalMucosa = auto()

    # Head others

    # face = auto()
    # scalp = auto()
    # blood = auto()
    # abdominal_cavity = auto()
    # breast_implant = auto()

    # Head ear
    # OAR_Cochlea_L = auto()
    # Head Mouth

    # Liquids others
    # others

    # foreign object
    # OAR_Brainstem = auto()
    # OAR_Esophagus_S = auto()

    # OAR_Larynx_SG = auto()
    OAR_Lips = auto()

    # OAR_SpinalCord = auto()
    LAST = auto()
    ZERO = 0


mapping_full = {
    # Organs (no fatty)
    1: Region.spleen,
    2: Region.kidney_right,
    3: Region.kidney_left,
    4: Region.gallbladder,
    5: Region.liver,
    6: Region.stomach,
    # arteries and veins
    7: Region.aorta,
    8: Region.inferior_vena_cava,
    9: Region.portal_vein_and_splenic_vein,
    10: Region.pancreas,
    11: Region.adrenal_gland_right,
    12: Region.adrenal_gland_left,
    # Lung
    13: Region.ZERO,  # .lung_upper_lobe_left,
    14: Region.ZERO,  # .lung_lower_lobe_left,
    15: Region.ZERO,  # .lung_upper_lobe_right,
    16: Region.ZERO,  # .lung_middle_lobe_right,
    17: Region.ZERO,  # .lung_lower_lobe_right,
    # vertebra
    18: Region.vertebrae,
    19: Region.vertebrae,
    20: Region.vertebrae,
    21: Region.vertebrae,
    22: Region.vertebrae,
    23: Region.vertebrae,
    24: Region.vertebrae,
    25: Region.vertebrae,
    26: Region.vertebrae,
    27: Region.vertebrae,
    28: Region.vertebrae,
    29: Region.vertebrae,
    30: Region.vertebrae,
    31: Region.vertebrae,
    32: Region.vertebrae,
    33: Region.vertebrae,
    34: Region.vertebrae,
    35: Region.vertebrae,
    36: Region.vertebrae,
    37: Region.vertebrae,
    38: Region.vertebrae,
    39: Region.vertebrae,
    40: Region.vertebrae,
    41: Region.vertebrae,
    #
    42: Region.esophagus,
    43: Region.trachea,
    # heart
    44: Region.heart_myocardium,
    45: Region.heart_atrium_left,
    46: Region.heart_ventricle_left,
    47: Region.heart_atrium_right,
    48: Region.heart_ventricle_right,
    # artery
    49: Region.pulmonary_artery,
    # brain
    # 50: Region.brain,
    # arteries and veins
    51: Region.iliac_artery_left,
    52: Region.iliac_artery_right,
    53: Region.iliac_vena_left,
    54: Region.iliac_vena_right,
    #
    55: Region.small_bowel,
    56: Region.duodenum,
    57: Region.colon,
    # RIBS
    58: Region.rib,
    59: Region.rib,
    60: Region.rib,
    61: Region.rib,
    62: Region.rib,
    63: Region.rib,
    64: Region.rib,
    65: Region.rib,
    66: Region.rib,
    67: Region.rib,
    68: Region.rib,
    69: Region.rib,
    70: Region.rib,
    71: Region.rib,
    72: Region.rib,
    73: Region.rib,
    74: Region.rib,
    75: Region.rib,
    76: Region.rib,
    77: Region.rib,
    78: Region.rib,
    79: Region.rib,
    80: Region.rib,
    81: Region.rib,
    82: Region.humerus_left,
    83: Region.humerus_right,
    84: Region.scapula_left,
    85: Region.scapula_right,
    86: Region.clavicula_left,
    87: Region.clavicula_right,
    88: Region.femur_left,
    89: Region.femur_right,
    90: Region.hip_left,
    91: Region.hip_right,
    92: Region.sacrum,
    # 93: Region.face,
    94: Region.gluteus_maximus_left,
    95: Region.gluteus_maximus_right,
    96: Region.gluteus_medius_left,
    97: Region.gluteus_medius_right,
    98: Region.gluteus_minimus_left,
    99: Region.gluteus_minimus_right,
    100: Region.autochthon_left,
    101: Region.autochthon_right,
    102: Region.iliopsoas_left,
    103: Region.iliopsoas_right,
    104: Region.urinary_bladder,
    105: Region.sternum,
    106: Region.thyroid_gland,
    107: Region.right_adrenal_gland,
    108: Region.left_adrenal_gland,
    109: Region.right_psoas_major,
    110: Region.left_psoas_major,
    111: Region.right_rectus_abdominis,
    112: Region.left_ectus_abdominis,
    113: Region.ZERO,  # Brainstem but brocken
    114: Region.spinal_canal,
    115: Region.left_parotid_gland,
    116: Region.right_parotid_gland,
    117: Region.left_submandibular_gland,
    118: Region.right_submandibular_gland,
    119: Region.larynx,
    120: Region.sigmoid,
    121: Region.rectum,
    122: Region.prostate,
    123: Region.seminal_vasicle,
    124: Region.left_mammary_gland,
    125: Region.right_mammary_gland,
    126: Region.white_matter,
    127: Region.ZERO,
    # 128: Region.csf,
    129: Region.head_bone,
    # 130: Region.scalp,
    131: Region.ZERO,  # Region.eye_balls,
    132: Region.compact_bone,
    133: Region.spongy_bone,
    # 134: Region.blood,
    135: Region.head_muscles,
    136: Region.white_matter,
    137: Region.gray_matter,
    138: Region.csf,
    # 139: Region.OAR_Bone_Mandible,
    # 140: Region.OAR_Brainstem,
    141: Region.ZERO,  # Region.eye_balls,
    142: Region.head_bone,
    143: Region.head_bone,
    144: Region.brain_arteries_and_veins,
    145: Region.OAR_Cricopharyngeus,
    146: Region.esophagus,  # Region.OAR
    # 147: Region.OAR_Eye_AL,
    # 148: Region.OAR_Eye_AR,
    # 149: Region.OAR_Eye_PL,
    # 150: Region.OAR_Eye_PR,
    # 151: Region.OAR_Glnd_Lacrimal_L,
    # 152: Region.OAR_Glnd_Lacrimal_R,
    # 153: Region.OAR_Glnd_Submand_L,
    # 154: Region.OAR_Glnd_Submand_R,
    # 155: Region.OAR_Glnd_Thyroid,
    # 156: Region.OAR_Glottis,
    # 157: Region.OAR_Larynx_SG,
    # 158: Region.OAR_Lips,
    # 159: Region.OAR_OpticChiasm,
    # 160: Region.OAR_OpticNrv_L,
    # 161: Region.OAR_OpticNrv_R,
    # 162: Region.OAR_Parotid_L,
    # 163: Region.OAR_Parotid_R,
    # 164: Region.OAR_Pituitary,
    # 165: Region.OAR_SpinalCord,
    # 167: Region.muscle,
    168: Region.brain,
    169: Region.ZERO,  # thoracic_cavity,
    # 170: Region.bones,
    # 171: Region.glands,
    # 172: Region.pericardium,
    # 173: Region.breast_implant,
    174: Region.mediastinum,
    # 175: Region.brain_head,
    # 176: Region.spinal_cord,
    166: Region.subcutaneous_tissue,
}


mapping_251 = {
    1: Region.ZERO,
    2: Region.ZERO,
    3: Region.ZERO,
    4: Region.ZERO,
    5: Region.ZERO,
    6: Region.ZERO,
    7: Region.ZERO,
    8: Region.ZERO,
    9: Region.ZERO,
    10: Region.ZERO,
    11: Region.ZERO,
    12: Region.ZERO,
    13: Region.lung_upper_lobe_left,
    14: Region.lung_lower_lobe_left,
    15: Region.lung_upper_lobe_right,
    16: Region.lung_middle_lobe_right,
    17: Region.lung_lower_lobe_right,
}
mapping_257 = {
    1: Region.OAR_A_Carotid_L,
    2: Region.OAR_A_Carotid_R,
    3: Region.ZERO,  # "OAR_Arytenoid",
    4: Region.OAR_Bone_Mandible,
    5: Region.brainstem,
    6: Region.OAR_BuccalMucosa,
    7: Region.OAR_Cavity_Oral,
    8: Region.ZERO,  # "OAR_Cochlea_L",
    9: Region.ZERO,  # "OAR_Cochlea_R",
    10: Region.ZERO,  # "OAR_Cricopharyngeus",
    11: Region.esophagus,  # "OAR_Esophagus_S", TODO
    12: Region.OAR_Eye_AL,
    13: Region.OAR_Eye_AR,
    14: Region.OAR_Eye_PL,
    15: Region.OAR_Eye_PR,
    16: Region.OAR_Glnd_Lacrimal_L,
    17: Region.OAR_Glnd_Lacrimal_R,
    18: Region.OAR_Glnd_Submand_L,
    19: Region.OAR_Glnd_Submand_R,
    20: Region.OAR_Glnd_Thyroid,
    21: Region.OAR_Glottis,
    22: Region.larynx,  # "OAR_Larynx_SG",TODO
    23: Region.OAR_Lips,
    24: Region.OAR_OpticChiasm,
    25: Region.OAR_OpticNrv_L,
    26: Region.OAR_OpticNrv_R,
    27: Region.OAR_Parotid_L,
    28: Region.OAR_Parotid_R,
    29: Region.OAR_Pituitary,  # TODO remove?
    30: Region.spinal_cord,  # "OAR_SpinalCord",
}
mapping_258 = {
    1: "white matter",
    2: "gray matter",
    3: "csf",
    4: "head bone",
    5: "scalp",
    6: "eye balls",
    7: "compact bone",
    8: "spongy bone",
    9: "blood",
    10: "head muscles",
}
border = [
    0,
    Region.subcutaneous_tissue.value,  # SKIN 1
    Region.pericardium.value,  # Organs 2
    Region.iliac_vena_right.value,  # vena/ateria 3
    Region.mediastinum.value,  # Lunge 4
    Region.spinal_cord.value,  # Brain/sinalcord 5
    Region.rectum.value,  # Digestions system 6
    Region.hip_right.value,  # Bone 7
    Region.seminal_vasicle.value,  # muscles 8 (except Skin)
    Region.LAST.value,  # others N. A.
]


def get_body_comp_nii(seg_path: str | Path, orientation=("R", "I", "P"), combined: NII | None = None):
    """Body compt
    0: Background
    1: Skin
    2: Muscle (including skin muscles)
    3: Inner Fat
    --- 4: muscle Fat (to be removed) ---
    ##### Later: 4 (new): Abdominal Organs (liver, kidney, pancreas,spleen)
    """
    if combined is None:
        combined = get_reduced_nii(seg_path, orientation=orientation)[0]
    arr_reduced = combined.get_array()
    ### BUILD body_comp
    # (comb) 166 -> 1
    # (combined) 8 -> 2
    # (combined) 10 -> 2
    # (combined) 9 -> 3
    out = arr_reduced.copy() * 0
    out[arr_reduced == 1] = 1
    out[arr_reduced == 8] = 2
    out[arr_reduced == 10] = 2
    out[arr_reduced == 9] = 3
    return combined.set_array(out)


def get_reduced_nii(seg_path: str | Path, orientation=("R", "I", "P")):
    seg_path = Path(seg_path)
    if not seg_path.is_dir():
        seg_path = seg_path.parent
    comb_seg = next(seg_path.glob("*_combined.nii.gz"))
    seg_251 = next(seg_path.glob("*_251.nii.gz"))
    seg_257 = next(seg_path.glob("*_257.nii.gz"))
    # seg_258 = next(path.glob("*_258.nii.gz"))
    rough_seg = next(seg_path.glob("*_259.nii.gz"))
    ### Fetch from COMBI
    comb_seg = NII.load(comb_seg, True).reorient_(orientation)
    out_arr = comb_seg.get_array() * 0
    in_arr = comb_seg.get_array()
    for i in comb_seg.unique():
        try:
            out_arr[in_arr == i] = mapping_full[i].value
        except KeyError as e:
            print(e)

    ### 251
    for mapping, seg_path in [(mapping_251, seg_251), (mapping_257, seg_257)]:
        seg_temp = NII.load(seg_path, True).reorient_(orientation)
        in_arr = seg_temp.get_array()
        for i in seg_temp.unique():
            try:
                v = mapping[i].value
                if v == 0:
                    continue
                out_arr[in_arr == i] = v
            except KeyError as e:
                print(e, seg_path.name)

    seg = comb_seg.set_array(out_arr)
    arr_reduced = out_arr.copy() * 0
    subregions = []
    for e, (i, j) in enumerate(zip(border[:-1], border[1:], strict=True), start=1):
        arr = out_arr.copy()
        arr[arr > j] = 0
        arr[arr <= i] = 0
        arr_reduced[arr != 0] = e
        organs = seg.set_array(arr)
        subregions.append(organs)

    seg2 = NII.load(rough_seg, True).reorient_(orientation)
    arr2 = seg2.get_array()

    arr_reduced[np.logical_and(arr_reduced == 0, arr2 == 3)] = arr_reduced.max() + 1
    arr_reduced[np.logical_and(arr_reduced == 0, arr2 == 2)] = arr_reduced.max() + 1
    rough_bone = seg2.extract_label(5)
    rough_bone = rough_bone.erode_msk(2).dilate_msk(1) * rough_bone
    arr_reduced[np.logical_and(arr_reduced == 0, rough_bone.get_array() == 1)] = arr_reduced.max() + 1

    nii_reduced = comb_seg.set_array(arr_reduced)
    return nii_reduced, subregions, seg


def get_all_files(split="train", root="/DATA/NAS/datasets_processed/CT_fullbody/dataset-CACTS/ct_ready_1mm/", expect_find_all=True):
    np.random.seed(1337)
    if split == "train":
        split_id = 1
    elif split == "test":
        split_id = 2
    elif split == "val":
        split_id = 0
    else:
        raise NotImplementedError(split)
    out_list = []
    for path in sorted(Path(root).iterdir()):
        if not path.is_dir():
            continue
        xlsx_path = path / "dataset_summary.xlsx"

        if xlsx_path.exists():
            xlsx = pd.read_excel(xlsx_path)

            if len(xlsx["split"].value_counts()) == 1:
                new_split = np.random.randint(0, 9, xlsx.shape[0])
                new_split[new_split > 2] = 1
                xlsx["split"] = new_split

            if len(xlsx["split"].value_counts()) == 2:
                xlsx[xlsx["split"] == 3] = 1
                new_split = np.random.randint(0, 2, xlsx.shape[0]) * 2
                arr = xlsx["split"].to_numpy().astype(float)
                new_split[arr == 1] = 1
                xlsx["split"] = new_split
                assert len(np.unique(xlsx["split"])) == 3, (xlsx_path, np.unique(xlsx["split"]))

            xlsx[xlsx["split"] == 3] = 2
            xlsx = xlsx.fillna(2)

            if len(xlsx["split"].value_counts()) != 3:
                print(xlsx["split"].value_counts())
                assert False
            xlsx = xlsx[xlsx["split"] == split_id]
            for key in list(xlsx["ids"]):
                if key == "Average/Sum":
                    continue
                new_file = path / "images" / (str(key) + "_0000.nii.gz")
                if not new_file.exists():
                    f = path / "images"
                    try:
                        new_file = next(f.glob("*" + str(key) + "*_0000.nii.gz"))
                    except StopIteration:
                        if expect_find_all:
                            print(new_file)
                            raise FileNotFoundError(new_file) from None

                        continue
                assert new_file.exists(), new_file
                out_list.append(new_file)
        else:
            p = path / "images"
            out_list += list(p.glob(f"*{split}*"))
    out_list_filtered = []
    out_list_filtered_seg = []
    for i in out_list:
        path = get_seg_from_file(i)
        if path is None:
            continue
        out_list_filtered.append(i)
        out_list_filtered_seg.append(path)

    return out_list_filtered, out_list_filtered_seg


def get_seg_from_file(files: Path | str):
    files = str(files).replace("_0000.nii.gz", "")
    files = str(files).replace("ct_ready_1mm", "generated_segmentations")
    if not Path(files).exists():
        return None
    return files


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        root = "/DATA/NAS/datasets_processed/CT_fullbody/dataset-CACTS/ct_ready_1mm/"
        expect_find_all = True
        out_folder = "/DATA/NAS/datasets_processed/CT_fullbody/dataset-CACTS/reduced"

        if not Path(root).exists():
            root = "D:/data/totalseg/ct_ready_1mm"
            out_folder = "D:/data/totalseg/ct_ready_1mm/reduced"

            expect_find_all = False
        files, files_seg = get_all_files(split=split, root=root, expect_find_all=expect_find_all)
        from BIDS import NII

        for i, (ct_path, seg_path_root) in enumerate(zip(files, files_seg, strict=True)):
            print(i, ct_path)
            try:
                out_path = Path(out_folder, *list(Path(ct_path).parts[-3:]))
                out_name = out_path.name
                out_path = out_path.parent
                out_final = Path(out_path, out_name.replace("0000.nii.gz", "all.nii.gz"))
                out_body_comp = Path(out_path, out_name.replace("0000.nii.gz", "body-comp.nii.gz"))
                out_reduced = Path(out_path, out_name.replace("0000.nii.gz", "reduced.nii.gz"))
                if not out_final.exists():
                    out_path.mkdir(exist_ok=True, parents=True)
                    ct = NII.load(ct_path, False)
                    comb, _, seg_ = get_reduced_nii(seg_path_root, orientation=ct.orientation)
                    comb.save(out_reduced)
                    seg_.save(out_final)
                if not out_body_comp.exists():
                    nii = get_body_comp_nii(seg_path_root, combined=NII.load(out_reduced, True))
                    nii.save(out_body_comp)
            except Exception as e:
                print(e)

        # niis = dict(ct=ct, seg=seg)
        # out = Path("~/TotalSegmentorPNG").expanduser()
        # out = Path("/DATA/NAS/ongoing_projects/robert/train_dataset/totalSeg")
