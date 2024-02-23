import gzip
import pickle
import sys
from pathlib import Path

from BIDS import BIDS_Global_info, Log_Type, No_Logger, calc_centroids_from_subreg_vert, v_name2idx

sys.path.append(str(Path(__file__).parent.parent))
from script_axial_seg_no_stiching import get_snap_name, remap_centroids  # noqa: E402

### REMOVE in Axial
remove_ctd: dict[str, dict[str, list[int]]] = {
    "m073580": {"20111101": [v_name2idx["L1"]]},
    "m094683": {"20181115": [v_name2idx["L3"]], "20201104": [v_name2idx["L4"]]},
    "m142267": {"20200727": [v_name2idx["L3"]]},
    # "m153977": {"20180422": [v_name2idx["L4"], v_name2idx["L5"], v_name2idx["S1"]]},
    "m153977": {"20200529": [v_name2idx["L4"]]},
    "m198946": {"20180304": [v_name2idx["L3"]], "20200518": [v_name2idx["L3"]]},
    "m255816": {"20131113": [v_name2idx["L2"]], "20190416": [v_name2idx["L2"]]},
    "m282301": {"20191021": [v_name2idx["L3"]], "20191208": [v_name2idx["L3"]], "20210322": [v_name2idx["L4"], v_name2idx["L3"]]},
    "m321757": {"20171228": [v_name2idx["L3"]], "20200228": [v_name2idx["L3"]]},
    "m363201": {"20200706": [v_name2idx["L3"]], "20200820": [v_name2idx["L4"], v_name2idx["L3"]], "20210311": [v_name2idx["L2"]]},
    "m422496": {"20150804": [v_name2idx["L3"]], "20160609": [v_name2idx["L2"]]},
    "m468624": {"20190308": [v_name2idx["L3"]]},
    "m469393": {
        "20140630": [v_name2idx["L3"]],
        "20151124": [v_name2idx["L3"]],
        "20170518": [v_name2idx["L2"]],
        "20180723": [v_name2idx["L2"]],
    },
    "m476758": {"20140127": [v_name2idx["L1"]]},
    "m516440": {"20200318": [v_name2idx["L4"]]},
    "m519705": {"20140409": [v_name2idx["L3"]]},
    "m527202": {"20090928": [v_name2idx["L2"], v_name2idx["S1"]], "20140925": [v_name2idx["L3"]]},
    "m556495": {"20200901": [v_name2idx["L2"]], "20210430": [v_name2idx["L2"]]},
    "m640268": {"20150820": [v_name2idx["L3"]]},
    "m693352": {"20190301": [v_name2idx["L2"]]},
    "m698817": {"20130531": [v_name2idx["L2"]]},
    "m702654": {"20180501": [v_name2idx["L2"]]},
    "m707076": {"20180220": [v_name2idx["L4"]], "20200403": [v_name2idx["L4"]]},
    "m725157": {"20200729": [v_name2idx["L3"]], "20201207": [v_name2idx["L2"]], "20210616": [v_name2idx["L2"]]},
    "m747680": {"20191108": [v_name2idx["L4"]]},
    "m778290": {"20151217": [v_name2idx["L2"]], "20211115": [v_name2idx["L3"], v_name2idx["L4"]]},
    "m782225": {"20200402": [v_name2idx["L3"]]},
    "m892796": {"20180420": [v_name2idx["L5"]], "20200224": [v_name2idx["L4"]]},
    "m894179": {"20200318": [v_name2idx["L2"]]},
    "m941952": {"20181113": [v_name2idx["L3"]], "20090921": [v_name2idx["L1"]]},
    "m967205": {"20190708": [v_name2idx["L3"]]},
}


### remap axial
remap_ctd: dict[str, list[str]] = {
    "m271612": ["20140825"],
    "m238139": ["20211213"],
    "m601667": ["20210311"],
    "m640268": ["20181108"],
    "m693352": ["20190301"],
    "m707324": ["20190909"],
    "m782225": ["20201013"],
    "m894179": ["20210929"],
    "m142267": ["20070531"],
}
# REDO
# "m153977": ["20180422"]x
# "m153977": ["20180606"]x
# "m238139": ["20210602"] x
# "m468624": ["20200311"], x
# m527202 20100118 x
# m527202 20160901 x
# m693352 20140204 x
# m693352 20190301 x
# m782225 20191223 x
# m782225 20200402

out_pk = Path("/media/data/robert/code/dae/datagen/fixed.pk")

if out_pk.exists():
    with open(out_pk, "rb") as handle:
        done = pickle.load(handle)
else:
    done: dict[str, list[str]] = {}


def validate(in_ds: Path, raw="rawdata", der="derivatives", sort=True):
    # INPUT
    in_ds = Path(in_ds)
    head_logger = No_Logger()  # (in_ds, log_filename="source-convert-to-unet-train", default_verbose=True)

    block = ""  # put i.e. 101 in here for block
    parent_raw = str(Path(raw).joinpath(str(block)))
    parent_der = str(Path(der).joinpath(str(block)))

    BIDS_Global_info.remove_splitting_key("chunk")
    bids_ds = BIDS_Global_info(datasets=[in_ds], parents=[parent_raw, parent_der], verbose=False)

    for name, subject in bids_ds.enumerate_subjects(sort=True):
        logger = head_logger.add_sub_logger(name=name)
        q = subject.new_query()
        q.flatten()
        q.filter("part", "inphase", required=False)
        q.filter("seg", lambda x: x != "manual", required=False)
        q.filter("lesions", lambda x: x != "manual", required=False)
        q.unflatten()
        q.filter_format("T2w")
        q.filter_filetype("nii.gz")
        q_sag = q.copy()
        q_sag.filter("acq", "sag")
        # q_sag.filter("desc", lambda _: False, required=False)

        q_ax = q.copy()
        q_ax.filter("acq", "iso")
        q_ax.filter("desc", "superres")

        sag = q_sag.loop_dict(sort=sort)
        ax = {next(iter(a.values()))[0].get("ses"): a for a in q_ax.loop_dict(sort=sort)}
        if len(ax) == 0:
            continue
        for f in sag:
            try:
                ses = next(iter(f.values()))[0].get("ses")
                ax_t2w = ax[ses]["T2w"][0]
                if name in done and ax_t2w.get("ses") in done[name]:
                    continue

                ax_subreg = ax[ses]["msk_seg-spine"][0]
                ax_vert = ax[ses]["msk_seg-vert"][0]
                if "ctd_seg-spine" in ax[ses]:
                    poi_ax_bids = ax[ses]["ctd_seg-spine"][0]
                    poi_ax = poi_ax_bids.open_poi()
                else:
                    print(ax[ses]["msk_seg-vert"])
                    print(ax[ses]["msk_seg-spine"])
                    poi_ax_bids = ax_subreg.get_changed_bids("json", format="ctd")
                    poi_ax = calc_centroids_from_subreg_vert(
                        ax_vert, ax_subreg, buffer_file=poi_ax_bids.file["json"], save_buffer_file=True
                    )
                # sag_t2w = f["T2w"][0]
                # sag_subreg = f["msk_seg-spine"][0]
                # sag_vert = f["msk_seg-vert"][0]
                poi_sag = f["ctd_seg-spine"][0]
                snps = list(get_snap_name(ax_t2w, der))
                snps[0].parent.mkdir(exist_ok=True)
                snps[1].parent.mkdir(exist_ok=True)
                if not snps[1].exists():
                    continue

                poi_ax_new, mapping = remap_centroids(poi_ax, poi_sag.open_poi())
                ax_vert_nii = None
                # ax_subreg_nii: NII = None  # type: ignore
                # print(name, name in remap_ctd, name in remap_ctd and ax_vert.get("ses") in remap_ctd[name])
                if name in remap_ctd and ax_vert.get("ses") in remap_ctd[name]:
                    if ax_vert_nii is None:
                        ax_vert_nii = ax_vert.open_nii()
                        # ax_subreg_nii = ax_subreg.open_nii()
                    poi_ax = poi_ax_new
                    ax_vert_nii.map_labels_(mapping)

                if name in remove_ctd and ax_vert.get("ses") in remove_ctd[name]:
                    print("A", name, name in remove_ctd, name in remove_ctd and ax_vert.get("ses") in remove_ctd[name])
                    if ax_vert_nii is None:
                        ax_vert_nii = ax_vert.open_nii()
                        # ax_subreg_nii = ax_subreg.open_nii()
                    rm_l = remove_ctd[name][ax_vert.get("ses")]
                    # ax_subreg_nii *= -ax_vert_nii.extract_label(rm_l) + 1
                    ax_vert_nii.remove_labels_(*rm_l)
                    for ver, i in poi_ax.copy().items_2D():
                        if ver not in rm_l:
                            continue
                        for sub in i:
                            print()
                            poi_ax = poi_ax.remove_centroid_((ver, sub))
                if ax_vert_nii is None:
                    continue
            except KeyError:
                continue  # logger.print("File Not Found:", f.family_id, e.args[0], Log_Type.FAIL)
            except gzip.BadGzipFile:
                logger.print("BadGzipFile:", f.family_id, Log_Type.FAIL)
                logger.print_error()
                exit()

            print("B", poi_ax)
            poi_ax.save(poi_ax_bids.file["json"], verbose=True)
            ax_vert_nii.save(ax_vert.file["nii.gz"])
            # (ax_subreg_nii * msk).save(ax_subreg.file["nii.gz"])
            snps[0].unlink()
            snps[1].unlink()

            if name not in done:
                done[name] = []
            done[name].append(ax_t2w.get("ses"))
            # mapping = {k: v for k, v in mapping.items() if k != v}
            # cord_poi = poi_ax_new.extract_subregion(Location.Spinal_Canal_ivd_lvl)
            # frame_0 = Snapshot_Frame(ax_t2w, segmentation=vertebra_level, centroids=cord_poi, hide_centroids=True, mode="MRI")
            # frame_1 = Snapshot_Frame(ax_t2w, segmentation=ax_vert, centroids=poi_ax_new.extract_subregion(50), coronal=True, mode="MRI")
            # frame_2 = Snapshot_Frame(ax_t2w, segmentation=ax_subreg, centroids=poi_ax.extract_subregion(50), coronal=True, mode="MRI")
            # frame_3 = Snapshot_Frame(ax_t2w, centroids=cord_poi, coronal=True, mode="MRI")
            # frame_4 = Snapshot_Frame(sag_t2w, segmentation=sag_vert, centroids=poi_sag, coronal=True, mode="MRI")
            # create_snapshot(snps, [frame_0, frame_1, frame_2, frame_3, frame_4])

        # for f in families:
        #    try:
        #        pass
        #    except Exception:
        #        logger.print_error()
        with open(out_pk, "wb") as handle:
            pickle.dump(done, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    ds = "/media/data/robert/test_paper4/"
    from script_axial_seg_no_stiching import validate as validate2

    validate2(Path(ds)) if True else None
    validate(Path(ds))
    validate2(Path(ds))
