from dataclasses import dataclass
from pathlib import Path

from TPTBox import BIDS_FILE, BIDS_Global_info, No_Logger


@dataclass()
class session:
    axial_stiched: BIDS_FILE | None
    iso_stiched: BIDS_FILE | None
    axial_chunks: list[BIDS_FILE]
    sag: BIDS_FILE | None

    def get_seg(self, x: BIDS_FILE):
        q = x.get_sequence_files().new_query(flatten=True)
        q.filter("acq", x.get("acq", lambda _: False), required=False)  # type: ignore
        q.filter("desc", x.get("desc", lambda _: False), required=x.get("desc", None) is not None)  # type: ignore
        q.filter("chunk", x.get("chunk", lambda _: False), required=x.get("chunk", None) is not None)  # type: ignore
        q.unflatten()
        l_files = list(q.loop_dict())
        assert len(l_files) == 1, l_files
        return l_files[0]

    def get_sag_stiched_seg(self):
        assert self.sag is not None
        return self.get_seg(self.sag)

    def get_iso_stiched_seg(self):
        assert self.iso_stiched is not None
        return self.get_seg(self.iso_stiched)

    def get_axial_chunks_seg(self):
        assert len(self.axial_chunks) != 0
        return [self.get_seg(s) for s in self.axial_chunks]


def get_sessions(in_ds: Path | str, raw="rawdata", der="derivatives", other_folders=(), sort=True, require_sag=True, require_stiched=True):
    # INPUT
    in_ds = Path(in_ds)
    head_logger = No_Logger()
    block = ""  # put i.e. 101 in here for block
    parent_raw = str(Path(raw).joinpath(str(block)))
    parent_der = str(Path(der).joinpath(str(block)))
    # check available models
    BIDS_Global_info.remove_splitting_key("chunk")
    BIDS_Global_info.remove_splitting_key("acq")
    bids_ds = BIDS_Global_info(datasets=[in_ds], parents=[parent_raw, parent_der, der, *other_folders], verbose=False)
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

        families = q.loop_dict(sort=sort, key_addendum=["acq", "desc"])
        for f in families:
            try:
                assert "T2w_acq-ax" in f, f
                ax = f["T2w_acq-ax"]
                if len(ax) == 1:
                    ax_s = ax[0]
                elif require_stiched:
                    assert "T2w_acq-ax_desc-stitched" in f, f
                    assert len(f["T2w_acq-ax_desc-stitched"]) == 1, f["T2w_acq-ax_desc-stitched"]
                    ax_s = f["T2w_acq-ax_desc-stitched"][0]
                else:
                    ax_s = None
                if "T2w_acq-sag" in f and len(f["T2w_acq-sag"]) == 1:
                    sag = f["T2w_acq-sag"][0]
                else:
                    if require_sag:
                        assert "T2w_acq-sag_desc-stitched" in f, f
                        assert len(f["T2w_acq-sag_desc-stitched"]) == 1, f["T2w_acq-sag_desc-stitched"]
                    sag = f["T2w_acq-sag_desc-stitched"][0] if "T2w_acq-sag_desc-stitched" in f else None
                if require_stiched:
                    assert "T2w_acq-iso_desc-superres" in f, f
                    assert len(f["T2w_acq-iso_desc-superres"]) == 1, f["T2w_acq-iso_desc-superres"]
                    iso = f["T2w_acq-iso_desc-superres"][0]
                else:
                    iso = None
                yield session(ax_s, iso, ax, sag)
            except Exception:
                logger.print_error()
                # raise
