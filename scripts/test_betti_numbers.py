from TPTBox import BIDS_Global_info, Logger

log = Logger("/DATA/NAS/datasets_processed/MRI_spine/dataset-MSFU/", " betti_numbers", default_verbose=True)
log.print(
    "Betti Numbers: \n\
        - B0 (b0): Number of connected components.\n\
        - B1 (b1): Number of holes.\n\
        - B2 (b2): Number of fully engulfed empty spaces.\n"
)
if __name__ == "__main__":
    bgi = BIDS_Global_info(["/DATA/NAS/datasets_processed/MRI_spine/dataset-MSFU/"])
    for _, sub in bgi.iter_subjects():
        q = sub.new_query(flatten=True)
        q.filter("seg", "vert")
        q.filter("desc", "superres")
        for f in q.loop_list():
            nii = f.open_nii()
            nii.clamp_(0, 50)
            nii.remove_labels_(50)
            nii.remove_labels_(int(nii.max()))
            nii.remove_labels_(1)
            log.print("Betti number", f)
            betti = nii.betti_numbers(verbose=True)
            sus = {k: v for k, v in betti.items() if v != (1, 1, 0)}
            log.print(sus)
            log.flush()
