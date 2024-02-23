import pickle

from BIDS import BIDS_FILE, BIDS_Global_info

niis = []
with open("/media/data/robert/code/dae/datagen/subject_spit.pk", "rb") as handle:
    a = pickle.load(handle)

a = [k for k, v in a.items() if v == 2]
bgi = BIDS_Global_info(
    ["/media/data/robert/datasets/dataset-neuropoly"],
    parents=["rawdata", "rawdata_upscale", "derivatives", "derivatives_seg"],
    verbose=False,
)
count = 0
print()
for name, sub in bgi.iter_subjects():
    if name not in a:
        continue
    #### CHUNKS ###
    q = sub.new_query(flatten=True)
    # q.filter("acq", "ax")
    # q.filter("chunk", lambda _: True)
    q.filter_format(["T2w"])
    q.filter("seg", lambda x: x != "manual", required=False)
    q.filter_non_existence("lesions")
    # q.filter_filetype("nii.gz")

    def test(a: BIDS_FILE):
        if "acq-sag" in a.BIDS_key and "chunk" in a.BIDS_key:
            return False
        if "acq-sag" in a.BIDS_key and "chunk" in a.BIDS_key:
            return False
        return True

    lis = [a for a in q.loop_list() if test(a)]
    ses = []

    count += len(lis)
    for x in lis:
        x.save_changed_path(parent=x.get_parent().split("_")[0], dataset_path="/media/data/robert/test_paper4/")
        print(x)
    if len(lis) == 0:
        continue
    print(lis)
    # exit()
    # q = sub.new_query(flatten=True)
    # q.filter("acq", "ax")
    # q.filter("desc", "stitched")
    # q.filter_format("T2w")
    # q.filter_non_existence("seg")
    # q.filter_non_existence("lesions")
    # q.filter_filetype("nii.gz")
    # lis_stitched = list(q.loop_list())
    # for i in lis_stitched:
    #    if i.get("ses") in ses:
    #        out = i.get_changed_path(parent="rawdata", dataset_path="/media/data/robert/datasets/dataset-neuropoly-test/")
    #        i.open_nii().save(out)
    # niis.append(str(random.choice(lis).file["nii.gz"]))
    print(count, len(niis))
    print(niis)
