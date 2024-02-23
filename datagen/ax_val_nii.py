import pickle
import random

from BIDS import BIDS_Global_info

niis = []
with open("/media/data/robert/code/dae/datagen/subject_spit.pk", "rb") as handle:
    a = pickle.load(handle)

    a = [k for k, v in a.items() if v == 1]
    print(len(a))
    bgi = BIDS_Global_info(["/media/data/robert/datasets/dataset-neuropoly"], verbose=False)
    l = 0
    coin = True
    for name, sub in bgi.iter_subjects():
        if name not in a:
            continue
        coin = not coin
        if coin:
            continue
        q = sub.new_query(flatten=True)
        q.filter("acq", "ax")
        q.filter_format("T2w")
        q.filter_non_existence("seg")
        q.filter_non_existence("lesions")
        q.filter_filetype("nii.gz")
        lis = [a for a in q.loop_list() if "stitched" not in a.BIDS_key]
        l += len(lis)
        # print(l,len(list(q.loop_list())))
        for x in lis:
            print(x)
        if len(lis) == 0:
            continue
        niis.append(str(random.choice(lis).file["nii.gz"]))

    print(l, len(niis))

    print(niis)
