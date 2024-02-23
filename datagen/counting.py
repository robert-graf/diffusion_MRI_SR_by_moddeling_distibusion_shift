import pickle

from BIDS import BIDS_Global_info

niis = []
with open("/media/data/robert/code/dae/datagen/subject_spit.pk", "rb") as handle:
    a = pickle.load(handle)

    a = [k for k, v in a.items() if v == 2]
    print(a, len(a))
    bgi = BIDS_Global_info(["/media/data/robert/datasets/dataset-neuropoly"], parents=["rawdata"], verbose=False)
    number_of_axial_images = 0
    number_of_sag_images = 0
    number_of_ses_images = 0
    for name, sub in bgi.iter_subjects():
        if name not in a:
            continue
        #### CHUNKS ###
        q = sub.new_query(flatten=True)
        q.filter("acq", "ax")
        q.filter_format("T2w")
        q.filter_non_existence("seg")
        q.filter_non_existence("lesions")
        q.filter_filetype("nii.gz")
        lis = [a for a in q.loop_list() if "stitched" not in a.BIDS_key]
        ses = {a.get("ses") for a in lis}
        number_of_axial_images += len(lis)
        number_of_ses_images += len(ses)
        q = sub.new_query(flatten=True)
        q.filter("acq", "sag")
        q.filter_format("T2w")
        q.filter_non_existence("seg")
        q.filter_non_existence("lesions")
        q.filter_filetype("nii.gz")
        lis = [a for a in q.loop_list() if "stitched" not in a.BIDS_key]
        number_of_sag_images += len(lis)
    print("number_of_axial_images", number_of_axial_images, "    ")
    print("number_of_sag_images", number_of_sag_images)
    print("number_of_ses_images", number_of_ses_images)
