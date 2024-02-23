from pathlib import Path

from BIDS import BIDS_Global_info

bgi = BIDS_Global_info(datasets=["/DATA/NAS/datasets_processed/MRI_spine/dataset-spider/"], parents=["rawdata"])
idx = 0
for _, sub in bgi.iter_subjects():
    q = sub.new_query()
    q.filter("acq", "iso")
    q.flatten()
    for a in q.loop_list(sort=True):
        idx += 1
        f_lr = Path(f"/DATA/NAS/ongoing_projects/robert/datasets/dataset-spider-iso/lr/{a.BIDS_key}.nii.gz")
        # f_stop_gap_iso = Path(f"/DATA/NAS/ongoing_projects/robert/datasets/dataset-spider-iso/iso_integrated/{a.BIDS_key}.nii.gz")
        f_iso = Path(f"/DATA/NAS/ongoing_projects/robert/datasets/dataset-spider-iso/iso/{a.BIDS_key}.nii.gz")
        # if not f_stop_gap.exists() or True:
        out = a.open_nii().reorient_()  # .rescale_((-1, -1, 0.8571))
        out = out.set_dtype_()
        out -= out.min()
        out /= out.max()

        out_iso = out.rescale((0.8571, 0.8571, -1))
        out_iso.save(f_iso)
        r, i, p = out.zoom
        arr = out.get_array()
        out.rescale_((0.8571, 5, -1))
        out.save(f"/DATA/NAS/ongoing_projects/robert/datasets/dataset-spider-iso/ax/{a.BIDS_key}.nii.gz")
        # arr2 = out.get_array() * 0
        # height = 0
        # start = 0
        # idx2 = 0
        # for idx in range(arr.shape[-2]):
        #    height += p
        #    if height >= 5.0:
        #        height -= 5.0
        #        start = idx
        #    # print(height, p)
        #    if height >= 4.0 and height - p < 4.0:
        #        arr2[..., idx2, :] = arr[..., start:idx, :].sum(-2) / (idx - start + 1)
        #        idx2 += 1
        #        if idx2 == arr2.shape[-2]:
        #            break
        #        # print(arr2[..., idx2, :].sum())
        ## arr2[..., idx2, :] = arr[..., start:, :].sum(-2) / (start - arr.shape[-2])
        # out.set_array_(arr2).rescale_((-1, 5, 0.8571)).save(f_stop_gap)
        # else:
        #    out = NII.load(f_stop_gap, False)
        #    out_iso = NII.load(f_iso, False)

        out.resample_from_to_(out_iso).clamp_(0, 1).save(f_lr)
        print(out.min(), out.max())
