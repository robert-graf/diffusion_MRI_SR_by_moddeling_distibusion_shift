import sys

sys.path.append("..")

from dataloader.dataset_factory import SPLIT
from utils.arguments import DataSet_Option


def get_dataset(opt: DataSet_Option, split: SPLIT = "train", label=None):
    if opt.ds_type == "csv_3D_unpaired":
        from .datasets.dataset_csv_3D import Dataset_CSV_3D

        return Dataset_CSV_3D(opt=opt, split=split, unpaired=True)
    elif opt.ds_type == "csv_3D_paired":
        from .datasets.dataset_csv_3D import Dataset_CSV_3D

        return Dataset_CSV_3D(opt=opt, split=split, unpaired=False)
    return None
