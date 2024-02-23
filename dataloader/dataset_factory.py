from multiprocessing import get_context
from pathlib import Path
from typing import Literal

from dataloader.datasets.dataset_csv import Dataset_CSV
from dataloader.datasets.dataset_inpainting import Dataset_CSV_inpainting
from dataloader.transforms import get_transforms2D
from utils.arguments import CycleGAN_Option, DataSet_Option


def get_num_channels(opt: DataSet_Option):  # noqa: ARG001
    return 1


SPLIT = Literal["train", "val", "test"]


def get_dataset(opt: DataSet_Option, split: SPLIT = "train", super_res=False, label=None):
    print(opt.__class__)
    ### CycleGAN ################################
    if isinstance(opt, CycleGAN_Option):
        from dataloader import dataset_factory_cycleGAN

        ds = dataset_factory_cycleGAN.get_dataset(opt, split, label)
        if ds is not None:
            return ds
        if opt.super_res:
            super_res = True
    ####################################

    ds = _get_dataset(opt, split, super_res, label=label)
    if opt.inpainting is not None:
        ds = Dataset_CSV_inpainting(opt, ds)  # type: ignore
    return ds


def get(obj, name, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    return default


def _get_dataset(opt: DataSet_Option, split: SPLIT = "train", super_res=False, label=None):
    dataset = opt.dataset

    if opt.dataset_val is not None and split == "val":
        dataset = opt.dataset_val
    if Path("/DATA/NAS/ongoing_projects/robert/train_dataset/neuropoly").exists():
        dataset = dataset.replace(
            "/media/data/robert/datasets/dataset-neuropoly/training_img/", "/DATA/NAS/ongoing_projects/robert/train_dataset/neuropoly/"
        )
    ds_type = opt.ds_type

    if ds_type == "csv_2D":
        transf1 = get_transforms2D(opt, split)
        if super_res or "img_lr" in opt.palette_condition:
            from dataloader.datasets.dataset_superres_stop_gap import Dataset_CSV_super

            return Dataset_CSV_super(opt=opt, path=dataset, transform=transf1, split=split, label=label)
        return Dataset_CSV(path=dataset, transform=transf1, split=split, label=label)
    if ds_type == "csv_2D_super":
        if super_res or "img_lr" in opt.palette_condition or get(opt, "image_name") == "img_lr":
            from dataloader.datasets.dataset_superres_v2 import Dataset_CSV_super_v2

            return Dataset_CSV_super_v2(opt=opt, path=dataset, split=split, label=label)
        transf1 = get_transforms2D(opt, split)
        return Dataset_CSV(path=dataset, transform=transf1, split=split, label=label)
    if ds_type == "csv_2D_npz":
        from dataloader.datasets.dataset_csv_npz_paired import Dataset_CSV_NPZ_paired

        return Dataset_CSV_NPZ_paired(dataset, opt, split=split)
    raise NotImplementedError(ds_type)


def get_data_loader(
    opt: DataSet_Option,
    dataset,
    shuffle: bool,
    drop_last: bool = True,
    parallel=False,
    split: SPLIT = "train",  # FIXME always False?
):
    from torch import distributed
    from torch.utils.data import DataLoader, WeightedRandomSampler

    if parallel and distributed.is_initialized():
        # drop last to make sure that there is no added special indexes
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
    elif hasattr(dataset, "sample_weights") and split == "train" and opt.train_mode.is_manipulate():  # type: ignore
        print("using weighted sampler for imbalanced data")
        sampler = WeightedRandomSampler(dataset.sample_weights(), len(dataset))
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=sampler,
        # with sampler, use the sample instead of this option
        shuffle=False if sampler else shuffle,
        num_workers=opt.num_cpu,
        pin_memory=True,
        drop_last=drop_last,
        multiprocessing_context=get_context("fork"),
        persistent_workers=True,
    )
