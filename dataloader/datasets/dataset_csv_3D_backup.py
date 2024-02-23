import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datagen import fastnumpyio
from dataloader.datasets import transforms3D
from dataloader.datasets.dataset_utils import apply_target_pad, calc_target_pad
from utils.arguments import CycleGAN_Option, DataSet_Option

flips = [
    # (-1, -2),
    # (-1, -3),
    (-3, -2),
    # (-3, -1),
]


class Dataset_CSV_3D(Dataset):
    def __init__(self, opt: DataSet_Option, split: None | Literal["train", "val", "test"] = None, col="file_path", unpaired=False):
        assert opt.dims == 3, opt.dims
        # num_imgs = len(opt._names) - (opt.dims if opt.linspace else 0)
        self.unpaired = unpaired
        self.datasets = {}
        path = Path(opt.dataset)
        if path.is_dir():
            for name in opt._names:
                if "linspace" in name:
                    continue
                n = list(path.glob(f"*{name}*.xlsx"))
                assert len(n) == 1, (path, [p.name for p in path.iterdir()])
                df = pd.read_excel(n[0])  # noqa: PD901
                df = df.loc[df["Split"] == split]  # noqa: PD901
                df.reset_index()
                self.datasets[name] = df
        else:
            raise NotImplementedError()
        self.linspace = opt.linspace
        self.dims = opt.dims
        self.size = opt.shape
        self.len = min([len(x) for x in self.datasets.values()])
        self.opt = opt
        self.col = col
        self.colorJitter = transforms.Compose([transforms3D.ColorJitter3D(brightness_min_max=(0.8, 1.2), contrast_min_max=(0.8, 1.2))])

        print(self.__len__(), "samples")
        assert self.__len__() != 0, self.datasets

    def __len__(self):
        return self.len

    def load_img(self, key, index: int):
        row: str = self.datasets[key].iloc[index][self.col]
        if row.endswith(".fnio"):
            img = fastnumpyio.load(row).astype(np.float32)
        else:
            raise NotImplementedError()
        if key == "ct":
            img += 1024
            img /= 2048
            img = np.clip(img, 0, 1)
        if key in ("msk", "seg"):
            pass
        else:
            img -= img.min()
            img /= img.max()
        assert img.max() <= 1.0, img.max()
        assert img.min() >= 0.0, img.min()
        return img

    def get_rand_idx(self, key):
        index = random.randint(0, len(self.datasets) - 1)
        shape_org = self.load_img(key, index).shape
        return index, shape_org

    @torch.no_grad()
    def __getitem__(self, index):
        if self.unpaired:
            return self.get_unpaired()
        return self.get_paired()

    def get_unpaired(self):
        assert isinstance(self.opt, CycleGAN_Option)
        out = {}
        out = self.load_group(out, self.opt.side_a[0], self.opt.side_a, "_A_")
        out = self.load_group(out, self.opt.side_b[0], self.opt.side_b, "_B_")
        return out

    def get_paired(self):
        out = {}
        keys = list(self.datasets.keys())
        return self.load_group(out, keys[0], keys)

    def load_group(self, out, first_key, keys, linspace_name_addendum=""):
        index, shape_org = self.get_rand_idx(first_key)

        crop = transforms3D.calc_random_crop3D(self.size, shape_org)  # type: ignore
        pad = calc_target_pad(shape_org, self.size)

        for key in keys:
            if "linspace" in key:
                continue
            assert key not in out
            img: np.ndarray = self.load_img(key, index)
            assert shape_org == img.shape, (shape_org, img.shape)
            img = apply_target_pad(img, pad, mode="reflect")
            img_t = torch.from_numpy(img)
            (img_t,) = transforms3D.apply_random_crop3D(crop, img_t)
            if key.lower() != "ct" or key.lower() in ["seg", "msk", "linspace"]:
                img_t = self.colorJitter(img_t)
            out[key] = img_t * 2 - 1

        if self.linspace:
            transforms3D.add_linspace_embedding(out, shape_org, dim=self.dims, crop=crop, pad=pad, name_addendum=linspace_name_addendum)
        self.final_transforms(out)
        return out

    def final_transforms(self, out: dict[str, torch.Tensor]):
        if random.random() > 0.5:
            r = random.randint(0, len(flips) - 1)
            for k, img in out.items():
                out[k] = img.swapaxes(*flips[r])
        for k, img in out.items():
            if len(img.shape) == 3:
                img = img.unsqueeze_(0)
            if len(img.shape) == 5:
                img = img.squeeze_(0)
            out[k] = img
