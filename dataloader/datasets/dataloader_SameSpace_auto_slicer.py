import math
import sys
from pathlib import Path
import zipfile

from dataloader.datasets.transforms3D import ColorJitter3D

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))
from pathlib import Path
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from torch.nn import functional as F
from math import floor, ceil
import torch
import pandas
from torch import Tensor

import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
import math
from dataloader.datasets import fastnumpyio


class AutoCropDatasetSingleImage(Dataset):
    def __init__(
        self,
        buffer_slices: str,
        root: str,
        condition_types,
        target_shape,
        flip: bool = False,
        padding="reflect",  # constant, edge, reflect or symmetric
        mri_transform=transforms.Compose([ColorJitter3D(brightness_min_max=(0.8, 1.2), contrast_min_max=(0.8, 1.2))]),
        train=True,
        output_channels=1,
    ):
        assert Path(buffer_slices).parent.exists(), buffer_slices
        # assert ";" not in str(root), "Not yet supported"
        self.slice_buffer = Path(buffer_slices, ("train" if train else "val"))

        self.condition_types = [x for x in condition_types if x != "linespace"]
        self.output_channels = output_channels
        self.flip = flip
        self.padding = padding
        self.mri_transform = mri_transform
        self.train = train
        self.target_shape = target_shape
        self.count = 10

        #### MAKE FNIO ####
        if not self.slice_buffer.exists():
            print(self.slice_buffer)
            self.slice_buffer.mkdir(exist_ok=True, parents=True)
            self.root = Path(root)
            if ";" in str(root):
                self.df = pandas.concat([pandas.read_excel(r) for r in str(root).split(";")])
            elif self.root.is_file():
                if self.root.name.endswith(".csv"):
                    self.df = pandas.read_csv(root)
                else:
                    self.df = pandas.read_excel(root)
                self.root = self.root.parent
                if "resampled" in self.root.name:
                    self.root = self.root.parent
            else:
                self.df = pandas.read_csv(Path(self.root, "train.csv"))

            self.df = self.df[self.df["Phase"] == ("train" if train else "val")]
            self.df["num_slices"] = pandas.Series(dtype="int")
            assert len(self.df) != 0, self.df
            from joblib import Parallel, delayed

            def make_new(idx):
                niis = self.load_3D_file(idx, self.condition_types)
                if niis is None:
                    return
                for i in range(niis[0].shape[0]):
                    arr = np.stack([arr[i] for arr in niis])
                    assert len(arr.shape) == 3, (arr.shape, i)

                    fastnumpyio.save(Path(self.slice_buffer, f"{idx:09}_{i:03}.numpy"), arr)

            # def make_parr():
            Parallel(n_jobs=64)(delayed(make_new)(idx) for idx in range(len(self.df)))
            # self.l = list(self.slice_buffer.glob(f"*_*.numpy"))

            # import threading

            # t = threading.Thread(target=make_parr)
            # t.start()

        ###################

        self.l = list(self.slice_buffer.glob(f"*_*.numpy"))
        assert len(self.l) != 0, self.slice_buffer

    def load_3D_file(self, id, keys=["CT", "T2w"]):
        self.count -= 1

        from BIDS import NII

        id1 = id  # % len(self.df)
        out = []
        files = []
        for key in keys:
            # f = #self.root /
            assert isinstance(self.df.iloc[id1][key], str) or not math.isnan(self.df.iloc[id1][key]), "detected empty entry\n" + str(
                self.df.iloc[id1]
            )
            p = self.df.iloc[id1]["Path"]
            if not str(p).startswith("/"):
                p = Path(self.root, p)
            f = Path(p, self.df.iloc[id1][key])
            if not f.exists():
                f = str(f).replace("rawdata", "derivatives")
            if not Path(f).exists():
                f = str(f).replace("derivatives", "rawdata")
            if not Path(f).exists():
                f = str(f).replace("rawdata", "resampled")
            if not Path(f).exists():
                f = str(f).replace("rawdata", "translated/rawdata")
            if not Path(f).exists():
                f = str(f).replace("rawdata", "translated/derivatives")
            if not Path(f).exists():
                return None
                # assert False, f"{f} does not exit"
            # print(f)
            nii: nib.Nifti1Image = nib.load(str(f))  # type: ignore
            aff = nii.affine
            ornt_fr = nio.io_orientation(aff)
            try:
                arr: np.ndarray = nii.get_fdata()
            except EOFError:
                print("EOF-ERROR", f)
                Path(f).unlink()
                return None  # self.load_file(id + 1, keys=keys)
            except Exception as e:
                print(f)
                print(f)
                print(f)
                print(e)
                print(f)
                print(f)
                print(f)
                return None  # self.load_file(id + 1, keys=keys)
            ornt_to = nio.axcodes2ornt(("R", "I", "P"))
            ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)

            arr = nio.apply_orientation(arr, ornt_trans)
            aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
            new_aff = np.matmul(aff, aff_trans)
            from BIDS import NII

            if len(out) == 0:
                inital = nib.Nifti1Image(arr, new_aff)
                print("Inital", NII(inital, False))
            elif arr.shape != out[0].shape:
                print("resampled", arr.shape, inital.shape, "\n", files)
                nii = nip.resample_from_to(nib.Nifti1Image(arr, new_aff), inital, order=3, cval=0)
                arr = nii.get_fdata()
            if str(key).upper() == "CT":
                arr /= 1000
                arr = arr.clip(-1, 1)
            else:
                print(f) if np.max(arr) == 0 else None
                arr = arr / np.max(arr)
                arr = arr.clip(0, 1e6)

            out.append(arr)
            files.append(f)

        return out

    def load_file_npz(self, name, keys):
        dict_mods = []
        if str(name).endswith(".npz"):
            try:
                f = np.load(name)
            except zipfile.BadZipFile:
                print(name)
                return None
            for k in keys:  # type: ignore
                dict_mods.append(torch.from_numpy(f[k].astype("f").copy()))

            f.close()  # type: ignore
            return dict_mods
        else:
            arr = fastnumpyio.load(name).astype("f")  # .swapaxes(-1, -2)
            if arr.max() >= 1:
                arr /= float(arr.max())
            return [arr[i] for i in range(arr.shape[0])]

        assert False, "Expected a .npz file"

    def load_file(self, id, keys=["CT", "T2w"]):
        return self.load_file_npz(self.l[id], keys)

    @torch.no_grad()
    def transform(self, items_in):
        # Transform to tensor
        items = map(Tensor, items_in)

        ## Padding
        items = list(map(lambda x: pad_size(x, self.target_shape[-2:], self.padding), items))
        # Coordinate-encoding
        # shape = items[0].shape
        # l1 = np.tile(np.linspace(0, 1, shape[0]), (shape[1], shape[2], 1))
        # l2 = np.tile(np.linspace(0, 1, shape[1]), (shape[0], shape[2], 1))
        # l3 = np.tile(np.linspace(0, 1, shape[2]), (shape[0], shape[1], 1))
        # l1 = Tensor(l1).permute(2, 0, 1)
        # l2 = Tensor(l2).permute(0, 2, 1)
        # l3 = Tensor(l3)
        # assert l1.shape == l2.shape, (l1.shape, l2.shape)
        # assert l3.shape == l2.shape, (l3.shape, l2.shape)
        # assert shape == l2.shape, (shape, l2.shape)
        # items.append(l1)
        # items.append(l2)
        # items.append(l3)
        ## Random crop
        items = list(random_crop(self.target_shape[-2:], *items))

        for i, (x, y) in enumerate(zip(items, self.condition_types)):
            if y in ["MRI", "T1", "t1", "T2", "t2", "T1GD", "FLAIR", "water", "fat", "T1w", "T2w"]:
                items[i] = self.mri_transform(x)

            if y.upper() != "CT":
                items[i] = items[i] * 2 - 1

        # Random flipping
        if self.flip and random.random() > 0.5:
            self.spacial_flip.update()
            items = list(map(self.spacial_flip, items))
        out = tuple(a.to(torch.float32).unsqueeze_(0) for a in items)
        try:
            if self.output_channels == 1:
                return out[0], torch.cat(out[1:], 0)
            else:
                return torch.cat(out[: self.output_channels], 0), torch.cat(out[self.output_channels :], 0)
        except:
            print([o.shape for o in out])
            print([o.shape for o in out])
            exit()

    def __getitem__(self, index):
        index = index % len(self)
        # if index in self.blacklist:
        #    return self.__getitem__(random.randint(0, len(self)))
        try:
            list_of_items = self.load_file(index, self.condition_types)
            if list_of_items is None:
                self.l.remove(self.l[index])
                return self.__getitem__(random.randint(0, len(self)))
            for i in list_of_items:
                if len(i.shape) == 1:
                    print(index, "wrong shape", [i.shape for i in list_of_items])
                    self.l[index].unlink()
                    self.l.remove(self.l[index])

                    return self.__getitem__(random.randint(0, len(self)))
            out = self.transform(list_of_items)
            # print(out[0].shape, out[1].shape)
            return out

        except Exception as e:
            self.l.remove(self.l[index])
            # if 'buffer is too small for requested array' in str(e):
            print(index, str(e))
            # print("ERRRRROR", self.df.iloc[index])
            raise e

            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.l)
        # if self._len is None:
        #    self._len = max(self._len_df, int(sum(self.lens_dict.values())))
        # return self._len
