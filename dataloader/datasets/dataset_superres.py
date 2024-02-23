import random
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from degrade.degrade import fwhm_needed, fwhm_units_to_voxel_space
from resize.pytorch import resize
from torchvision import transforms

from dataloader.datasets.dataset_csv import Dataset_CSV
from dataloader.datasets.dataset_utils import calc_extended_patch_size, parse_kernel, target_pad
from utils.arguments import DataSet_Option

mri_transform = transforms.Compose([transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2))])


class Dataset_CSV_super(Dataset_CSV):
    """Argumentation like ins SMORE"""

    def __init__(
        self, opt: DataSet_Option, path, transform, split: None | Literal["train", "val", "test"] = None, col="file_path", label=None
    ):
        super().__init__(path, transform, split, col, label=label)
        print(path)
        if opt.sr_source is None:
            opt.sr_source = [5, 5.2, 6.2, 6, 5.8, 7.6, 7.2, 8, 7, 10]
        resolutions = [(opt.sr_target, s) for s in opt.sr_source]
        self.opt = opt
        self.hflip = True
        self.vflip = True
        self.t_flip = True
        self.elastic = 0.5
        if self.elastic != 0:
            self.elastic_trans = torchvision.transforms.ElasticTransform([1.0, 10.0], sigma=[0.1, 3.0])  # type: ignore
        self.kernel = []
        for hr, lr in resolutions:
            # Model the blurring-effect of a LR slice (lr x lr) compared to (hr,hr)
            # Axial images 0.5 mm x 0.5 mm x 5-6 mm
            # Sagittal (1.5 T) images 1.1 mm x 1.1 mm x 2-4 mm
            # Sagittal (3 T) images 0.95-0.8 mm x 0.95-0.8 mm x 2-4 mm
            blur_fwhm = fwhm_units_to_voxel_space(fwhm_needed(hr, lr), hr)
            slice_separation = float(lr / hr)
            self.kernel.append((parse_kernel(None, "rf-pulse-slr", blur_fwhm), slice_separation))
        self.patch_size = (self.opt.img_size, self.opt.img_size)
        self.slice_separation_strength = 0.7
        self.noise = True

    def __len__(self):
        return len(self.dataset)

    def get_img(self, index, c=None):
        patch_hr = self.load_img(index)
        patch_hr = np.array(patch_hr).astype(np.float32) / 255
        # patch_hr = self.transform(patch_hr)
        if c is None:
            blur_kernel, slice_separation = self.kernel[random.randint(0, len(self.kernel) - 1)]
        else:
            blur_kernel, slice_separation = self.kernel[c]
        ext_patch_size, ext_patch_crop = calc_extended_patch_size(blur_kernel, self.patch_size)

        # apply the pad
        patch_hr, pads = target_pad(patch_hr, ext_patch_size, mode="reflect")

        patch_hr = torch.from_numpy(patch_hr)
        patch_hr = patch_hr.unsqueeze(0).unsqueeze(1)
        patch_lr = F.conv2d(patch_hr, blur_kernel, padding="same")
        if self.elastic != 0 and self.elastic <= random.random():
            patch_lr = self.elastic_trans(patch_lr)
        assert patch_hr.max() <= 1.0, patch_hr.max()
        assert patch_hr.min() >= 0.0, patch_hr.min()
        ext_patch_crop = (slice(None, None), slice(None, None), *ext_patch_crop)
        patch_hr = patch_hr[ext_patch_crop]
        patch_lr = patch_lr[ext_patch_crop]
        # patch_lr2 = patch_lr.clone()
        slice_separation *= random.random() + self.slice_separation_strength
        patch_lr: torch.Tensor = resize(patch_lr, (slice_separation, 1), order=3)  # type: ignore
        if self.noise:
            patch_lr += torch.rand_like(patch_lr) * (random.randint(0, 15) / 100.0)
        patch_lr: torch.Tensor = resize(patch_lr, (1 / slice_separation, 1), order=3)  # type: ignore order=random.choice([0, 1, 3])
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(patch_hr, output_size=self.patch_size)  # type: ignore
        patch_hr = TF.crop(patch_hr, i, j, h, w)
        patch_lr = TF.crop(patch_lr, i, j, h, w)
        if self.noise:
            patch_lr += torch.rand_like(patch_lr) * (random.randint(0, 5) / 100.0)

        # patch_lr2 = TF.crop(patch_lr2, i, j, h, w)
        patch_hr, patch_lr = self.final_transforms(patch_hr, patch_lr)
        # patch_lr2 = patch_lr2.squeeze(0)
        out = {"img": patch_hr, "img_lr": patch_lr}  # , "index": target, "cls_labels": target}
        self.add_label(out, index)
        return out

    def final_transforms(self, patch_hr, patch_lr):
        # Random horizontal flipping
        if self.hflip and random.random() > 0.5:
            patch_hr = TF.hflip(patch_hr)
            patch_lr = TF.hflip(patch_lr)

        # Random vertical flipping
        if self.vflip and random.random() > 0.5:
            patch_hr = TF.vflip(patch_hr)
            patch_lr = TF.vflip(patch_lr)

        if self.t_flip and random.random() > 0.5:
            patch_hr = patch_hr.swapaxes(-1, -2)
            patch_lr = patch_lr.swapaxes(-1, -2)

        # Normalize to -1, 1
        patch_hr = patch_hr * 2 - 1
        patch_lr = patch_lr * 2 - 1
        # patch_lr2 = patch_lr2 * 2 - 1

        patch_hr = patch_hr.squeeze(0)
        patch_lr = patch_lr.squeeze(0)
        return patch_hr, patch_lr

    def __getitem__(self, index):
        return self.get_img(index)

    def get_extended_info(self, index):
        return self.dataset.iloc[index], *self.__getitem__(index)
