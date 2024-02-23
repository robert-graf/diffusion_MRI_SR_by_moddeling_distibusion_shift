from typing import Literal

import numpy as np
import torchvision.transforms as transforms
from torchvision import transforms

from dataloader.datasets.dataset_csv import Dataset_CSV
from dataloader.transforms import get_paired_transform_by_list
from utils.arguments import DataSet_Option

mri_transform = transforms.Compose([transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2))])


class Dataset_CSV_super_v2(Dataset_CSV):
    def __init__(self, opt: DataSet_Option, path, split: None | Literal["train", "val", "test"] = None, col="file_path", label=None):
        train = split == "train"
        self.transform = get_paired_transform_by_list(opt, train, jpg=True)
        super().__init__(path, self.transform, split, col, label=label)
        print(path)
        self.opt = opt

    def __len__(self):
        return len(self.dataset)

    def get_img(self, index, c=None):
        patch_hr = np.array(self.load_img(index))
        # patch_hr = patch_hr).astype(np.float32) / 255
        # patch_hr = torch.from_numpy(patch_hr).unsqueeze(0)
        out = self.transform({"hq": patch_hr})
        assert "img_lr" in out, out.keys()
        # self.final_transforms(out)
        self.add_label(out, index)

        return out

    def __getitem__(self, index):
        return self.get_img(index)

    def get_extended_info(self, index):
        return self.dataset.iloc[index], *self.__getitem__(index)
