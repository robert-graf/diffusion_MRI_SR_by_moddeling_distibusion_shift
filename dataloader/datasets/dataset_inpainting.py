import random
import numpy as np
from torch.utils.data import Dataset

from dataloader.datasets.dataset_csv import Dataset_CSV
from utils.arguments import DataSet_Option
from utils.enums import Inpainting


class Dataset_CSV_inpainting(Dataset):
    """Argumentation like ins SMORE"""

    def __init__(self, opt: DataSet_Option, dataset: Dataset_CSV):
        self.opt = opt
        self.dataset = dataset
        self.inpainting = opt.inpainting
        self.size = self.opt.img_size
        self.shape = self.opt.shape

    def __len__(self):
        return len(self.dataset)

    def get_img(self, index, c=None):
        dict_images = self.dataset[index]
        img = dict_images["img"]
        if self.inpainting == Inpainting.random_ege:
            # print('inpainting - random_ege')
            mask: Tensor = np.ones_like(img)  # type: ignore
            assert self.size != 0
            if random.random() < 0.33:
                mask_height = int((random.random() * 0.3 + 0.1) * self.size) + 1
                mask[..., :mask_height] = 0
            if random.random() < 0.33:
                mask_height = int((random.random() * 0.3 + 0.1) * self.size) + 1
                mask[..., -mask_height:] = 0
            if random.random() < 0.33:
                mask_height = int((random.random() * 0.3 + 0.1) * self.size) + 1
                mask[..., :mask_height, :] = 0
                mask_height = int((random.random() * 0.3 + 0.1) * self.size) + 1
            if random.random() < 0.33:
                mask_height = int((random.random() * 0.3 + 0.1) * self.size) + 1
                mask[..., -mask_height:, :] = 0
            dict_images["mask"] = mask
            return dict_images
        elif self.inpainting == Inpainting.perlin:
            assert self.size != 0
            assert self.opt.dims == 2, "perlin is implemented only for 2D"
            import dataloader.datasets.perlin as perlin

            if random.random() > 0.5:
                dict_images["mask"] = perlin.rand_perlin_2d_mask(self.shape, random.choice([2, 4, 8, 16]), (0.4, 0.7)).unsqueeze(0).numpy()
            else:
                dict_images["mask"] = np.ones_like(img)  # type: ignore
            return dict_images
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        return self.get_img(index)

    def get_extended_info(self, index):
        return self.dataset.get_extended_info(index)[0], *self.__getitem__(index)
