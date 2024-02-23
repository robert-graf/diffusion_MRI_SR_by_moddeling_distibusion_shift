import random
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset

from dataloader.datasets.dataset_superres import target_pad
from utils.arguments import DataSet_Option


class Dataset_CSV_NPZ_paired(Dataset):
    def __init__(self, path, opt: DataSet_Option, split: None | Literal["train", "val", "test"] = None, col="file_path", label=None):
        print(path)
        dataset = pd.read_excel(path) if not str(path).endswith(".csv") else pd.read_csv(path)
        self.opt = opt
        assert col in dataset, dataset
        if label is not None:
            unique_labels = sorted(set(dataset[label].to_list()))
            if any(isinstance(ul, str) for ul in unique_labels):
                self.label_map = {label: i for i, label in enumerate(unique_labels)}
                self.label_map_inv = {v: k for k, v in self.label_map.items()}
        self.dataset = dataset.loc[dataset["Split"] == split]
        self.dataset.reset_index()

        self.col = col
        print(self.__len__(), "samples")
        self.label = label
        assert self.__len__() != 0, self.dataset

    def __len__(self):
        return len(self.dataset)

    def load_img(self, index):
        row = self.dataset.iloc[index]
        from_im = np.load(row[self.col])
        out = {}
        for key, value in from_im.items():
            if key == "seg":
                continue
            value: np.ndarray
            out[key] = value.astype(np.float32).copy()
        return out

    def add_label(self, out, index):
        if self.label is not None:
            label = self.dataset.iloc[index][self.label]
            if self.label_map is not None:
                label = self.label_map[label]
            out["label"] = label

    def __getitem__(self, index):
        out = self.load_img(index)
        out = self.transform(out)
        self.add_label(out, index)
        return out

    def transform(self, inp: dict[str, np.ndarray]):
        out = {}
        crop = None
        for key, v in inp.items():
            # apply the pad

            img, pads = target_pad(v, self.opt.shape, mode="reflect")

            img = torch.from_numpy(img)
            img = img.unsqueeze(0).unsqueeze(1)
            assert img.max() <= 1.0, img.max()
            assert img.min() >= 0.0, img.min()
            # Random crop
            if crop is None:
                crop = torchvision.transforms.RandomCrop.get_params(img, output_size=self.opt.shape)  # type: ignore
            img = tf.crop(img, *crop)

            out[key] = img
        out = self.final_transforms(out)
        # print([(k, a.shape) for k, a in out.items()])
        return out

    def final_transforms(self, inp: dict[str, torch.Tensor]):
        # Random horizontal flipping
        if random.random() > 0.5:
            inp = {key: tf.hflip(v) for key, v in inp.items()}

        # Random vertical flipping
        if random.random() > 0.5:
            inp = {key: tf.vflip(v) for key, v in inp.items()}

        if random.random() > 0.5:
            inp = {key: v.swapaxes(-1, -2) for key, v in inp.items()}
        inp = {key: v.squeeze(0) * 2 - 1 for key, v in inp.items()}
        return inp

    def get_extended_info(self, index):
        return self.dataset.iloc[index], *self.__getitem__(index)
