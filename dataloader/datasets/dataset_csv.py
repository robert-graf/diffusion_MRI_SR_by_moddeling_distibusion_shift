from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Dataset_CSV(Dataset):
    def __init__(self, path, transform, split: None | Literal["train", "val", "test"] = None, col="file_path", label=None):
        print(path)
        puffer = ""
        self.label_map = None
        if Path(path).is_dir():
            path = Path(path)
            puffer = path.parent / f"_{path.name}_{split}.xlsx"
            if puffer.exists():
                path = puffer
        if Path(path).is_dir():
            unique_labels = list(map(str, list(Path(path).glob("*.png")) + list(Path(path).glob("*/*.png"))))
            l2 = [split for _ in range(len(unique_labels))]
            self.dataset = pd.DataFrame({col: unique_labels, "Split": l2})
            self.dataset.copy().to_excel(puffer)
        else:
            dataset = pd.read_excel(path) if not str(path).endswith(".csv") else pd.read_csv(path)
            assert col in dataset, dataset
            assert not isinstance(transform, tuple)
            if label is not None:
                unique_labels = sorted(set(dataset[label].to_list()))

                if any(isinstance(ul, str) for ul in unique_labels):
                    self.label_map = {label: i for i, label in enumerate(unique_labels)}
                    self.label_map_inv = {v: k for k, v in self.label_map.items()}
            self.dataset = dataset.loc[dataset["Split"] == split]
            self.dataset.reset_index()

        self.col = col
        self.transform = transform
        print(self.__len__(), "samples")
        self.label = label
        assert self.__len__() != 0, self.dataset

    def __len__(self):
        return len(self.dataset)

    def load_img(self, index):
        row = self.dataset.iloc[index]
        from_im = Image.open(row[self.col])
        from_im = from_im.convert("L")
        return from_im

    def add_label(self, out, index):
        if self.label is not None:
            label = self.dataset.iloc[index][self.label]
            if self.label_map is not None:
                label = self.label_map[label]
            out["label"] = label

    def __getitem__(self, index):
        from_im = self.load_img(index)
        from_im = self.transform(from_im)

        out = {"img": from_im}  # , "index": target, "cls_labels": target}
        self.add_label(out, index)
        return out

    def get_extended_info(self, index):
        return self.dataset.iloc[index], *self.__getitem__(index)
