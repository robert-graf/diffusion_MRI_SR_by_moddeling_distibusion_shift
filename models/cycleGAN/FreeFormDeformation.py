import deepali.core.functional
import torch
from deepali.spatial import Grid, ImageTransformer, StationaryVelocityFreeFormDeformation
from torch import nn

from utils.perlin import rand_perlin_2d


def next8(number: int):
    if number % 8 == 0:
        return number
    return number + 8 - number % 8


class DeformationLayer(nn.Module):
    def __init__(self, shape, stride=10) -> None:
        super().__init__()
        self.shape = shape
        grid = Grid(size=shape)
        self.field = StationaryVelocityFreeFormDeformation(grid, stride=stride, params=self.params)  # type: ignore
        self.field.requires_grad_(False)
        self.transformer = ImageTransformer(self.field)
        self.transformer_inv = ImageTransformer(self.field.inverse(link=True))

    def params(self, *args, **kargs):
        # print(args, kargs)
        return self._parm

    def new_deformation(self, device):
        shape = self.field.data_shape
        s = (next8(shape[-2]), next8(shape[-1]))
        noise_2d = rand_perlin_2d(s, (4, 4)) * 0.05
        noise_2d += rand_perlin_2d(s, (8, 8)) * 0.03
        noise_2d += rand_perlin_2d(s, (2, 2)) * 0.2
        noise_2d = noise_2d[: shape[-2], : shape[-1]]
        self._parm = torch.stack([noise_2d for _ in range(shape[-3])], 0).unsqueeze(0).to(device)
        self.field.condition_()

    def deform(self, i: torch.Tensor):
        if len(i) == 3:
            i = i.unsqueeze(0)
        return self.transformer.forward(i)

    def back_deform(self, i: torch.Tensor):
        if len(i) == 3:
            i = i.unsqueeze(0)
        return self.transformer_inv.forward(i)

    def get_gird(self, stride=16, device=None):
        high_res_grid = self.field.grid().resize(self.shape[-2:])
        return deepali.core.functional.grid_image(high_res_grid, num=1, stride=stride, inverted=True, device=device)


if __name__ == "__main__":
    import sys

    sys.path.append("../..")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from BIDS import NII

    nii = NII.load("/DATA/NAS/ongoing_projects/robert/NAKO/rawdata/sub-101001/T2w/sub-101001_acq-sag_chunk-HWS_sequ-29_T2w.nii.gz", False)
    img = torch.Tensor(nii.get_array()[..., 7].astype(np.float32)).T
    img.shape

    def show(*img):
        img = [(i.detach().cpu() if isinstance(i, torch.Tensor) else torch.from_numpy(i)) for i in img]
        img = [i / i.max() for i in img]

        img = [i.unsqueeze(0) if len(i.shape) == 2 else i for i in img]
        img = [i if len(i.shape) == 3 else i.squeeze(0) for i in img]
        np_img = torch.cat(img, dim=-1).numpy()

        plt.figure(figsize=(20, 6))
        print(np_img.shape)
        plt.imshow(np.transpose(np_img, (1, 2, 0)), interpolation="nearest", cmap="gray")

    i = img.unsqueeze(0)
    shape = i.shape[-2:]

    deform_layer = DeformationLayer(shape)

    with torch.no_grad():
        deform_layer.new_deformation()
        out = deform_layer.deform(i)
        out2 = deform_layer.back_deform(out)
        show(img.squeeze(), out.squeeze(), out2.squeeze(), deform_layer.deform(deform_layer.get_gird()))
