import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pytorch_lightning.utilities.types import _METRIC

sys.path.append("..")
from collections import deque

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import torch.utils.tensorboard
import torchvision.transforms.functional as F
from monai.utils.misc import set_determinism
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from dataloader.dataset_factory import get_data_loader, get_dataset
from utils.arguments import DataSet_Option

from .pl_utils.dist import get_world_size


class LitModel_with_dataloader(pl.LightningModule):
    ###### INIT PROCESS ######
    def __init__(self, conf: DataSet_Option):
        super().__init__()
        if conf.seed is not None:
            pl.seed_everything(conf.seed)
            set_determinism(seed=conf.seed)

        self.save_hyperparameters()
        self.opt = conf
        self.dataset_is_loaded = False  # type: ignore
        self._counter: int = 0

    def prepare_data(self):
        if not self.dataset_is_loaded:
            print("load training dataset")
            self.train_data = get_dataset(self.opt, split="train")
            print("load val dataset")
            self.val_data = get_dataset(self.opt, split="val")
            self.dataset_is_loaded = True

            print("train data:", len(self.train_data))
            print("val data:", len(self.val_data))
        return self.train_data

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.opt.seed is not None:
            seed = self.opt.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print("local seed:", seed)
        ##############################################

    def log(
        self,
        name: str,
        value: _METRIC | deque,
        prog_bar: bool = False,
        logger: bool | None = None,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        reduce_fx: str | Callable[..., Any] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Any | None = None,
        add_dataloader_idx: bool = True,
        batch_size: int | None = None,
        metric_attribute: str | None = None,
        rank_zero_only: bool = False,
    ) -> None:
        if isinstance(value, deque):
            value = np.mean(np.array(value)).item()
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().item()

        return super().log(
            name,
            value,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            metric_attribute,
            rank_zero_only,
        )

    ####### DATA LOADER #######
    def train_dataloader(self):
        return self._shared_loader("train")

    def val_dataloader(self):
        return self._shared_loader("val")

    def _shared_loader(self, mode: Literal["train", "val"], super_res=False):
        opt = self.opt
        print(f"on {mode} dataloader start ...")
        # if self.opt.train_mode.require_dataset_infer():
        #    raise NotImplementedError("latent_infer_path")
        #    # return the dataset with pre-calculated conds
        #    loader_kwargs = dict(dataset=TensorDataset(self.conds), shuffle=True)
        #    return get_data_loader()  # TODO
        # else:
        train = mode == "train"
        return get_data_loader(opt, self.train_data if train else self.val_data, shuffle=train, drop_last=not train)

    def on_train_start(self):
        super().on_train_start()
        try:
            early_stopping = next(c for c in self.trainer.callbacks if isinstance(c, EarlyStopping))  # type: ignore
            early_stopping.patience = self.opt.early_stopping_patience
        except StopIteration:
            pass

    def is_last_accum(self, batch_idx):
        """
        is it the last gradient accumulation loop?
        used with gradient_accum > 1 and to see if the optimizer will perform 'step' in this iteration or not
        """
        return (batch_idx + 1) % self.opt.accum_batches == 0

    #### Optimizer ####
    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # fix the fp16 + clip grad norm problem with pytorch lightinng
        # this is the currently correct way to do it
        grad_clip = self.opt.grad_clip
        if grad_clip > 0:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            clip_grad_norm_(params, max_norm=grad_clip)

    @property
    def num_samples(self):
        """
        (global) batch size * iterations
        """
        # batch size here is global!
        # global_step already takes into account the accum batches
        return self.global_step * self.opt.batch_size_effective

    def log_image(self, tag: str, image: torch.Tensor | np.ndarray | Image.Image, step: int) -> None:
        if self.global_rank == 0:
            experiment: SummaryWriter = self.logger.experiment  # type: ignore
            if isinstance(experiment, SummaryWriter):
                print(image.shape)
                experiment.add_image(tag, image, step)
            # elif isinstance(experiment, wandb.sdk.wandb_run.Run):
            #    experiment.log(
            #        {tag: [wandb.Image(image.cpu())]},
            #        # step=step,
            #    )
            else:
                raise NotImplementedError()

    def log_image_2D(self, image_dict: dict[str, torch.Tensor | np.ndarray] | dict[str, torch.Tensor] | dict[str, np.ndarray], postfix=""):
        sample_dir = self.opt.get_sample_dir(f"sample{postfix}")
        sample_dir.mkdir(exist_ok=True)
        path = Path(sample_dir, f"{self._counter}.png")
        remove_old_jpgs(path)
        img = to_2D_image(image_dict)
        img.save(path)
        self.log_image("conditional image", np.array(img).transpose(2, 0, 1), self._counter)
        self._counter += 1


buffer_image_names = deque()
images_keept_anyway = 0
images_removed = -100


def remove_old_jpgs(path):
    global images_removed
    global images_keept_anyway
    if len(buffer_image_names) >= 100:
        old_path = buffer_image_names.popleft()
        p = random.random() + images_removed / (images_keept_anyway * images_keept_anyway * 100 + 1)
        if p < 1:
            Path(old_path).unlink()
            images_removed += 1
        else:
            images_keept_anyway += 1
    buffer_image_names.append(path)


def to_2D_image(image_dict: dict[str, torch.Tensor | np.ndarray]):
    # Assume you have a dictionary of tensors
    tensor_dict = {}
    for k, v in image_dict.copy().items():
        v = v.detach().cpu() if isinstance(v, torch.Tensor) else torch.from_numpy(v)
        v -= v.min()
        tensor_dict[k] = v / v.max()
    # Concatenate the images along the batch axis
    concatenated_images = torch.cat(list(tensor_dict.values()), dim=0)
    border = 2
    # Get the batch size and spatial dimensions from the concatenated tensor
    try:
        batch_size, channels, height, width = concatenated_images.shape
    except ValueError:
        print(concatenated_images.shape)
        raise
    width += border
    height += border

    # Calculate the total width and height of the resulting image
    total_width = len(tensor_dict) * width
    head_space = int(height / 6)
    total_height = batch_size * height / len(tensor_dict) + head_space
    # Create a blank PIL image with the appropriate size and 3 channels for RGB
    image = Image.new("RGB", (total_width, int(total_height)), color=(255, 255, 255))

    # Use PIL to draw the tensors on the image with red text
    draw = ImageDraw.Draw(image)
    y_offset = head_space
    # Set the font size
    font_size = int(head_space * 0.8)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf", font_size)

    for idx, (key, tensor) in enumerate(tensor_dict.items()):
        # Iterate over the batch dimension
        for i in range(tensor.shape[0]):
            pil_image = F.to_pil_image(tensor[i].repeat(3, 1, 1)) if channels == 1 else F.to_pil_image(tensor[i])
            image.paste(pil_image, (width * list(tensor_dict.keys()).index(key) + 1, y_offset))
            if i == 0:
                draw.text((width * idx + 1, 1), key, fill=(0, 0, 0), font=font)  # Red text
            y_offset += height  # Assuming the height of the tensor is the same for all
        y_offset = head_space
    return image
