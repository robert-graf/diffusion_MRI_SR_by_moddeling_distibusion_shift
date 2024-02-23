import copy
import os
import pickle
import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.utils.tensorboard
from monai.visualize import plot_2d_or_3d_image
from pytorch_lightning import loggers as pl_loggers
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from dataloader.datasets.dataset_utils import target_pad
from diffusion.ddim_sampler import get_sampler
from models import Model
from models.model_factory import get_discriminator, get_model
from models.srmodel.RRDBNet import MultiStepRestartLR
from pl_models.pl_model_prototype import LitModel_with_dataloader
from pl_models.pl_utils import loss_computation
from utils.arguments import DAE_Option
from utils.enums import TrainMode
from utils.enums_model import OptimizerType
from utils.hessian_penalty_pytorch import hessian_penalty
from utils.mri import extract_slices_from_volume

from .pl_utils.dist import get_world_size

if TYPE_CHECKING:
    from TPTBox import NII


class DAE_LitModel(LitModel_with_dataloader):
    ###### INIT PROCESS ######
    def __init__(self, conf: DAE_Option):
        super().__init__(conf)
        self.conf = conf
        self.model: Model = get_model(conf)
        self.ema_model: Model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        self.last_100_loss = deque()
        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print("Model params: %.2f M" % (model_size / 1024 / 1024))

        self.model = self.model.cuda()
        # this is shared for both model and latent
        if conf.train_mode == TrainMode.ESRGANModel:
            self.model_type_handler: loss_computation.Loss_Computation = loss_computation.ESRGANModel(conf)
        elif conf.train_mode == TrainMode.Pix2Pix:
            self.model_type_handler = loss_computation.Pix2Pix(conf)
        elif conf.train_mode == TrainMode.RCAN:
            self.model_type_handler = loss_computation.L1(conf)
        else:
            self.model_type_handler = loss_computation.Diffusion_Loss_Computation(conf)

        # initial variables for consistent sampling
        self.register_buffer("x_T", torch.randn(conf.sample_size, self.conf.in_channels, *conf.shape), persistent=False)

        if conf.pretrain is not None:
            print(f"loading pretrain ... {conf.pretrain.name}")
            state = torch.load(conf.pretrain.path, map_location="cpu")
            print("step:", state["global_step"])
            self.load_state_dict(state["state_dict"], strict=False)
        if conf.train_mode.has_discriminator():
            self.automatic_optimization = False
            self.discriminator = get_discriminator(conf)

        # Pytorch Lightning calls the following things.
        # self.prepare_data()
        # self.setup(stage)
        # self.train_dataloader()
        # self.val_dataloader()
        # self.test_dataloader()
        # self.predict_dataloader()

    ###### Training #####
    def forward(self, noise=None, x_start=None, ema_model: bool = False, **qargs):
        with autocast(False):
            palette_condition = None
            if palette_condition is None and len(self.conf.palette_condition) != 0:
                palette_condition = [qargs[a] for a in self.conf.palette_condition]

            self.model_type_handler.forward(
                self, noise=noise, x_start=x_start, ema_model=ema_model, palette_condition=palette_condition, **qargs
            )

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        self.last_100_loss.append(loss["loss"].detach().cpu().numpy())
        if len(self.last_100_loss) == 101:
            self.last_100_loss.popleft()
        self.log("train/avg_loss", value=np.mean(np.array(self.last_100_loss)).item(), prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def upscale_nii(self, nii_org: "NII", rescale=True):
        nii = nii_org.reorient_()
        if rescale:
            nii = nii.rescale((0.8571, 0.8571, 0.8571))
        arr_out = nii.get_array().astype(float)
        size = (nii.shape[0] + 8 - nii.shape[0] % 8, nii.shape[1] + 8 - nii.shape[1] % 8)
        rand = torch.randn((size[1], size[0]))
        batch_size_ = self.opt.batch_size
        for i in range(0, nii.shape[-1], batch_size_):
            j = i + batch_size_  # min(i + batches, arr_out.shape[-1] - 1)
            if j > arr_out.shape[-1]:
                j = None
                batch_size_ = arr_out.shape[-1] - i
            arr_new = arr_out[:, :, i:j].copy().astype(float)
            arr, pads = target_pad(arr_new, [size[0], size[1], batch_size_])
            reversed_pad = tuple(slice(b, -a if a != 0 else None) for a, b in pads)
            with torch.no_grad():
                img_lr = torch.from_numpy(arr).permute((2, 1, 0)).unsqueeze_(1).to(self.device, torch.float32)
                img_lr = img_lr * 2 - 1
                cond = self.encode(img_lr)
                noise = torch.stack([rand for _ in range(img_lr.shape[0])], 0).unsqueeze_(1).to(img_lr.device)
                if len(self.conf.palette_condition) != 0:
                    pred2: torch.Tensor = self.render(noise, cond, T=20, palette_condition=[img_lr])
                else:
                    pred2: torch.Tensor = self.render(noise, img_lr, T=20)
                pred2 = pred2.squeeze_(1).permute((2, 1, 0)).cpu().numpy()[reversed_pad]
                arr_out[:, :, i:j] = pred2
        nii_out = nii.set_array(arr_out + 1)
        return nii_out

    def on_validation_end(self):
        opt = self.conf
        if self.current_epoch % opt.val_niis_every_epoch != 0 or self.current_epoch < opt.val_niis_every_epoch_start:
            return

        if self.opt.val_niis is not None:
            out_ssim_pk = Path(self.logger.log_dir, "ssim_ax.pkl")  # type: ignore
            out_psnr_pk = Path(self.logger.log_dir, "psnr_ax.pkl")  # type: ignore
            if out_psnr_pk.exists():
                with open(out_ssim_pk, "rb") as handle:
                    ssim_dict = pickle.load(handle)
                with open(out_psnr_pk, "rb") as handle:
                    psnr_dict = pickle.load(handle)
            else:
                ssim_dict: dict[str | int, list[str | float]] = {"files": [Path(a).name for a in self.opt.val_niis]}
                psnr_dict: dict[str | int, list[str | float]] = {"files": [Path(a).name for a in self.opt.val_niis]}
            if self.current_epoch in ssim_dict:
                return
            print("\nstart evaluation")
            ssim_dict[self.current_epoch] = []
            psnr_dict[self.current_epoch] = []
            for nii_lr in self.opt.val_niis:  # type: ignore
                from BIDS import NII

                nii_lr = NII.load(nii_lr, False)
                nii_lr /= nii_lr.max()
                nii_out = self.upscale_nii(nii_lr, rescale=True).resample_from_to(nii_lr, verbose=False)
                ssim = nii_lr.ssim(nii_out, min_v=0)
                ssim_dict[self.current_epoch].append(ssim)
                psnr = nii_lr.psnr(nii_out, min_v=0)
                psnr_dict[self.current_epoch].append(psnr)
                print(round(ssim, ndigits=4), round(psnr, ndigits=4))

            with open(out_ssim_pk, "wb") as handle:
                pickle.dump(ssim_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(out_psnr_pk, "wb") as handle:
                pickle.dump(psnr_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            df = pd.DataFrame.from_dict(ssim_dict)
            df.to_excel(str(out_ssim_pk).rsplit(".", maxsplit=0)[0] + ".xlsx")
            df = pd.DataFrame.from_dict(psnr_dict)
            df.to_excel(str(out_psnr_pk).rsplit(".", maxsplit=0)[0] + ".xlsx")
            # TODO compare nii_out
            print(
                "SSIM",
                sum(ssim_dict[self.current_epoch]) / len(ssim_dict[self.current_epoch]),
                "PSNR",
                sum(psnr_dict[self.current_epoch]) / len(psnr_dict[self.current_epoch]),
            )  # type: ignore
        if self.opt.val_niis_pairs is not None:
            out_ssim_pk = Path(self.logger.log_dir, "ssim_paired.pkl")  # type: ignore
            out_psnr_pk = Path(self.logger.log_dir, "psnr_paired.pkl")  # type: ignore
            if out_psnr_pk.exists():
                with open(out_ssim_pk, "rb") as handle:
                    ssim_dict2 = pickle.load(handle)
                with open(out_psnr_pk, "rb") as handle:
                    psnr_dict2 = pickle.load(handle)
            else:
                ssim_dict2: dict[str | int, list[str | float]] = {"files": [Path(a[0]).name for a in self.opt.val_niis_pairs]}
                psnr_dict2: dict[str | int, list[str | float]] = {"files": [Path(a[0]).name for a in self.opt.val_niis_pairs]}
            if self.current_epoch in ssim_dict2:
                return
            print("\nstart evaluation pairs")
            ssim_dict2[self.current_epoch] = []
            psnr_dict2[self.current_epoch] = []

            for nii_lr_, nii_iso_ in self.opt.val_niis_pairs:  # type: ignore
                from BIDS import NII

                nii_lr = NII.load(nii_lr_, False).clamp(0, 1)
                nii_iso = NII.load(nii_iso_, False).clamp(0, 1)
                nii_out = self.upscale_nii(nii_lr, rescale=False)
                ssim = nii_iso.ssim(nii_out, min_v=0)
                ssim_dict2[self.current_epoch].append(ssim)
                psnr = nii_iso.psnr(nii_out, min_v=0)
                psnr_dict2[self.current_epoch].append(psnr)
                print(round(ssim, ndigits=4), round(psnr, ndigits=4))

            with open(out_ssim_pk, "wb") as handle:
                pickle.dump(ssim_dict2, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(out_psnr_pk, "wb") as handle:
                pickle.dump(psnr_dict2, handle, protocol=pickle.HIGHEST_PROTOCOL)

            df = pd.DataFrame.from_dict(ssim_dict2)
            df.to_excel(str(out_ssim_pk).rsplit(".", maxsplit=0)[0] + ".xlsx")
            df = pd.DataFrame.from_dict(psnr_dict2)
            df.to_excel(str(out_psnr_pk).rsplit(".", maxsplit=0)[0] + ".xlsx")

            # TODO compare nii_out
            print(
                "SSIM",
                sum(ssim_dict2[self.current_epoch]) / max(1, len(ssim_dict2[self.current_epoch])),
                "PSNR",
                sum(psnr_dict2[self.current_epoch]) / max(1, len(psnr_dict2[self.current_epoch])),
            )  # type: ignore

    def load_images(self, batch: dict):
        mask = batch.get("mask", 1)
        x_start = batch[self.conf.x_start]

        ### palette ###
        palette_condition = None
        if len(self.conf.palette_condition) != 0:
            palette_condition = [batch[a] * mask for a in self.conf.palette_condition]
        model_kwargs = {"palette_condition": palette_condition}

        ### palette ###
        if "img_aug" in batch:
            model_kwargs = {"x_start_aug": batch["img_aug"], "palette_condition": palette_condition}
        model_kwargs["x_start_aug"] = batch[self.conf.image_name] * mask
        return x_start, model_kwargs, mask, model_kwargs["x_start_aug"]

    def _shared_step(self, batch, batch_idx, step_mode: str):
        """
        given an input, calculate the loss function
        no optimization at this stage.
        """
        with autocast(True):
            losses = {}
            # if self.conf.train_mode == TrainMode.diffusion:
            """
            main training mode!!!
            """
            x_start, model_kwargs, *_ = self.load_images(batch)
            losses = self.model_type_handler.loss(self, x_start, model_kwargs=model_kwargs, train=step_mode == "train")

            # hessian_penalty
            if self.conf.hessian_penalty != 0:
                # https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510579.pdf
                assert "cond_emb" in losses
                hl = hessian_penalty(self.model.encoder, x_start, G_z=losses["cond_emb"])
                losses["hessian_penalty"] = hl.detach()
                losses["loss"] = losses["loss"] + hl
            # else:
            #    raise NotImplementedError()
            losses = {k: v.detach() if k != "loss" else v for k, v in losses.items() if not isinstance(v, list)}
            # divide by accum batches to make the accumulated gradient exact!
            for key in losses:
                losses[key] = self.all_gather(losses[key]).mean()  # type: ignore
                self.log(f"loss/{step_mode}_{key}", losses[key].item(), rank_zero_only=True)
        return losses

    def on_train_batch_end(self, outputs, batch: dict, batch_idx: int) -> None:
        """
        after each training step ...
        """
        if self.is_last_accum(batch_idx):
            ema(self.model, self.ema_model, self.conf.ema_decay)
            x_start, model_kwargs, mask, imgs_lr = self.load_images(batch)
            # logging
            palette_condition = None
            if len(self.conf.palette_condition) != 0:
                palette_condition = [batch[a] for a in self.conf.palette_condition]

            if not self.trainer.fast_dev_run:  # and self.conf.train_mode.is_diffusion():  # type: ignore
                self.log_sample(x_start=x_start, x_start_lr=imgs_lr, palette_condition=palette_condition, mask=mask)
                # self.evaluate_scores()

    #### Optimizer ####
    def configure_optimizers(self):
        if self.conf.optimizer == OptimizerType.adam:
            optim = torch.optim.Adam(
                self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay, betas=(self.conf.beta1, self.conf.beta2)
            )
        elif self.conf.optimizer == OptimizerType.adamW:
            optim = torch.optim.AdamW(
                self.model.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay, betas=(self.conf.beta1, self.conf.beta2)
            )
        else:
            raise NotImplementedError()
        optims = [optim]
        optim_d = None
        if self.conf.train_mode.has_discriminator():
            if self.conf.optimizer == OptimizerType.adam:
                optim_d = torch.optim.Adam(
                    self.discriminator.parameters(),
                    lr=self.conf.lr,
                    weight_decay=self.conf.weight_decay,
                    betas=(self.conf.beta1, self.conf.beta2),
                )
            elif self.conf.optimizer == OptimizerType.adamW:
                optim_d = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=self.conf.lr,
                    weight_decay=self.conf.weight_decay,
                    betas=(self.conf.beta1, self.conf.beta2),
                )
            else:
                raise NotImplementedError()
            optims.append(optim_d)
        self.has_schedular = True
        if self.conf.schedular == "":
            self.has_schedular = False
            return optims
        elif self.conf.schedular == "MultiStepRestartLR":
            schedular = [MultiStepRestartLR(optim, milestones=[400000], gamma=0.5)]
            if self.conf.train_mode.has_discriminator():
                schedular.append(MultiStepRestartLR(optim_d, milestones=[400000], gamma=0.5))
            return optims, schedular
        elif self.conf.schedular == "MultiStepLR":
            from torch.optim.lr_scheduler import MultiStepLR

            schedular = [MultiStepLR(optim, milestones=[50000], gamma=0.5)]
            if self.conf.train_mode.has_discriminator():
                schedular.append(MultiStepLR(optim, milestones=[50000], gamma=0.5))
            return optims, schedular
        else:
            raise NotImplementedError(self.conf.schedular)

    def render(
        self,
        noise: torch.Tensor | None,
        cond: torch.Tensor | None,
        T: int | None = None,
        x_start: torch.Tensor | None = None,
        palette_condition: list | None = None,
        ema_model=True,
    ):
        return self.model_type_handler.render(
            self, noise=noise, cond=cond, T=T, x_start=x_start, palette_condition=palette_condition, ema_model=ema_model
        )

    def encode(self, x) -> torch.Tensor | None:
        if not self.conf.model_type.has_encoder(self.model):
            return None
        cond = self.model.encode(x)  # type: ignore
        return cond

    def encode_stochastic(self, x, cond, T=None, palette_condition: list | None = None):
        sampler = self.model_type_handler.eval_sampler if T is None else get_sampler(self.conf, eval=False, T=T)  # type: ignore

        out = sampler.ddim_reverse_sample_loop(
            self.ema_model,
            x,
            model_kwargs={"cond": cond, "palette_condition": palette_condition},
        )
        return out["sample"]

    def log_sample(
        self,
        x_start: torch.Tensor,
        x_start_lr: torch.Tensor,
        palette_condition: list | None,
        mask: torch.Tensor | Literal[1] | None,
    ):
        """
        put images to the tensorboard
        """
        if mask == 1:
            mask = None
        if self.conf.val_check_interval > 0 and is_time(
            self.num_samples,
            self.conf.val_check_interval,
            self.conf.batch_size_effective,
        ):
            inp = {"self": self, "x_start": x_start, "x_start_lr": x_start_lr, "palette_condition": palette_condition, "mask": mask}
            if self.conf.model_type.has_autoenc() and self.conf.model_type.can_sample():
                _log_sample(**inp, postfix="", use_xstart=False)
                # autoencoding mode
                _log_sample(**inp, postfix="_enc", use_xstart=True, save_real=True)
            else:
                _log_sample(**inp, postfix="", use_xstart=True, save_real=True)

        self.model.train()
        self.ema_model.train()

    def split_tensor(self, x):
        """
        extract the tensor for a corresponding 'worker' in the batch dimension
        Args:
            x: (n, c)
        Returns: x: (n_local, c)
        """
        n = len(x)
        rank = self.global_rank
        world_size = get_world_size()
        # print(f'rank: {rank}/{world_size}')
        per_rank = n // world_size
        return x[rank * per_rank : (rank + 1) * per_rank]


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay),
            non_blocking=True,
        )


class WarmupLR:
    def __init__(self, warmup) -> None:
        self.warmup = warmup

    def __call__(self, step):
        return min(step, self.warmup) / self.warmup


def is_time(num_samples, every, step_size):
    closest = (num_samples // every) * every
    return num_samples - closest < step_size


@torch.no_grad()
def _log_sample(
    self: DAE_LitModel,
    x_start: torch.Tensor,  # type: ignore
    x_start_lr: torch.Tensor,  # type: ignore
    palette_condition: list | None,
    mask: torch.Tensor | None = None,
    postfix="",
    use_xstart: bool = True,
    save_real=False,
    # interpolate=False,
):
    global buffer_image_names
    all_x_T = self.split_tensor(self.x_T)
    batch_size = min(len(all_x_T), self.conf.batch_size_eval)
    ## allow for super large models
    loader = DataLoader(all_x_T, batch_size=batch_size)  # type: ignore
    Gen = []
    for x_T in loader:  # tqdm(loader, desc="img", total=len(loader)):
        if palette_condition is not None:
            if mask is not None:
                _palette_condition = [i[: len(x_T)] * mask[: len(x_T)] for i in palette_condition]
            else:
                _palette_condition = [i[: len(x_T)] for i in palette_condition]

        else:
            _palette_condition = None

        if use_xstart:
            if x_start_lr is None:
                _xstart = x_start[: len(x_T)]

            else:
                _xstart = x_start_lr[: len(x_T)]
                _xstart_hr = x_start[: len(x_T)]
            if mask is not None:
                _xstart *= mask[: len(x_T)]
        else:
            _xstart = None
        cond = None
        gen = self.render(noise=x_T, cond=cond, x_start=_xstart, palette_condition=_palette_condition, ema_model=False)
        break
        # Gen.append(gen)
    # gen: torch.Tensor = torch.cat(Gen)
    gen = self.all_gather(gen)  # type: ignore
    if (gen.dim() - self.conf.dims) == 3:
        # collect tensors from different workers
        # (n, c, h, w)
        gen = gen.flatten(0, 1)
    if self.conf.dims == 3:
        if self.global_rank == 0:
            if isinstance(self.logger, pl_loggers.TensorBoardLogger):
                plot_2d_or_3d_image(
                    torch.cat((torch.clamp(gen * 2, -1, 1), x_start), -3),
                    self.global_step,
                    self.logger.experiment,
                    tag=f"sample{postfix}/fake",
                    frame_dim=-1,
                )
            elif isinstance(self.logger, pl_loggers.WandbLogger):
                raise NotImplementedError()
        return
        # gen = extract_slices_from_volume(gen[..., : min(self.opt.shape), : min(self.opt.shape), : min(self.opt.shape)])
        gen = None
    sample_dir = os.path.join(self.conf.log_dir, self.conf.experiment_name, f"sample{postfix}")
    if self.global_rank == 0 and not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    def grid_params(t, f=1):
        return {
            "tensor": t,
            "nrow": 3 if self.conf.dims == 3 else int(np.sqrt(t.size(0) / f)),
            "normalize": True,
            "padding": 0,
            "value_range": (-1, 1),
        }

    log_img = []
    if save_real and use_xstart and x_start_lr is not None:
        a = _save_image(self, _xstart_hr, grid_params, postfix, sample_dir)
        log_img.append(_xstart_hr)
    if save_real and use_xstart:
        a = _save_image(self, _xstart, grid_params, postfix, sample_dir)
        log_img.append(_xstart)
    if self.global_rank == 0 and gen is not None:
        # save samples to the tensorboard
        gen_grid = make_grid(**grid_params(torch.concat([*log_img, gen], dim=-1), f=3))
        path = Path(sample_dir, f"{self.global_step}.png")
        remove_old_jpgs(path)
        save_image(gen_grid, path)
        self.log_image(f"sample{postfix}/fake", gen_grid, self.global_step)
    # x_start_lr


def _save_image(self: DAE_LitModel, _xstart, grid_params, postfix, sample_dir):
    # save the original images to the tensorboard
    real: torch.Tensor = self.all_gather(_xstart)  # type: ignore
    if (real.dim() - self.conf.dims) == 3:
        real = real.flatten(0, 1)
    if self.conf.dims == 3:
        # visualize volume using MONAI
        if self.global_rank == 0:
            if isinstance(self.logger, pl_loggers.TensorBoardLogger):
                plot_2d_or_3d_image(
                    real,
                    self.global_step,
                    self.logger.experiment,
                    tag=f"sample{postfix}/real",
                    frame_dim=-1,
                )
            elif isinstance(self.logger, pl_loggers.WandbLogger):
                raise NotImplementedError()
                # log as 3d object
                # TODO: add rendering as mesh
        # extract 2d slice from different sequences
        return None
        real = extract_slices_from_volume(real[..., : min(self.opt.shape), : min(self.opt.shape), : min(self.opt.shape)])
    if self.global_rank == 0:
        real_grid = make_grid(**grid_params(real))
        # self.log_image(f"sample{postfix}/real", real_grid, self.global_step)
        path = Path(sample_dir, "real.png")
        remove_old_jpgs(path)
        save_image(real_grid, path)
        return real_grid
    return None


buffer_image_names = deque()
images_keept_anyway = 0
images_removed = -100


def remove_old_jpgs(path):
    global images_removed
    global images_keept_anyway
    buffer_image_names.append(path)
    if len(buffer_image_names) >= 100:
        old_path = buffer_image_names.popleft()
        p = random.random() + images_removed / (images_keept_anyway * images_keept_anyway * 100 + 1)
        if p < 1:
            Path(old_path).unlink()
            images_removed += 1
        else:
            images_keept_anyway += 1
