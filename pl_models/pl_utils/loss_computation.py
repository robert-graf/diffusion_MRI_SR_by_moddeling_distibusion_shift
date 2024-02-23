from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import Tensor, nn
from torchmetrics.functional import structural_similarity_index_measure

from diffusion.beta_schedule import ScheduleSampler
from diffusion.ddim_sampler import get_sampler
from diffusion.renderer import render_condition, render_uncondition
from utils.arguments import DAE_Option

if TYPE_CHECKING:
    from pl_models.DEA import DAE_LitModel


class Optimizer:
    def __init__(self, pl_module: pl.LightningModule, opt: torch.optim.Adam | LightningOptimizer, train=True):
        self.pl_module = pl_module
        self.opt = opt
        self.train = train

    def __enter__(self):
        if self.train:
            self.pl_module.toggle_optimizer(self.opt)

    def __exit__(self, *args):
        if self.train:
            self.opt.step()
            self.opt.zero_grad()  # type: ignore
            self.pl_module.untoggle_optimizer(self.opt)


class Loss_Computation(ABC):
    loss_dict = {}  # noqa: RUF012

    def log(self, name, loss1):
        self.loss_dict[name] = loss1.detach().cpu()

    def forward(
        self, pl_model: "DAE_LitModel", cond=None, x_start=None, noise=None, ema_model: bool = False, palette_condition=None, **qargs
    ):
        model = pl_model.ema_model if ema_model else pl_model.model
        if cond is None:
            if x_start is None and palette_condition is not None:
                cond = palette_condition[0] if len(palette_condition) == 1 else torch.cat(palette_condition, 1)
            else:
                cond = x_start

        # {'noise': <class 'torch.Tensor'>, 'cond': <class 'NoneType'>, 'T': <class 'int'>, 'x_start': <class 'NoneType'>, 'palette_condition': <class 'list'>, 'ema_model': <class 'bool'>}
        return model.forward(cond)

    @abstractmethod
    def loss(self, pl_model: "DAE_LitModel", x_start: torch.Tensor, model_kwargs, train=True) -> dict[str, torch.Tensor]:
        ...

    def render(self, pl_model: "DAE_LitModel", **qargs) -> torch.Tensor:
        return self.forward(pl_model, **qargs)  # type: ignore


class Diffusion_Loss_Computation(Loss_Computation):
    def __init__(self, conf: DAE_Option) -> None:
        self.sampler = get_sampler(conf, eval=False)
        self.eval_sampler = get_sampler(conf, eval=True)
        self.T_sampler = ScheduleSampler(conf.num_timesteps)

    def forward(self, self_: "DAE_LitModel", cond=None, x_start=None, noise=None, ema_model: bool = False, palette_condition=None, **qargs):
        model = self_.ema_model if ema_model else self_.model
        return self.eval_sampler.sample(model=model, noise=noise, x_start=x_start, palette_condition=palette_condition, cond=cond)

    def loss(self, pl_model: "DAE_LitModel", x_start: torch.Tensor, model_kwargs, **qargs):  # noqa: ARG002
        t = self.T_sampler.sample(len(x_start), x_start.device)
        losses = self.sampler.training_losses(model=pl_model.model, x_start=x_start, t=t, model_kwargs=model_kwargs)
        return losses

    def render(
        self,
        self_: "DAE_LitModel",
        noise=None,
        cond=None,
        T=None,
        x_start=None,
        palette_condition: list | None = None,
        ema_model=True,
        **qargs,
    ) -> torch.Tensor:
        model = self_.ema_model if ema_model else self_.model
        model.eval()
        sampler = self.eval_sampler if T is None else get_sampler(self_.conf, eval=False, T=T)

        if cond is not None or (palette_condition is not None and len(palette_condition) != 0):
            pred_img = render_condition(
                self_.conf,
                model,
                noise,
                sampler=sampler,
                cond=cond,
                x_start=x_start,
                palette_condition=palette_condition,
            )
        else:
            raise NotImplementedError()
            pred_img = render_uncondition(self_.conf, model, noise, sampler=sampler, latent_sampler=None)
        return pred_img


class ESRGANModel(Loss_Computation):
    def __init__(self, opt: "DAE_Option") -> None:
        super().__init__()
        # define losses
        self.cri_pix = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()
        self.gan_type = "vanilla"
        if self.gan_type == "vanilla":
            self.cri_gan = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.cri_gan = nn.MSELoss()
        elif self.gan_type == "hinge":
            self.cri_gan = nn.ReLU()
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} is not implemented.")

    # https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/models/esrgan_model.py
    def loss(self, pl_model: "DAE_LitModel", x_start: torch.Tensor, model_kwargs: dict, train=True):
        palette_condition = model_kwargs.get("palette_condition")
        lq = model_kwargs.get("x_start_aug").detach()

        if palette_condition is None:
            lq: torch.Tensor | None = lq
        else:
            lq = torch.cat([lq, *palette_condition], 1)  # type: ignore
        gt = x_start
        optimizer_g, optimizer_d = pl_model.optimizers()  # type: ignore
        # sch1, sch2 = pl_model.lr_schedulers()  # type: ignore

        # optimize net_g
        with Optimizer(pl_model, optimizer_g, train=train):
            output = pl_model.model(lq)
            l_g_total = 0
            loss_dict = OrderedDict()
            # pixel loss
            # if self.cri_pix:
            l_g_pix = self.cri_pix(output, gt)
            l_g_total = l_g_total + l_g_pix
            loss_dict["loss_g_pix"] = l_g_pix
            # perceptual loss
            # if self.cri_perceptual:
            #    l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
            #    if l_g_percep is not None:
            #        l_g_total += l_g_percep
            #        loss_dict['l_g_percep'] = l_g_percep
            #    if l_g_style is not None:
            #        l_g_total += l_g_style
            #        loss_dict['l_g_style'] = l_g_style
            # gan loss (relativistic gan)
            real_d_pred = pl_model.discriminator(gt).detach()
            fake_g_pred = pl_model.discriminator(output)
            l_g_real = self.cri_gan(self.sigmoid(real_d_pred - torch.mean(fake_g_pred)), torch.zeros_like(real_d_pred))
            l_g_fake = self.cri_gan(self.sigmoid(fake_g_pred - torch.mean(real_d_pred)), torch.ones_like(real_d_pred))
            l_g_gan = (l_g_real + l_g_fake) / 2

            l_g_total = l_g_total + l_g_gan * 0.1
            loss_dict["loss_g_gan"] = l_g_gan
            loss_dict["loss"] = l_g_total
            pl_model.manual_backward(l_g_total) if train else None

        with Optimizer(pl_model, optimizer_d, train=train):
            # gan loss (relativistic gan)
            # real

            fake_d_pred = pl_model.discriminator(output).detach()
            real_d_pred = pl_model.discriminator(gt)
            l_d_real = self.cri_gan(self.sigmoid(real_d_pred - torch.mean(fake_d_pred)), torch.ones_like(real_d_pred)) * 0.5
            pl_model.manual_backward(l_d_real) if train else None
            # fake
            fake_d_pred = pl_model.discriminator(output.detach())
            l_d_fake = self.cri_gan(self.sigmoid(fake_d_pred - torch.mean(real_d_pred.detach())), torch.zeros_like(fake_d_pred)) * 0.5
            pl_model.manual_backward(l_d_fake) if train else None

        loss_dict["loss_d_real"] = l_d_real
        loss_dict["loss_d_fake"] = l_d_fake
        # sch1.step() if train else None  # type: ignore
        # sch2.step() if train else None  # type: ignore
        return loss_dict


class Pix2Pix(Loss_Computation):
    def __init__(self, opt: "DAE_Option") -> None:
        super().__init__()
        self.opt = opt
        # define losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_paired = torch.nn.L1Loss()
        self.first = True
        self.use_paired = True
        self.lambda_paired = 10
        self.lambda_ssim = 1
        self.lambda_GAN = 1

    # @torch.no_grad()
    # def render(self, self_: "DAE_LitModel", noise=None, cond=None, T=20, palette_condition: list[Tensor] | None = None):
    #    if palette_condition is None:
    #        palette_condition = [cond]  # type: ignore
    #    return self.forward(self_, palette_condition=palette_condition)

    def loss(self, pl_model: "DAE_LitModel", x_start: torch.Tensor, model_kwargs: dict, train=True):
        palette_condition = model_kwargs.get("palette_condition")
        lq = model_kwargs.get("x_start_aug", x_start).detach()

        if palette_condition is None:
            lq: torch.Tensor | None = lq
        else:
            lq = torch.cat([lq, *palette_condition], 1)  # type: ignore
        opt_g, opt_d_B = pl_model.optimizers()  # type: ignore
        self.loss_dict = {}
        real_A: Tensor = lq  # type: ignore
        real_B = x_start
        with Optimizer(pl_model, opt_g, train=train):
            loss1, fake_B = self.training_step_G(pl_model.model, pl_model.discriminator, real_A, real_B, None, "A2B")
            pl_model.manual_backward(loss1)
            self.loss_dict["loss"] = loss1.detach().cpu()

        # Compute forward and loss. Log loss. return one loss value.
        with Optimizer(pl_model, opt_d_B, train=train):
            loss = self.training_step_D(pl_model.discriminator, real_B, real_A, fake_B.detach(), None, None, "B")
            pl_model.manual_backward(loss)

        self.first = False
        return self.loss_dict

    def training_step_G(self, gan, discriminator, real_A: Tensor, real_B_inp: Tensor, mask: Tensor | None, name):
        opt = self.opt

        gan = gan.to(real_A.device)
        t = torch.zeros((real_A.shape[0], 1), device=real_A.device)
        fake_B: Tensor = gan(real_A, t=t)
        # First, G(A) should fake the discriminator
        ZERO = torch.zeros(1, device=gan.device)
        loss_G_GAN = ZERO
        fake = torch.cat([fake_B, real_A], dim=1) if self.use_paired else fake_B
        real_B_output = real_B_inp[:, : -opt.dims] if opt.linspace else real_B_inp
        print("discriminator G", fake.shape) if self.first else None

        if self.lambda_GAN > 0.0:
            pred_fake: Tensor = discriminator(fake, seg=mask)
            real_label = torch.ones((pred_fake.shape[0], 1), device=gan.device)
            loss_G_GAN = self.criterion_GAN(pred_fake, real_label) * self.lambda_GAN
        loss_paired = 0
        loss_ssim = 0
        #########################################
        if self.use_paired:
            print("use_paired", real_B_output.shape, fake_B.shape) if self.first else None
            loss_paired = self.criterion_paired(real_B_output, fake_B)
            self.log(f"train/loss_paired_{name}", loss_paired.detach())
            if self.lambda_ssim > 0.0:
                loss_ssim = self.lambda_ssim * (1 - structural_similarity_index_measure(real_B_output + 1, fake_B + 1, data_range=2.0))  # type: ignore
                self.log(f"train/loss_ssim_{name}", loss_ssim.detach())
            loss_paired = self.lambda_paired * (loss_ssim + loss_paired)
        loss_G = loss_G_GAN + loss_paired

        return loss_G, fake_B

    def training_step_D(
        self, discriminator, real_inp, other_img, fake_inp, mask_inp: Tensor | None = None, mask_fake: Tensor | None = None, name=""
    ) -> Tensor:
        if self.opt.linspace:
            real_inp = real_inp[:, : -self.opt.dims]  # remove linspace
        if self.use_paired:
            fake = torch.cat([fake_inp, other_img], dim=1)
            real = torch.cat([real_inp, other_img], dim=1)
        else:
            fake = fake_inp
            real = real_inp

        print("discriminator", fake.shape, real.shape) if self.first else None

        # Fake loss, will be fake_B if unpaired and fake_B||real_A if paired
        pred_fake = discriminator(fake, seg=mask_fake)
        fake_label = torch.zeros((pred_fake.shape[0], 1), device=discriminator.device)
        loss_D_fake = self.criterion_GAN(pred_fake, fake_label).mean()  # is mean really necessary?

        pred_real = discriminator(real, seg=mask_inp)
        real_label = torch.ones((pred_real.shape[0], 1), device=discriminator.device)
        loss_D_real = self.criterion_GAN(pred_real, real_label).mean()
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        self.log(f"train/D_real_{name}", loss_D_real.detach())
        self.log(f"train/D_fake_{name}", loss_D_fake.detach())
        return loss_D


class L1(Loss_Computation):
    def __init__(self, opt: "DAE_Option") -> None:
        super().__init__()
        self.opt = opt
        # define losses
        self.criterion_paired = torch.nn.L1Loss()
        self.first = True
        self.lambda_paired = 1
        self.lambda_ssim = 0

    # @torch.no_grad()
    # def render(self, self_: "DAE_LitModel", noise=None, cond=None, T=20, palette_condition: list[Tensor] | None = None, **qargs):
    #    if palette_condition is None:
    #        palette_condition = [cond]  # type: ignore
    #    return self.forward(self_, palette_condition=palette_condition)

    def loss(self, pl_model: "DAE_LitModel", x_start: torch.Tensor, model_kwargs: dict, train=True):  # noqa: ARG002
        palette_condition = model_kwargs.get("palette_condition")
        lq = model_kwargs.get("x_start_aug", x_start).detach()
        if palette_condition is None:
            lq: torch.Tensor | None = lq
        else:
            lq = torch.cat([lq, *palette_condition], 1)  # type: ignore
        self.loss_dict = {}
        loss1 = self.training_step_G(pl_model.model, lq, x_start, "A2B")  # type: ignore
        self.loss_dict["loss"] = loss1
        self.first = False
        return self.loss_dict

    def training_step_G(self, gan, lq: Tensor, hq: Tensor, name):
        opt = self.opt
        fake_B: Tensor = gan(lq)
        # First, G(A) should fake the discriminator
        real_B_output = hq[:, : -opt.dims] if opt.linspace else hq
        loss_paired = 0
        loss_ssim = 0
        #########################################
        print("use_paired", real_B_output.shape, fake_B.shape) if self.first else None
        loss_paired = self.criterion_paired(real_B_output, fake_B)
        self.log(f"train/loss_paired_{name}", loss_paired.detach())
        if self.lambda_ssim > 0.0:
            loss_ssim = self.lambda_ssim * (1 - structural_similarity_index_measure(real_B_output + 1, fake_B + 1, data_range=2.0))  # type: ignore
            self.log(f"train/loss_ssim_{name}", loss_ssim.detach())
        loss_paired = self.lambda_paired * (loss_ssim + loss_paired)
        return loss_paired
