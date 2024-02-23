import torch

from diffusion.ddim_sampler import Gaussian_Diffusion
from models import Model
from utils.arguments import DAE_Option
from utils.enums import TrainMode


def render_uncondition(
    conf: DAE_Option,
    model: Model,
    x_T,
    sampler: Gaussian_Diffusion,
    latent_sampler: None,
    conds_mean=None,
    conds_std=None,
    clip_latent_noise: bool = False,
):
    device = x_T.device
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.can_sample(), conf.model_type
        return sampler.sample(model=model, noise=x_T)
    elif conf.train_mode.is_latent_diffusion():
        raise NotImplementedError()
        if conf.train_mode == TrainMode.latent_diffusion:
            latent_noise = torch.randn(len(x_T), conf.style_ch, device=device)
        else:
            raise NotImplementedError()

        if clip_latent_noise:
            latent_noise = latent_noise.clip(-1, 1)

        cond = latent_sampler.sample(
            model=model.latent_net,
            noise=latent_noise,
            clip_denoised=conf.latent_clip_sample,
        )

        if conf.latent_znormalize:
            cond = cond * conds_std.to(device) + conds_mean.to(device)

        # the diffusion on the model
        return sampler.sample(model=model, noise=x_T, cond=cond)
    else:
        raise NotImplementedError()


def render_condition(
    conf: DAE_Option, model: Model, noise, sampler: Gaussian_Diffusion, x_start=None, cond=None, palette_condition: list | None = None
):
    if conf.train_mode == TrainMode.diffusion:
        assert conf.model_type.has_autoenc() or palette_condition is not None
        # returns {'cond', 'cond2'}
        if conf.model_type.has_encoder(model) and cond is None:
            cond = model.encode(x_start)
        return sampler.sample(model=model, noise=noise, cond=cond, x_start=x_start, palette_condition=palette_condition)
    else:
        raise NotImplementedError()
