# Source: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.lr_scheduler import _LRScheduler

from models.blocks import TimestepBlock
from models.nn import GroupNorm32, linear, timestep_embedding


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  # noqa: SIM114
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb=None, cond=None, lateral=None):
        for layer in self:
            x = layer(x, emb=emb, cond=cond, lateral=lateral) if isinstance(layer, TimestepBlock) else layer(x)
        return x


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = [basic_block(**kwarg) for _ in range(num_basic_block)]
    return TimestepEmbedSequential(*layers)


def pixel_unshuffle(x, scale):
    """Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class MultiStepRestartLR(_LRScheduler):
    """MultiStep with restarts learning rate scheme.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        restarts (list): Restart iterations. Default: [0].
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self, optimizer, milestones, gamma=0.1, restarts=(0,), restart_weights=(1,), last_epoch=-1):
        from collections import Counter

        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(self.restart_weights), "restarts and their weights do not match."
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group["initial_lr"] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma ** self.milestones[self.last_epoch] for group in self.optimizer.param_groups]


def apply_conditions(
    h,
    emb: torch.Tensor | None,
    cond: torch.Tensor | None,
    scale_bias: float | list[float] = 1,
    in_channels: int = 512,
):
    """
    apply conditions on the feature maps

    Args:
        emb: time conditional (ready to scale + shift)
        cond: encoder's conditional (read to scale + shift)
    """

    ## FIX shape to same lenth
    if emb is not None:
        # adjusting shapes
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]
    if cond is not None:
        # adjusting shapes
        while len(cond.shape) < len(h.shape):
            cond = cond[..., None]
    # MAKE list of shifts
    if emb is not None and cond is not None:
        # time first
        scale_shifts_: list[torch.Tensor | None] = [emb, cond]
    else:
        # "cond" is not used with single cond mode
        scale_shifts_ = [emb]

    # support scale, shift or shift only
    scale_shifts: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
    for _, each in enumerate(scale_shifts_):
        # special case: the condition is not provided
        a = None
        b = None
        if each is not None:
            if each.shape[1] == in_channels * 2:
                a, b = torch.chunk(each, 2, dim=1)
            else:
                a = each
                b = None
        scale_shifts.append((a, b))

    # condition scale bias could be a list
    biases: list[float] = [scale_bias] * len(scale_shifts) if isinstance(scale_bias, Number) else scale_bias  # type: ignore
    # default, the scale & shift are applied after the group norm but BEFORE SiLU
    # spilt the post layer to be able to scale up or down before conv
    # post layers will contain only the conv
    # scale and shift for each condition
    for i, (scale, shift) in enumerate(scale_shifts):
        # if scale is None, it indicates that the condition is not provided
        if scale is not None:
            # print("         scale", h.shape, scale.shape, None if shift is None else shift.shape)
            h = h * (biases[i] + scale)
            if shift is not None:
                h = h + shift
    return h


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(8, channels), channels)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.norm = normalization(num_grow_ch)  # type: ignore
        self.lrelu = nn.SiLU()  # nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x, emb=None, cond=None, lateral=None):  # noqa: ARG002
        x1 = self.lrelu(self.norm(self.conv1(x)))
        x2 = self.lrelu(self.norm(self.conv2(torch.cat((x, x1), 1))))
        x3 = self.lrelu(self.norm(self.conv3(torch.cat((x, x1, x2), 1))))
        x4 = self.lrelu(self.norm(self.conv4(torch.cat((x, x1, x2, x3), 1))))
        x4_ = torch.cat((x, x1, x2, x3, x4))
        x4_ = apply_conditions(h=x4_, emb=emb, cond=cond, scale_bias=1, in_channels=x4_.shape[1])
        x5 = self.conv5(x4_)
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(TimestepBlock):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class RRDBNet_Diffusion(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(
        self, num_in_ch, num_out_ch, scale=1, num_feat=64, num_block=23, num_grow_ch=32, embed_channels=512, time_embed_channels=None
    ):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.time_emb_channels = time_embed_channels or num_feat

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.time_embed = nn.Sequential(linear(self.time_emb_channels, embed_channels), nn.SiLU(), linear(embed_channels, embed_channels))

    def forward(self, x, t=1, palette_condition=None, **qargs):  # type: ignore
        if palette_condition is None or x is not None:
            x: Tensor | None = x
        elif palette_condition is not None or x is None:
            x: Tensor | None = torch.cat(palette_condition, 1)
        else:
            try:
                x = torch.cat([x, *palette_condition], 1)
            except RuntimeError:
                print(x.shape, [y.shape for y in palette_condition])
                raise
        t_emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat, t_emb))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(functional.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(functional.interpolate(feat, scale_factor=2, mode="nearest")))
        feat += x
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
