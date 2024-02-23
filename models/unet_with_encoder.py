from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from models.conv_encoder import ConvEmb
from utils.enums import EmbeddingType

from .latentnet import MLPSkipNet
from .nn import linear
from .unet import BeatGANsEncoderModel, BeatGANsUNetModel, ResBlock, timestep_embedding


class Diffusion_Autoencoder_Model(BeatGANsUNetModel):
    def __init__(
        self,
        enc_out_channels: int = 512,
        enc_attn_resolutions: tuple[int, ...] | None = None,
        enc_pool: EmbeddingType = EmbeddingType.adaptivenonzero,
        enc_num_res_block: int = 2,
        enc_channel_mult: tuple[int, ...] | None = None,
        enc_grad_checkpoint: bool = False,
        latent_net: MLPSkipNet | None = None,
        image_size: int = 64,
        in_channels: int = 3,
        # base channels, will be multiplied
        model_channels: int = 64,
        # output of the unet
        # suggest: 3
        # you only need 6 if you also model the variance of the noise prediction (usually we use an analytical variance hence 3)
        out_channels: int = 3,
        # how many repeating resblocks per resolution
        # the decoding side would have "one more" resblock
        # default: 2
        num_res_blocks: int = 2,
        # you can also set the number of resblocks specifically for the input blocks
        # default: None = above
        num_input_res_blocks: int | None = None,
        # number of time embed channels and style channels
        embed_channels: int = 512,
        # at what resolutions you want to do self-attention of the feature maps
        # attentions generally improve performance
        # default: [16]
        # beatgans: [32, 16, 8]
        attention_resolutions: list[int] = [16],
        # number of time embed channels
        time_embed_channels: int | None = None,
        # dropout applies to the resblocks (on feature maps)
        dropout: float = 0.1,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        input_channel_mult: tuple[int] | None = None,
        conv_resample: bool = True,
        # always 2 = 2d conv
        dims: int = 2,
        # don't use this, legacy from BeatGANs
        num_classes: int | None = None,
        use_checkpoint: bool = False,
        # number of attention heads
        num_heads: int = 1,
        # or specify the number of channels per attention head
        num_head_channels: int = -1,
        # what's this?
        num_heads_upsample: int = -1,
        # use resblock for upscale/downscale blocks (expensive)
        # default: True (BeatGANs)
        resblock_updown: bool = True,
        # never tried
        use_new_attention_order: bool = False,
        resnet_cond_channels: int | None = None,
        # init the decoding conv layers with zero weights, this speeds up training
        # default: True (BeattGANs)
        resnet_use_zero_module: bool = True,
        # gradient checkpoint the attention operation
        attn_checkpoint: bool = False,
        palette_condition_channels: int = 0,
        add_palette_condition_to_encoding: bool = False,
    ):
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            num_input_res_blocks=num_input_res_blocks,
            embed_channels=embed_channels,
            attention_resolutions=attention_resolutions,
            time_embed_channels=time_embed_channels,
            dropout=dropout,
            channel_mult=channel_mult,
            input_channel_mult=input_channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            resnet_two_cond=enc_pool,
            resnet_cond_channels=resnet_cond_channels,
            resnet_use_zero_module=resnet_use_zero_module,
            attn_checkpoint=attn_checkpoint,
            palette_condition_channels=palette_condition_channels,
        )
        self.model_channels = model_channels
        self.add_palette_condition_to_encoding = add_palette_condition_to_encoding
        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=model_channels,
            time_out_channels=embed_channels,
        )
        emb_in_channels = in_channels if not add_palette_condition_to_encoding else in_channels + palette_condition_channels
        if enc_pool == EmbeddingType.ConvEmb:
            self.encoder = ConvEmb(
                image_size=image_size,
                in_channels=emb_in_channels,
                model_channels=model_channels,
                num_res_blocks=num_res_blocks,
                num_input_res_blocks=num_input_res_blocks,
                embed_channels=embed_channels,
                attention_resolutions=attention_resolutions,
                dropout=dropout,
                channel_mult=channel_mult,
                input_channel_mult=input_channel_mult,
                conv_resample=conv_resample,
                dims=dims,
                num_classes=num_classes,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                num_heads_upsample=num_heads_upsample,
                resblock_updown=resblock_updown,
                use_new_attention_order=use_new_attention_order,
                resnet_cond_channels=resnet_cond_channels,
                resnet_use_zero_module=resnet_use_zero_module,
                attn_checkpoint=attn_checkpoint,
            )
        elif enc_pool == EmbeddingType.adaptivenonzero or enc_pool == EmbeddingType.adaptivenonzero.name:
            self.encoder = BeatGANsEncoderModel(
                image_size=image_size,
                in_channels=in_channels,
                model_channels=model_channels,
                out_channels=enc_out_channels,
                num_res_blocks=enc_num_res_block,
                attention_resolutions=(enc_attn_resolutions or attention_resolutions),  # type: ignore
                dropout=dropout,
                channel_mult=enc_channel_mult or channel_mult,
                use_time_condition=False,
                conv_resample=conv_resample,
                dims=dims,
                use_checkpoint=use_checkpoint or enc_grad_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                resblock_updown=resblock_updown,
                use_new_attention_order=use_new_attention_order,
                pool=enc_pool,
            )
        else:
            raise NotImplementedError(enc_pool)
        if latent_net is not None:
            self.latent_net = latent_net

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #    """
    #    Reparameterization trick to sample from N(mu, var) from
    #    N(0,1).
    #    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #    :return: (Tensor) [B x D]
    #    """
    #    assert self.is_stochastic
    #    std = torch.exp(0.5 * logvar)
    #    eps = torch.randn_like(std)
    #    return eps * std + mu
    #
    # def sample_z(self, n: int, device):
    #    assert self.is_stochastic
    #    return torch.randn(n, self.conf.enc_out_channels, device=device)

    # def noise_to_cond(self, noise: Tensor):
    #    raise NotImplementedError()
    #    assert self.conf.noise_net_conf is not None
    #    return self.noise_net.forward(noise)

    def encode(self, x) -> Tensor:
        return self.encoder.forward(x)  # type: ignore

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])  # type: ignore
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S

    def forward(self, x, t, y=None, x_start=None, cond=None, noise=None, t_cond=None, palette_condition=None, **kwargs):  # type: ignore
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """
        if palette_condition is None or x is None:
            x: Tensor | None = x
        else:
            x = torch.cat([x] + palette_condition, 1)

        if t_cond is None:
            t_cond = t

        if noise is not None:
            raise NotImplementedError()
            # if the noise is given, we predict the cond from noise
            cond_emb = self.noise_to_cond(noise)

        if cond is None:
            if x is not None:
                assert x_start is not None
                assert len(x) == len(x_start), f"{len(x)} != {len(x_start)}"

            # get augmented version of x_start if given
            x_start_enc = kwargs.get("x_start_aug", x_start)
            if self.add_palette_condition_to_encoding:
                assert palette_condition is not None
                x_start_enc = torch.cat([x_start_enc] + palette_condition, 1)

            cond_emb = self.encode(x_start_enc)
        else:
            cond_emb = cond
        if t is None:
            raise NotImplementedError(t)
        _t_emb = timestep_embedding(t, self.model_channels)
        _t_cond_emb = timestep_embedding(t_cond, self.model_channels)
        emb = self.time_embed.forward(time_emb=_t_emb, time_cond_emb=_t_cond_emb).time_emb

        assert (y is not None) == (self.num_classes is not None), "must specify y if and only if the model is class-conditional"

        if self.num_classes is not None:
            raise NotImplementedError()

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        # where in the model to supply style conditions
        if isinstance(cond_emb, (tuple, list)):
            enc_cond_emb = cond_emb
            mid_cond_emb = cond_emb[-1]
            dec_cond_emb = None
            _m = len(cond_emb) - 1
        else:
            enc_cond_emb = cond_emb
            mid_cond_emb = cond_emb
            dec_cond_emb = cond_emb
            _m = 0

        hs = [[] for _ in range(len(self.channel_mult))]

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0

            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    if isinstance(enc_cond_emb, (tuple, list)):
                        e = enc_cond_emb[min(k, _m)]
                    else:
                        e = enc_time_emb
                    h = self.input_blocks[k](h, emb=enc_time_emb, cond=e)
                    hs[i].append(h)
                    k += 1
            assert k == len(self.input_blocks)

            # middle blocks
            h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.channel_mult))]
        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                if dec_cond_emb is None:
                    e = None
                else:
                    e = dec_cond_emb
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                except IndexError:
                    lateral = None
                h = self.output_blocks[k](h, emb=dec_time_emb, cond=e, lateral=lateral)
                k += 1
        pred = self.out(h)
        return AutoencReturn(pred=pred, cond_emb=cond_emb)


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond_emb: Tensor | None = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor | None = None
    # time only
    time_emb: Tensor | None = None
    # style only (but could depend on time)
    style: Tensor | None = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        if cond is None:
            style = None
        else:
            style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)
