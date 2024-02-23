from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.enums import EmbeddingType

from .blocks import AttentionBlock, Downsample, ResBlock, ResBlockConfig, TimestepEmbedSequential, Upsample
from .nn import adaptive_avg_pool_nd, conv_nd, linear, normalization, timestep_embedding, zero_module


class BeatGANsUNetModel(nn.Module):
    def __init__(
        self,
        image_size: int | list[int] = 64,
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
        attention_resolutions: tuple[int, ...] | list[int] = (16,),
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
        resnet_two_cond: EmbeddingType | None = None,
        resnet_cond_channels: int | None = None,
        # init the decoding conv layers with zero weights, this speeds up training
        # default: True (BeattGANs)
        resnet_use_zero_module: bool = True,
        # gradient checkpoint the attention operation
        attn_checkpoint: bool = False,
        palette_condition_channels: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.channel_mult = channel_mult
        if num_heads_upsample == -1:
            self.num_heads_upsample = num_heads
        # embedding
        self.dtype = torch.float32
        self.time_emb_channels = time_embed_channels or model_channels
        self.time_embed = nn.Sequential(linear(self.time_emb_channels, embed_channels), nn.SiLU(), linear(embed_channels, embed_channels))

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, embed_channels)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels + palette_condition_channels, ch, 3, padding=1))]
        )

        kwargs = {
            "use_condition": True,
            "two_cond": resnet_two_cond,
            "use_zero_module": resnet_use_zero_module,
            # style channels for the resnet block
            "cond_emb_channels": resnet_cond_channels,
        }

        self._feature_size = ch

        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(channel_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(channel_mult))]

        ds = 1
        resolution = min(image_size) if isinstance(image_size, (list, tuple)) else image_size
        for level, mult in enumerate(input_channel_mult or channel_mult):
            for _ in range(num_input_res_blocks or num_res_blocks):
                layers: list[ResBlock | AttentionBlock | Upsample] = [
                    ResBlockConfig(
                        ch,
                        embed_channels,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(mult * model_channels)
                if resolution in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint or attn_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
                # print(input_block_chans)
            if level != len(channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch, embed_channels, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, down=True, **kwargs
                        ).make_model()
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                # input_block_chans.append(ch)
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(ch, embed_channels, dropout, dims=dims, use_checkpoint=use_checkpoint, **kwargs).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint or attn_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlockConfig(ch, embed_channels, dropout, dims=dims, use_checkpoint=use_checkpoint, **kwargs).make_model(),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # print(input_block_chans)
                # ich = input_block_chans.pop()
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connecions for all decoder blocks
                    ich = 0
                # print('pop:', ich)
                layers = [
                    ResBlockConfig(
                        # only direct channels when gated
                        channels=ch + ich,
                        emb_channels=embed_channels,
                        dropout=dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        # lateral channels are described here when gated
                        has_lateral=ich > 0,
                        lateral_channels=None,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(model_channels * mult)
                if resolution in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint or attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch, embed_channels, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, up=True, **kwargs
                        ).make_model()
                        if (resblock_updown)
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        # print(input_block_chans)
        # print('inputs:', self.input_num_blocks)
        # print('outputs:', self.output_num_blocks)

        if resnet_use_zero_module:
            self.out = nn.Sequential(normalization(ch), nn.SiLU(), zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)))
        else:
            self.out = nn.Sequential(normalization(ch), nn.SiLU(), conv_nd(dims, input_ch, out_channels, 3, padding=1))

    def forward(self, x, t=0, palette_condition=None, **kwargs):  # type: ignore
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if palette_condition is None and x is not None:
            x = x
        elif palette_condition is not None and x is None:
            x: Tensor | None = torch.cat(palette_condition, 1)

        else:
            try:
                x = torch.cat([x, *palette_condition], 1)  # type: ignore
            except RuntimeError:
                print(x.shape, [y.shape for y in palette_condition])
                raise

        # hs = []
        hs = [[] for _ in range(len(self.channel_mult))]
        t_emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        if self.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # new code supports input_num_blocks != output_num_blocks
        h = x.type(self.dtype)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=t_emb)
                # print(i, j, h.shape)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)

        h = self.middle_block(h, emb=t_emb)
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)
                h = self.output_blocks[k](h, emb=t_emb, lateral=lateral)
                k += 1

        h = h.type(x.dtype)
        pred = self.out(h)
        return Return(pred=pred)


class BeatGANsEncoderModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size: int | list[int],
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: tuple[int],
        dropout: float = 0,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        use_time_condition: bool = True,
        conv_resample: bool = True,
        dims: int = 2,
        use_checkpoint: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        pool: EmbeddingType = EmbeddingType.adaptivenonzero,
    ):
        super().__init__()
        self.dtype = torch.float32
        self.use_time_condition = use_time_condition
        self.pool = pool
        if use_time_condition:
            time_embed_dim = model_channels * 4
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        resolution = min(image_size) if isinstance(image_size, (list, tuple)) else image_size
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[ResBlock | AttentionBlock | Upsample] = [
                    ResBlockConfig(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_condition=use_time_condition,
                        use_checkpoint=use_checkpoint,
                    ).make_model()
                ]
                ch = int(mult * model_channels)
                if resolution in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_condition=use_time_condition,
                            use_checkpoint=use_checkpoint,
                            down=True,
                        ).make_model()
                        if (resblock_updown)
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_condition=use_time_condition,
                use_checkpoint=use_checkpoint,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlockConfig(
                ch, time_embed_dim, dropout, dims=dims, use_condition=use_time_condition, use_checkpoint=use_checkpoint
            ).make_model(),
        )
        self._feature_size += ch
        if pool == EmbeddingType.adaptivenonzero or pool == EmbeddingType.adaptivenonzero.name:
            self.out = nn.Sequential(
                normalization(ch), nn.SiLU(), adaptive_avg_pool_nd(dims, output_size=1), conv_nd(dims, ch, out_channels, 1), nn.Flatten()
            )
        # elif pool == EmbeddingType.ConvEmb:
        #    self.out = nn.Sequential(
        #        normalization(ch),
        #        nn.SiLU(),
        #        conv_nd(dims, ch, out_channels, 1),
        #        nn.Flatten(),
        #    )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def forward(self, x, t=None, return_2d_feature=False):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(t, self.model_channels)) if self.use_time_condition else None

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb=emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb=emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = torch.cat(results, 1)
        else:
            h = h.type(x.dtype)

        h_2d = h
        # B x 512 x [4x4(x4)]
        h = self.out(h)
        # B x 512

        if return_2d_feature:
            return h, h_2d
        else:
            return h

    def forward_flatten(self, x):
        """
        transform the last 2d feature into a flatten vector
        """
        h = self.out(x)
        return h


class SuperResModel(BeatGANsUNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = torch.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class Return(NamedTuple):
    pred: Tensor

    @property
    def cond_emb(self):
        return None
