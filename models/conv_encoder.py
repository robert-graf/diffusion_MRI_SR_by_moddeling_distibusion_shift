from typing import Tuple

import torch
from torch import nn

from utils.enums import EmbeddingType


from .blocks import (
    AttentionBlock,
    Downsample,
    ResBlock,
    ResBlockConfig,
    TimestepEmbedSequential,
    Upsample,
)
from .nn import conv_nd


class ConvEmb(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        # base channels, will be multiplied
        model_channels: int = 64,
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
        attention_resolutions: Tuple[int, ...] | list[int] = (16,),
        # dropout applies to the resblocks (on feature maps)
        dropout: float = 0.1,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        input_channel_mult: Tuple[int] | None = None,
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
    ):
        super().__init__()
        self.num_classes = num_classes
        self.channel_mult = channel_mult
        if num_heads_upsample == -1:
            self.num_heads_upsample = num_heads
        # embedding
        self.dtype = torch.float32

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))])

        kwargs = dict(
            use_condition=False,
            two_cond=None,
            use_zero_module=resnet_use_zero_module,
            # style channels for the resnet block
            cond_emb_channels=resnet_cond_channels,
        )

        self._feature_size = ch

        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(channel_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(channel_mult))]

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
                        **kwargs
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

    def forward(self, x, **kwargs):  # type: ignore
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # hs = []
        hs = []
        h = x.type(self.dtype)
        # hs.append(h)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=None)
                # print(i, j, h.shape)
                hs.append(h)
                k += 1
        # for a, c in enumerate(hs):
        #    print(a, c.shape)
        return hs
