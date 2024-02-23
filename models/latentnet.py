from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple, Tuple, Union

import torch
from torch import nn
from torch.nn import init


from .blocks import *
from .nn import timestep_embedding
from .unet import *


class LatentNetType(str, Enum):
    none = "none"
    # injecting inputs into the hidden layers
    skip = "skip"


class LatentNetReturn(NamedTuple):
    pred: torch.Tensor = None


class MLPSkipNet(nn.Module):
    """
    concat x to hidden layers

    default MLP for the latent DPM in the paper!
    """

    def __init__(
        self,
        num_channels: int,
        skip_layers: Tuple[int, ...],
        num_hid_channels: int,
        num_layers: int,
        num_time_emb_channels: int = 64,
        activation: str = "silu",
        use_norm: bool = True,
        condition_bias: float = 1,
        dropout: float = 0,
        last_act: Union[str, None] = None,
        num_time_layers: int = 2,
        time_last_act: bool = False,
    ):
        super().__init__()
        self.num_time_emb_channels = num_time_emb_channels
        self.skip_layers = skip_layers
        layers = []
        for i in range(num_time_layers):
            if i == 0:
                a = num_time_emb_channels
                b = num_channels
            else:
                a = num_channels
                b = num_channels
            layers.append(nn.Linear(a, b))
            if i < num_time_layers - 1 or time_last_act:
                layers.append(activation.get_act())
        self.time_embed = nn.Sequential(*layers)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            if i == 0:
                act = activation
                norm = use_norm
                cond = True
                a, b = num_channels, num_hid_channels
                dropout = dropout
            elif i == num_layers - 1:
                act = None
                norm = False
                cond = False
                a, b = num_hid_channels, num_channels
                dropout = 0
            else:
                act = activation
                norm = use_norm
                cond = True
                a, b = num_hid_channels, num_hid_channels
                dropout = dropout

            if i in skip_layers:
                a += num_channels

            self.layers.append(
                MLPLNAct(
                    a,
                    b,
                    norm=norm,
                    activation=act,
                    cond_channels=num_channels,
                    use_cond=cond,
                    condition_bias=condition_bias,
                    dropout=dropout,
                )
            )
        self.last_act = last_act.get_act()

    def forward(self, x, t, **kwargs):
        t = timestep_embedding(t, self.num_time_emb_channels)
        cond = self.time_embed(t)
        h = x
        for i in range(len(self.layers)):
            if i in self.skip_layers:
                # injecting input into the hidden layers
                h = torch.cat([h, x], dim=1)
            h = self.layers[i].forward(x=h, cond=cond)
        h = self.last_act(h)
        return LatentNetReturn(h)


class MLPLNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool,
        use_cond: bool,
        activation: str,
        cond_channels: int,
        condition_bias: float = 0,
        dropout: float = 0,
    ):
        super().__init__()
        self.activation = activation
        self.condition_bias = condition_bias
        self.use_cond = use_cond

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = activation.get_act()
        if self.use_cond:
            self.linear_emb = nn.Linear(cond_channels, out_channels)
            self.cond_layers = nn.Sequential(self.act, self.linear_emb)
        if norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation == "relu":
                    init.kaiming_normal_(module.weight, a=0, nonlinearity="relu")
                elif self.activation == "lrelu":
                    init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
                elif self.activation == "silu":
                    init.kaiming_normal_(module.weight, a=0, nonlinearity="relu")
                else:
                    # leave it as default
                    pass

    def forward(self, x, cond=None):
        x = self.linear(x)
        if self.use_cond:
            # (n, c) or (n, c * 2)
            cond = self.cond_layers(cond)
            cond = (cond, None)

            # scale shift first
            x = x * (self.condition_bias + cond[0])
            if cond[1] is not None:
                x = x + cond[1]
            # then norm
            x = self.norm(x)
        else:
            # no condition
            x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
