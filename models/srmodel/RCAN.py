import math
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn


class RCAN_Settings:
    scale = [1]  # noqa: RUF012
    self_ensemble = False
    chop = True
    cpu = False
    pre_train = ""
    resume = 0
    n_res_groups = 10
    n_resblocks = 20
    n_feats = 64
    n_colors = 1
    reduction = 16
    res_scale = 1


# Source: https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/__init__.py
class Model(nn.Module):
    def __init__(self, args: RCAN_Settings, ckp=None):
        super().__init__()
        print("Making model...")

        self.scale = args.scale
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.device = torch.device("cpu" if args.cpu else "cuda")

        self.model = make_model(args)

        # if ckp is not None:
        #    self.load(ckp.dir, pre_train=args.pre_train, resume=args.resume, cpu=args.cpu)

    def forward(self, x):
        # self.idx_scale = idx_scale
        target = self.model
        if hasattr(target, "set_scale"):
            target.set_scale(self.idx_scale)  # type: ignore

        if self.self_ensemble and not self.training:
            forward_function = self.forward_chop if self.chop else self.model.forward

            return self.forward_x8(x, forward_function)
        elif self.chop and not self.training:
            return self.forward_chop(x)
        else:
            return self.model(x)

    # def get_model(self) -> "RCAN":
    #    # if self.n_GPUs == 1:
    #    return self.model
    #    # else:
    #    #    return self.model.module  # type: ignore

    # def state_dict(self, **kwargs):
    #    target = self.get_model()
    #    return target.state_dict(**kwargs)

    # def save(self, a_path, epoch, is_best=False):
    #    target = self.get_model()
    #    torch.save(target.state_dict(), Path(a_path, "model", "model_latest.pt"))
    #    if is_best:
    #        torch.save(target.state_dict(), Path(a_path, "model", "model_best.pt"))
    #
    #    if self.save_models:
    #        torch.save(target.state_dict(), Path(a_path, "model", f"model_{epoch}.pt"))

    # def load(self, a_path, pre_train=".", resume=-1, cpu=False):
    #    kwargs = {"map_location": lambda storage, _: storage} if cpu else {}
    #
    #    if resume == -1:
    #        self.get_model().load_state_dict(torch.load(Path(a_path, "model", "model_latest.pt"), **kwargs), strict=False)  # type: ignore
    #    elif resume == 0:
    #        if pre_train != ".":
    #            print(f"Loading model from {pre_train}")
    #            self.get_model().load_state_dict(torch.load(pre_train, **kwargs), strict=False)  # type: ignore
    #    else:
    #        self.get_model().load_state_dict(torch.load(Path(a_path, "model", f"model_{resume}.pt"), **kwargs), strict=False)  # type: ignore

    def forward_chop(self, x, shave=10, min_size=160000):
        scale = self.scale[self.idx_scale]
        n_GPUs = 1  # min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size) : w],
            x[:, :, (h - h_size) : h, 0:w_size],
            x[:, :, (h - h_size) : h, (w - w_size) : w],
        ]

        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i : (i + n_GPUs)], dim=0)
                sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [self.forward_chop(patch, shave=shave, min_size=min_size) for patch in lr_list]

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, c, h, w)
        output[:, :, 0:h_half, 0:w_half] = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] = sr_list[1][:, :, 0:h_half, (w_size - w + w_half) : w_size]
        output[:, :, h_half:h, 0:w_half] = sr_list[2][:, :, (h_size - h + h_half) : h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] = sr_list[3][:, :, (h_size - h + h_half) : h_size, (w_size - w + w_half) : w_size]

        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            if self.precision != "single":
                v = v.float()

            v2np = v.data.cpu().numpy()
            if op == "v":
                tf_np = v2np[:, :, :, ::-1].copy()
            elif op == "h":
                tf_np = v2np[:, :, ::-1, :].copy()
            elif op == "t":
                tf_np = v2np.transpose((0, 1, 3, 2)).copy()
            else:
                raise NotImplementedError()

            ret = torch.Tensor(tf_np).to(self.device)
            if self.precision == "half":
                ret = ret.half()

            return ret

        lr_list = [x]
        for tf in "v", "h", "t":
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], "t")
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], "h")
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], "v")

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output


def make_model(args: RCAN_Settings):
    return RCAN(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):  # noqa: B008
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):  # noqa: ARG002
        super().__init__()
        modules_body = []
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args: RCAN_Settings, conv=default_conv):
        super().__init__()
        reduction = False
        n_res_groups = args.n_res_groups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        # rgb_mean = (0.5, 0.5, 0.5)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks)
            for _ in range(n_res_groups)
        ]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, args.n_colors, kernel_size)]

        # self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find("tail") >= 0:
                        print("Replace pre-trained upsampler to new one...")
                    else:
                        raise RuntimeError(  # noqa: B904, TRY200
                            f"While copying the parameter named {name}, "
                            f"whose dimensions in the model are {own_state[name].size()} and "
                            f"whose dimensions in the checkpoint are {param.size()}."
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError(f'missing keys in state_dict: "{missing}"')


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)  # type: ignore
        self.bias.data.div_(std)  # type: ignore
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, act=nn.ReLU(True)):  # noqa: B008
        m: list = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), stride=stride, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super().__init__(*m)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):  # noqa: B008
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act: Callable | Literal[False] = False, bias=True):  # type: ignore
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feat))
                if act:
                    m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if act:
                m.append(act())
        else:
            raise NotImplementedError

        super().__init__(*m)


if __name__ == "__main__":
    from torchsummary import summary

    m = Model(RCAN_Settings())
    summary(m, (1, 160, 160))
