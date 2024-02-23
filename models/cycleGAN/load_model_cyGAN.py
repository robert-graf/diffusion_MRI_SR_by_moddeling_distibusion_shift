from torch.nn import Module

from models.cycleGAN.cyGAN_model import ChannelNormPatchDiscriminator, Discriminator, Discriminator3D, UNetDiscriminatorSN
from utils.arguments import CycleGAN_Option, Segmentation_Option
from utils.enums import Models


def _get_channels(opt: CycleGAN_Option | Segmentation_Option, in_channel, reverse_gan: bool):
    input_c = len(opt.side_b) if reverse_gan else len(opt.side_a)
    out_c = len(opt.side_a) if reverse_gan else len(opt.side_b)
    if opt.linspace:
        out_c -= 3
    num_cond = input_c
    return (input_c * in_channel, out_c * in_channel, num_cond)


def load_generator(opt: CycleGAN_Option | Segmentation_Option, in_channel, reverse_gan=False) -> Module:
    in_c, out_c, num_cond = _get_channels(opt, in_channel, reverse_gan)
    if opt.dims == 3:
        print("3D Unet")
        from models.diffusion_unet3D import Unet

        return Unet(
            dim=opt.model_channels,
            dim_mults=(1, 2, 4, 8),  # get_option(opt, "dim_multiples", (1, 2, 4, 8), separated_list=True),
            channels=in_channel,
            conditional_dimensions=num_cond - 1,  # additional conditions images beyond the first one
            learned_variance=False,
            conditional_label_size=0,
            conditional_embedding_size=0,
            out_dim=out_c,
        )  # type: ignore

    # 2D
    if not isinstance(opt.model_name, str):
        opt.model_name = opt.model_name.value
    if opt.model_name == Models.resnet.value:  # Default 2D case
        from models.cycleGAN.cyGAN_model import Generator

        return Generator(in_c, out_c, **opt.__dict__)
    if opt.model_name == Models.unet.value:
        from models.diffusion_unet import Unet

        return Unet(
            dim=opt.model_channels,
            dim_mults=(1, 2, 4, 8),  # get_option(opt, "dim_multiples", (1, 2, 4, 8), separated_list=True),
            channels=in_channel,
            conditional_dimensions=num_cond - 1,  # additional conditions images beyond the first one
            learned_variance=False,
            conditional_label_size=0,
            conditional_embedding_size=0,
            out_dim=out_c,
            attention_layer=opt.attention_level,
        )  # type: ignore

    raise NotImplementedError(opt.model_name)


def load_seg_model(opt: CycleGAN_Option | Segmentation_Option, in_channel, out_channel):
    assert len(opt.shape) == 2, opt.shape
    from models.diffusion_unet import Unet

    return Unet(
        dim=opt.model_channels,
        dim_mults=(1, 2, 4),  # get_option(opt, "dim_multiples", (1, 2, 4, 8), separated_list=True),
        channels=in_channel,
        conditional_dimensions=0,  # additional conditions images beyond the first one
        learned_variance=False,
        conditional_label_size=0,
        conditional_embedding_size=0,
        out_dim=out_channel,
        attention_layer=opt.attention_level,
    )  # type: ignore


def load_discriminator(opt: CycleGAN_Option, in_channel, use_paired, b_side=False, additional_channels=0) -> Module:
    from utils.enums import Discriminator as D

    in_c, out_c, num_cond = _get_channels(opt, in_channel, b_side)

    num_cond = (in_c + out_c if use_paired else out_c) + additional_channels
    # 3D
    num_cond += opt.seg_discriminator()
    if D.patch.value == opt.net_D.value:
        if opt.dims == 3:
            return Discriminator3D(num_cond, depth=opt.net_D_depth, channels=opt.net_D_channel)
        return Discriminator(num_cond, depth=opt.net_D_depth, channels=opt.net_D_channel)
    elif D.channel_norm_patch.value == opt.net_D.value:
        return ChannelNormPatchDiscriminator(num_cond, opt)
    elif D.UNetDiscriminatorSN.value == opt.net_D.value:
        return UNetDiscriminatorSN(num_cond, num_feat=opt.net_D_channel)
    raise NotImplementedError(opt.net_D)
