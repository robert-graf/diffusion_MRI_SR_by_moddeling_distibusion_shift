from typing import TYPE_CHECKING

from models import Model
from models.cycleGAN.cyGAN_model import UNetDiscriminatorSN
from models.srmodel.RRDBNet import RRDBNet
from models.srmodel.RRDBNet_diffusion import RRDBNet_Diffusion
from models.unet import BeatGANsUNetModel
from models.unet_with_encoder import Diffusion_Autoencoder_Model
from utils.enums_model import LatentNetType, ModelName, ModelType

if TYPE_CHECKING:
    from utils.arguments import DAE_Option  # Prevent cyclic imports
else:
    DAE_Option = object()


def get_model(opt: DAE_Option) -> Model:
    if not hasattr(opt, "palette_condition"):  # Legacy
        opt.palette_condition = []
    img_size = opt.img_size
    if isinstance(img_size, list) and len(img_size) == 1:
        img_size = img_size[0]

    # assert isinstance(img_size, int), img_size
    # if isinstance(int) else opt.
    ### UNIFY partial Settings ###
    if hasattr(opt, "no_dae_embedding") and opt.no_dae_embedding:  # Legacy (use palette_only) # type: ignore
        print("no_embedding")
        opt.model_type = ModelType.palette_only
        opt.model_name = ModelName.beatgans_ddpm
    elif opt.model_type == ModelType.palette_only:
        opt.model_name = ModelName.beatgans_ddpm
        opt.no_dae_embedding = True  # type: ignore
    # elif opt.model_name == ModelName.autoencoder:
    #    opt.model_type = ModelType.autoencoder
    if opt.model_name == ModelName.beatgans_ddpm:
        opt.model_type = ModelType.ddpm
    # else:
    #    raise NotImplementedError()
    if opt.net_latent_net_type == LatentNetType.none:
        latent_net_conf = None
    elif opt.net_latent_net_type == LatentNetType.skip:
        raise NotImplementedError(opt.net_latent_net_type)
        # latent_net_conf = MLPSkipNetConfig(
        #    num_channels=opt.style_ch,
        #    skip_layers=opt.net_latent_skip_layers,
        #    num_hid_channels=opt.net_latent_num_hid_channels,
        #    num_layers=opt.net_latent_layers,
        #    num_time_emb_channels=opt.net_latent_time_emb_channels,
        #    activation=opt.net_latent_activation,
        #    use_norm=opt.net_latent_use_norm,
        #    condition_bias=opt.net_latent_condition_bias,
        #    dropout=opt.net_latent_dropout,
        #    last_act=opt.net_latent_net_last_act,
        #    num_time_layers=opt.net_latent_num_time_layers,
        #    time_last_act=opt.net_latent_time_last_act,
        # )
    else:
        raise NotImplementedError(opt.net_latent_net_type)
    if opt.model_type in [ModelType.RRDBNet]:
        return RRDBNet((len(opt.palette_condition) + 1) * opt.in_channels, opt.in_channels, num_feat=64, num_block=23, num_grow_ch=32)
    elif opt.model_type in [ModelType.RCAN]:
        from models.srmodel.RCAN import RCAN, Model, RCAN_Settings

        return RCAN(RCAN_Settings())
        return Model(RCAN_Settings())
    ### FORK Models ###
    elif opt.model_type in [ModelType.RRDBNet_diffusion]:
        print("RRDBNet_Diffusion")
        return RRDBNet_Diffusion(
            num_in_ch=opt.in_channels + len(opt.palette_condition) * opt.in_channels,
            num_out_ch=opt.in_channels,
            num_feat=opt.model_channels,
        )
    elif opt.model_type in [ModelType.ddpm, ModelType.palette_only, ModelType.Pix2Pix]:
        print("BeatGANsUNetModel")
        return BeatGANsUNetModel(
            attention_resolutions=opt.attention_resolutions,
            channel_mult=opt.net_ch_mult,
            conv_resample=True,
            dims=opt.dims,
            dropout=opt.dropout,
            embed_channels=opt.embed_channels,
            image_size=img_size,
            in_channels=opt.in_channels,
            model_channels=opt.model_channels,
            num_classes=None,
            num_head_channels=-1,
            num_heads_upsample=-1,
            num_heads=opt.net_beatgans_attn_head,
            num_res_blocks=opt.net_num_res_blocks,
            num_input_res_blocks=opt.net_num_input_res_blocks,
            out_channels=opt.model_out_channels,
            resblock_updown=opt.net_resblock_updown,
            use_checkpoint=opt.net_beatgans_gradient_checkpoint,
            use_new_attention_order=False,
            resnet_two_cond=None,
            resnet_use_zero_module=opt.net_beatgans_resnet_use_zero_module,
            resnet_cond_channels=opt.net_beatgans_resnet_cond_channels,
            palette_condition_channels=len(opt.palette_condition) * opt.in_channels,
        )
    elif opt.model_type in [ModelType.autoencoder]:
        print("Diffusion_Autoencoder_Model")
        return Diffusion_Autoencoder_Model(
            attention_resolutions=opt.attention_resolutions,
            channel_mult=opt.net_ch_mult,
            conv_resample=True,
            dims=opt.dims,
            dropout=opt.dropout,
            embed_channels=opt.embed_channels,
            enc_out_channels=opt.enc_out_channels,
            enc_pool=opt.net_enc_pool,
            enc_num_res_block=opt.enc_num_res_blocks,
            enc_channel_mult=opt.enc_channel_mult,
            enc_grad_checkpoint=opt.enc_grad_checkpoint,
            enc_attn_resolutions=opt.enc_attn,
            image_size=img_size,
            in_channels=opt.in_channels,
            model_channels=opt.model_channels,
            num_classes=None,
            num_head_channels=-1,
            num_heads_upsample=-1,
            num_heads=opt.net_beatgans_attn_head,
            num_res_blocks=opt.net_num_res_blocks,
            num_input_res_blocks=opt.net_num_input_res_blocks,
            out_channels=opt.model_out_channels,
            resblock_updown=opt.net_resblock_updown,
            use_checkpoint=opt.net_beatgans_gradient_checkpoint,
            use_new_attention_order=False,
            resnet_use_zero_module=opt.net_beatgans_resnet_use_zero_module,
            latent_net=latent_net_conf,
            resnet_cond_channels=opt.net_beatgans_resnet_cond_channels,
            palette_condition_channels=len(opt.palette_condition) * opt.in_channels,
            add_palette_condition_to_encoding=opt.add_palette_condition_to_encoding,
        )
    else:
        raise NotImplementedError(opt.model_name, opt.model_type)

    return opt.model_conf


def get_discriminator(opt: DAE_Option):
    if opt.discriminator == "UNetDiscriminatorSN":
        return UNetDiscriminatorSN(opt.in_channels, num_feat=64, skip_connection=True)
    elif opt.discriminator.lower() == "patch":
        from models.cycleGAN.cyGAN_model import Discriminator

        return Discriminator(opt.in_channels, depth=4, channels=64)

    raise NotImplementedError(opt.discriminator)
