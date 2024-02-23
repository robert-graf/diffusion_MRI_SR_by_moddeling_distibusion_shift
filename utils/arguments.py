from dataclasses import dataclass, field
from pathlib import Path

from dataloader import transforms as T
from diffusion.beta_schedule import Beta_Schedule
from utils.auto_arguments import Option_to_Dataclass
from utils.enums_model import GenerativeType, LatentNetType, LossType, ModelMeanType, ModelName, ModelType, ModelVarType, OptimizerType

from .enums import Discriminator, EmbeddingType, Inpainting, Models, TrainMode


@dataclass
class Train_Option(Option_to_Dataclass):
    experiment_name: str = "NAKO_256"
    lr: float = 0.0001
    batch_size: int = 64
    batch_size_eval: int = 64
    debug: bool = True
    new: bool = False
    gpus: list[int] | None = None
    num_cpu: int = 16
    # Logging
    log_dir: str = "lightning_logs"
    log_every_n_steps = 3000
    fast_dev_run: bool = False
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0.0
    save_every_samples: int = 25000
    val_check_interval: int = 5000
    early_stopping_patience: int = 100

    def get_sample_dir(self, addendum=""):
        return Path(self.log_dir, self.experiment_name, addendum)

    @property
    def accum_batches(self):
        return 1

    grad_clip: float = 1.0
    limit_val_batches = 1
    monitor: str = "loss/val_loss"

    @property
    def batch_size_effective(self):
        return self.batch_size * self.accum_batches

    total_samples: int = 100000000
    # Validation
    fp32: bool = False
    # Debugging
    overfit: bool = False

    # Default values
    model_channels: int = 64


@dataclass
class DataSet_Option(Train_Option):
    palette_condition: list[str] = field(default_factory=list)  # img, img_lr

    dataset: str = "/media/data/robert/code/nako_embedding/dataset/train.csv"
    dataset_val: str | None = None
    val_niis: list[str] | None = None
    val_niis_pairs: list[tuple[str, str]] | None = None
    val_niis_every_epoch: int = 10
    val_niis_every_epoch_start: int = 25
    ds_type: str = "csv_2D"  # Literal["csv_2D"]
    transforms: list[T.Transforms_Enum] | None = None  # Outdated only 2D and unpaired
    transforms_3D: list[T.Transforms_Enum_3D] | None = None
    in_channels: int = 1  # Channel of the Noised input
    img_size: list[int] = 256  # type: ignore # TODO Diffusion_Autoencoder_Model can't deal with list[int] | list[int]
    dims: int = 2
    linspace: bool = False
    # Super resolution
    sr_target: float = 0.8571
    sr_source: list[float] | None = None
    #
    inpainting: Inpainting | None = None
    seed: int | None = 0
    prob_transforms: float = 0.3
    RandomInterlaceMovementArtifactFactor: float = 2

    @property
    def _names(self):
        a = self.palette_condition.copy()
        if hasattr(self, "side_a"):
            a += self.side_a  # type: ignore
        if hasattr(self, "side_b"):
            a += self.side_b  # type: ignore
        if hasattr(self, "x_start"):
            a += [self.x_start]  # type: ignore
        if hasattr(self, "image_name"):
            a += [self.image_name]  # type: ignore
        return list(set(a))

    @property
    def shape(self):
        if isinstance(self.img_size, int):
            return (self.img_size,) * self.dims
        if len(self.img_size) == 1:
            return self.img_size * self.dims
        if len(self.img_size) != self.dims:
            raise ValueError(
                f"dims ({self.dims}) is different length than image_size ({self.img_size}). Use same length for image_size or an int or update dims"
            )
        return tuple(self.img_size)

    def update(self):
        super().update()
        if self.linspace:
            if hasattr(self, "side_a"):
                self.side_a += [f"linspace_A_{i}" for i in range(self.dims)]
            if hasattr(self, "side_b"):
                self.side_b += [f"linspace_B_{i}" for i in range(self.dims)]

            self.palette_condition += [f"linspace{i}" for i in range(self.dims)]


@dataclass
class DAE_Model_Option:
    # no_dae_embedding: bool = False
    generative_type: GenerativeType = GenerativeType.ddim
    model_type: ModelType = ModelType.autoencoder

    attention_resolutions: list[int] = field(default_factory=lambda: [16])
    net_ch_mult: tuple[int, ...] = field(default_factory=lambda: (1, 1, 2, 2))
    dropout: float = 0.1
    embed_channels: int = 512

    @property
    def enc_out_channels(self):
        return self.embed_channels

    net_enc_pool: EmbeddingType = EmbeddingType.adaptivenonzero  # ConvEmb
    enc_num_res_blocks: int = 2
    enc_channel_mult: tuple[int, ...] = field(default_factory=lambda: (1, 1, 2, 4, 4))
    enc_grad_checkpoint = False
    enc_attn = None

    net_beatgans_attn_head = 1
    net_num_res_blocks = 2
    net_num_input_res_blocks = None
    net_resblock_updown = True
    net_beatgans_gradient_checkpoint = False

    net_beatgans_resnet_use_zero_module = True
    net_beatgans_resnet_cond_channels = None


@dataclass
class DAE_Option(DAE_Model_Option, DataSet_Option):
    image_name: str = None  # type: ignore # condition
    x_start: str = "img"  # type: ignore
    add_palette_condition_to_encoding: bool = False
    target_batch_factor: int = 8
    # Train
    schedular: str = ""
    discriminator: str = ""
    beta1: float = 0.9
    beta2: float = 0.999
    ablation_lvl: int = 7  # -1

    @property
    def model_out_channels(self):
        return self.in_channels

    # DIFFUSION
    beta_schedule: Beta_Schedule = Beta_Schedule.linear
    num_timesteps: int = 1000
    num_timesteps_ddim: int = 20  # | str | list[int]
    model_mean_type: ModelMeanType = ModelMeanType.eps
    model_var_type: ModelVarType = ModelVarType.fixed_large
    loss_type: LossType = LossType.mse
    ema_decay: float = 0.999
    rescale_timesteps: bool = False

    # Embedding
    hessian_penalty: float = 0
    train_mode: TrainMode = TrainMode.diffusion
    pretrain = None
    # Model
    model_name: ModelName = ModelName.autoencoder
    net_latent_net_type = LatentNetType.none

    def update(self):
        super().update()
        if self.image_name is None:
            self.image_name = self.x_start

    @property
    def target_batch_size(self):
        # if hasattr(self, "_target_batch_size"):
        return self.batch_size * self.target_batch_factor
        # gpu_name = torch.cuda.get_device_name(0)
        # _target_batch_size = self.batch_size
        # if self.dims == 2:
        #    if "NVIDIA GeForce RTX 3090" == gpu_name:
        #        max_batch_size_that_fits_in_memory = 16
        #    else:
        #        max_batch_size_that_fits_in_memory = 16
        #    if self.img_size == 128:
        #        max_batch_size_that_fits_in_memory *= 4
        # else:
        #    gpu_name = torch.cuda.get_device_name(0)
        #    if "A40" in gpu_name:
        #        max_batch_size_that_fits_in_memory = 4
        #    else:
        #        max_batch_size_that_fits_in_memory = 1
        #
        #    _target_batch_size = 32
        #
        #    _target_batch_size = _target_batch_size if not self.overfit else max_batch_size_that_fits_in_memory
        #
        # self._target_batch_size = _target_batch_size
        # self.batch_size = min(max_batch_size_that_fits_in_memory, _target_batch_size)
        # return self._target_batch_size

    @property
    def accum_batches(self):
        return self.target_batch_size // self.batch_size

    @property
    def sample_size(self):
        return self.batch_size


@dataclass
class CycleGAN_Option(DataSet_Option):
    # Train
    decay_epoch: int = 30
    lambda_GAN: float = 1.0
    lambda_Discriminator_unbalance: float = 1.0
    lambda_paired: float = 0
    lambda_ssim: float = 1
    # Cycle_GAN only
    lambda_cycle_consistency: float = 10
    lambda_cycle_consistency_deform: float = 0
    lambda_cycle_identity: float = 0
    lambda_deform: float = 0
    lambda_SteGANomaly_barrier: float = 0
    lambda_background_mask: bool = False
    lambda_seg: float = 0
    lambda_seg_disc: bool = False
    deformation_stride: int = 8
    lambda_input_noise_for_sr: float = 0
    nce_idt: bool = True
    # modes: pix2pix
    mode: str = "CycleGAN"  # "pix2pix"
    # Model
    # model_name Options (cut): resnet, base_unet, unet, style
    model_name: Models = Models.unet
    ## Discriminator
    net_D: Discriminator = Discriminator.patch
    net_D_depth: int = 3
    net_D_channel: int = 64
    ## Generator
    net_G_depth: int = 9
    net_G_channel: int = 64
    net_G_downsampling: int = 2
    net_G_drop_out: float = 0.5
    start_epoch: int = 0
    side_a: list[str] = field(default_factory=lambda: ["A"])
    side_b: list[str] = field(default_factory=lambda: ["B"])
    attention_level: list[int] | None = None  # Means all
    monitor: str = "train/All_avg"
    super_res: bool = False

    def seg_discriminator(self):
        if self.lambda_background_mask:
            return 0
        return 0

    # @property
    # def side_b(self) -> list[str]:
    #    if len(self.palette_condition) == 0:
    #        return ["B"]
    #    return self.palette_condition


@dataclass
class Segmentation_Option(DataSet_Option):
    limit_val_batches: None = None

    num_classes: int = 120
    side_a: list[str] = field(default_factory=lambda: ["A"])
    # model_name: Models = Models.unet
    net_G_depth: int = 9
    net_G_channel: int = 64
    net_G_downsampling: int = 2
    net_G_drop_out: float = 0.5
    monitor: str = "train/avg_loss"
    attention_level: list[int] | None = None  # Means all

    @property
    def side_b(self) -> list[str]:
        return ["seg" for i in range(self.num_classes)]


def get_latest_checkpoint(opt: Train_Option, version="*", log_dir_name="lightning_logs", best=False, verbose=True) -> str | None:
    import glob
    import os

    ckpt = "*"
    if best:
        ckpt = "*best*"
    print() if verbose else None
    checkpoints = None

    if isinstance(opt, str) or not opt.new:
        if isinstance(opt, str):
            checkpoints = sorted(
                glob.glob(f"{log_dir_name}/{opt}/version_{version}/checkpoints/{ckpt}.ckpt"),
                key=os.path.getmtime,
            )
        else:
            checkpoints = sorted(
                glob.glob(f"{log_dir_name}/{opt.experiment_name}/version_{version}/checkpoints/{ckpt}.ckpt"),
                key=os.path.getmtime,
            )

        checkpoints = None if len(checkpoints) == 0 else checkpoints[-1]
        print("Reload recent Checkpoint and continue training:", checkpoints) if verbose else None
    else:
        return None

    return checkpoints
