from enum import Enum
from typing import TypeGuard

from models import Model
from models.unet_with_encoder import Diffusion_Autoencoder_Model


class LatentNetType(str, Enum):
    none = "none"
    # injecting inputs into the hidden layers
    skip = "skip"


class ModelVarType(str, Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    # posterior beta_t
    fixed_small = "fixed_small"
    # beta_t
    fixed_large = "fixed_large"


class ModelMeanType(str, Enum):
    """
    Which type of output the model predicts.
    """

    eps = "eps"


class ModelName(str, Enum):
    """
    DEPRECATED use trainmode in enums.py instead

    List of all supported model classes
    """

    beatgans_ddpm = "beatgans_ddpm"
    autoencoder = "beatgans_autoenc"
    # simclr = "simclr"
    # simsiam = "simsiam"


class ModelType(str, Enum):
    """
    Kinds of the backbone models
    """

    # unconditional ddpm
    ddpm = "ddpm"
    # autoencoding ddpm cannot do unconditional generation
    autoencoder = "autoencoder"
    palette_only = "palette_only"
    RRDBNet = "RRDBNet"
    RRDBNet_diffusion = "RRDBNet_diffusion"
    RCAN = "RCAN"
    Pix2Pix = "Pix2Pix"

    def has_autoenc(self):
        return self in [ModelType.autoencoder, ModelType.palette_only, ModelType.RRDBNet, ModelType.Pix2Pix, ModelType.RCAN]

    def has_encoder(self, a: Model) -> TypeGuard[Diffusion_Autoencoder_Model]:
        return self in [ModelType.autoencoder]

    def can_sample(self):
        return self in [ModelType.ddpm]


class GenerativeType(str, Enum):
    """
    How's a sample generated
    """

    ddpm = "ddpm"
    ddim = "ddim"


class ManipulateLossType(str, Enum):
    bce = "bce"
    mse = "mse"


class LossType(str, Enum):
    mse = "mse"  # use raw MSE loss (and KL when learning variances)
    l1 = "l1"


class OptimizerType(str, Enum):
    adam = "adam"
    adamW = "adamw"
