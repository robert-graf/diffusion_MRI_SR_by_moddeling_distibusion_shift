from typing import Union

from models.srmodel.RCAN import RCAN
from models.srmodel.RRDBNet import RRDBNet
from models.srmodel.RRDBNet_diffusion import RRDBNet_Diffusion
from models.unet import BeatGANsUNetModel
from models.unet_with_encoder import Diffusion_Autoencoder_Model

Model = Union[Diffusion_Autoencoder_Model, BeatGANsUNetModel, RRDBNet, RRDBNet_Diffusion, RCAN]
