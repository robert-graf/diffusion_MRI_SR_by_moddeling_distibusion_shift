from enum import Enum, auto


class CMPEnum(Enum):
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Enum):
            return self.name == __value.name
        if isinstance(__value, str):
            return self.name == __value
        return super().__eq__(__value)


class TrainMode(str, Enum):
    # manipulate mode = training the classifier
    # manipulate = "manipulate"
    # default training mode!
    diffusion = "diffusion"
    ESRGANModel = "ESRGANModel"
    RCAN = "RCAN"
    Pix2Pix = "Pix2Pix"
    # default latent training mode!
    # fitting the a DDPM to a given latent
    # latent_diffusion = "latentdiffusion"
    # supervised base line
    # supervised = "supervised"

    # def is_manipulate(self):
    #    return self in [TrainMode.manipulate, TrainMode.supervised]

    def is_diffusion(self):
        return self in [TrainMode.diffusion]  # , TrainMode.latent_diffusion]

    def is_autoenc(self):
        # the network possibly does autoencoding
        return self in [TrainMode.diffusion]

    def has_discriminator(self):
        return self in [TrainMode.ESRGANModel, TrainMode.Pix2Pix]

    # def is_latent_diffusion(self):
    #    return self in [TrainMode.latent_diffusion]

    # def use_latent_net(self):
    #    return self.is_latent_diffusion()

    # def require_dataset_infer(self):
    #    """
    #    whether training in this mode requires the latent variables to be available?
    #    """
    #    # this will precalculate all the latents before hand
    #    # and the dataset will be all the predicted latents
    #    return self in [TrainMode.latent_diffusion, TrainMode.manipulate]


class Discriminator(str, Enum):
    patch = auto()
    channel_norm_patch = auto()
    UNetDiscriminatorSN = auto()


class Inpainting(str, CMPEnum):
    random_ege = auto()
    perlin = auto()


class EmbeddingType(str, CMPEnum):
    adaptivenonzero = auto()
    ConvEmb = auto()


class Models(str, CMPEnum):
    resnet = auto()
    unet = auto()


class SegLoss(str, CMPEnum):
    Dice_CE = auto()
    DiceMSELoss = auto()
    # marginalized_Dice_CE = auto()
    # Leaf_Dice_CE = auto()
    WassDice_CE_Total13 = auto()
    # WassDice_CE_BraTSv2 = auto()
