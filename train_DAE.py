from pprint import pprint
from typing import Literal

import torch

from pl_models.DEA import DAE_LitModel
from train_default import train
from utils import arguments


def train_(opt: arguments.DAE_Option, mode: Literal["train", "eval"] = "train"):
    pprint(opt.__dict__)
    model: DAE_LitModel = DAE_LitModel(opt)
    train(model, opt, mode)


def get_opt(config=None) -> arguments.DAE_Option:
    torch.cuda.empty_cache()
    opt = arguments.DAE_Option().get_opt(None, config)
    opt = arguments.DAE_Option.from_kwargs(**opt.parse_args().__dict__)

    opt.val_niis_pairs = []
    # if opt.ablation_lvl != -1:
    #    from dataloader.transforms import ablation
    #
    #    opt.transforms_3D = ablation(opt.ablation_lvl)
    #    opt.experiment_name = f"DAE_{opt.ablation_lvl}_" + opt.experiment_name
    # else:
    #    opt.experiment_name = "DAE_" + opt.experiment_name
    return opt


if __name__ == "__main__":
    train_(get_opt())
