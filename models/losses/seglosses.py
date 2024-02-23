import numpy as np
import torch
from monai.losses import DiceLoss
from monai.networks import one_hot
from torch import nn

from models.losses.GeneralizedWassersteinDiceLoss import GeneralizedWassersteinDiceLoss
from utils.enums import SegLoss


def get_seg_loss(loss: SegLoss, num_classes: int):
    if loss == SegLoss.Dice_CE:
        return DiceCELoss()
    if loss == SegLoss.DiceMSELoss:
        return DiceMSELoss(num_classes)
    if loss == SegLoss.WassDice_CE_Total13:
        m = np.array(
            [  # 0    1     2    3    4    5    6    7    8    9    10  11    12
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 0
                [1.0, 0.0, 0.8, 0.8, 0.6, 0.5, 0.8, 0.8, 0.4, 0.5, 0.6, 0.1, 0.2],  # SKIN 1
                [1.0, 0.8, 0.0, 0.1, 0.4, 1.0, 0.1, 0.2, 0.4, 1.0, 0.1, 1.0, 1.0],  # Organs 2
                [1.0, 0.8, 0.1, 0.0, 0.4, 1.0, 0.1, 0.2, 0.4, 1.0, 0.1, 1.0, 1.0],  # Vessels 3
                [1.0, 0.6, 0.4, 0.4, 0.0, 1.0, 0.2, 0.2, 0.8, 1.0, 1.0, 0.2, 1.0],  # Lung 4
                [1.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.1, 0.1, 1.0, 0.5, 0.8, 0.5, 1.0],  # Brain 5
                [1.0, 0.8, 0.0, 0.1, 0.4, 1.0, 0.0, 0.3, 0.4, 1.0, 0.1, 1.0, 1.0],  # Guts 6
                [1.0, 0.8, 0.2, 0.2, 0.2, 0.1, 0.3, 0.0, 0.3, 0.5, 0.9, 0.4, 0.8],  # Bone 7
                [1.0, 0.4, 0.4, 0.4, 0.8, 1.0, 0.3, 0.3, 0.0, 1.0, 0.5, 0.5, 1.0],  # Muskel 8
                [1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0],  # Rachen 9 (RM)
                [1.0, 0.6, 0.1, 0.1, 1.0, 0.8, 1.0, 0.9, 0.5, 1.0, 0.0, 0.2, 1.0],  # Torso Fat
                [1.0, 0.1, 1.0, 1.0, 0.2, 0.5, 0.2, 0.4, 0.5, 0.5, 0.2, 0.0, 1.0],  # Innter SKIN 11
                [1.0, 0.2, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 0.0],  # Cartelige 12 (RM)
            ],
            dtype=np.float64,
        )
        return GeneralizedWassersteinDiceLoss(np.array(m))
    raise NotImplementedError(loss)


class DiceCELoss(nn.Module):
    """Dice and Cross-entropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy


class DiceMSELoss(nn.Module):
    """Dice and Cross-entropy loss"""

    def __init__(self, num_classe):
        super().__init__()
        self.dice = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.MSELoss()
        self.num_classes = num_classe

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        cross_entropy = self.cross_entropy(y_pred, one_hot(y_pred, self.num_classes))
        return dice + cross_entropy
