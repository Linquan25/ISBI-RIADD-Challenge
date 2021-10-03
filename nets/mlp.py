import torch
import torch.nn as nn
from loss import soft_aar_loss
from metrics import *

from .base import BaseNet


class MLP(BaseNet):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(2048, 2024),
            nn.LeakyReLU(.2),
            nn.Dropout(.2),
            nn.Linear(2024, 2024),
            nn.LeakyReLU(.2),
            nn.Dropout(.2),
            nn.Linear(2024, 1)
        )
        self.loss_func = soft_aar_loss
