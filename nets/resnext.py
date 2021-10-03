
import torch
import torch.nn as nn
from loss import soft_aar_loss
from metrics import *
from torchvision import models
from loss import weighted_BCEWithLogitsLoss
from .baseISBI import BaseNet


class ResNext(BaseNet):
    def __init__(self, pretrained=True):
        super().__init__()
        # self.model = torch.hub.load(
        #     "pytorch/vision:v0.9.0", "resnext50_32x4d", pretrained=pretrained
        # )
        #self.model = models.resnext50_32x4d(pretrained=True)
        self.model = models.resnext101_32x8d(pretrained=True)
        self.model.fc = nn.Linear(2048, 29)
        #self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = weighted_BCEWithLogitsLoss
        #self.loss_func = ISBI_Challenge_Loss
