
import torch
import torch.nn as nn
from metrics import *
from torchvision import models
from .baseISBI import BaseNet


class ResNet152(BaseNet):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Linear(2048, 29)
        self.loss_func = nn.BCEWithLogitsLoss()
        #self.loss_func = nn.MSELoss()
        #self.loss_func = soft_aar_loss
