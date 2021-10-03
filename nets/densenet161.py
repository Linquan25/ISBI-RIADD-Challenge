
import torch
import torch.nn as nn
from metrics import *
from torchvision import models
from .baseISBI import BaseNet


class Densenet161(BaseNet):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = models.densenet161(pretrained=True)
        self.model.classifier = nn.Linear(2208, 29)
        self.loss_func = nn.BCEWithLogitsLoss()
        #self.loss_func = nn.MSELoss()
        #self.loss_func = soft_aar_loss
