
import torch
import torch.nn as nn
from loss import soft_aar_loss
from metrics import *
from torchvision import models
from loss import weighted_BCEWithLogitsLoss
from .baseFine import BaseNet
from collections import OrderedDict

class ensembleNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                        nn.Linear(49,1024),
                        nn.ReLU(),
                        nn.Linear(1024,2048),
                        nn.ReLU(),
                        nn.Linear(2048,2048),
                        nn.ReLU(),
                        nn.Linear(2048,29)
                    )
        self.loss_func = nn.BCEWithLogitsLoss()
        #self.loss_func = ISBI_Challenge_Loss
