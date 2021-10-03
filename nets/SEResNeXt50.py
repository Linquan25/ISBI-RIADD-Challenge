
import torch
import torch.nn as nn
from loss import soft_aar_loss
from metrics import *
import pretrainedmodels
from loss import weighted_BCEWithLogitsLoss
from .baseISBI import BaseNet


class SE_ResNeXt(BaseNet):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        self.model.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.model.last_linear = nn.Linear(2048, 29)
        #self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = weighted_BCEWithLogitsLoss
        #self.loss_func = ISBI_Challenge_Loss