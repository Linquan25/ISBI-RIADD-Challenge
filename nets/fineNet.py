
import torch
import torch.nn as nn
from loss import soft_aar_loss
from metrics import *
from torchvision import models
from loss import weighted_BCEWithLogitsLoss
from .baseFine import BaseNet
from .SELayer import SELayer
from collections import OrderedDict

class FineNet(BaseNet):
    def __init__(self, pretrained=True):
        super().__init__()
        # self.model = torch.hub.load(
        #     "pytorch/vision:v0.9.0", "resnext50_32x4d", pretrained=pretrained
        # )
        #self.model = models.resnext101_32x8d(pretrained=True)
        
        self.model = models.resnext101_32x8d(pretrained=False)
        ckpt = torch.load('saved_model/ISBI-WeightedBCE-ResNext101-epoch=013-val_loss=0.0892.ckpt', map_location=torch.device('cpu'))
        new_dict = {k.replace('vit.', 'model.'): v for k, v in ckpt['state_dict'].items()}
        new_state_dict = OrderedDict()
        for k, v in new_dict.items():
            name = k[6:] # remove `model.`
            new_state_dict[name] = v
        self.model.fc = nn.Linear(2048, 29)
        self.model.load_state_dict(new_state_dict)
        self.model.avgpool = SELayer(2048,1.5)
        self.model.fc = nn.Linear(2048, 19)
        #### freeze parameters for trained model
        for param in self.model.parameters():
            param.requires_grad = False 
        #### free the parameters of only the last to layers (SE block)
        for param in self.model.avgpool.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True
        self.loss_func = nn.BCEWithLogitsLoss()
        #self.loss_func = ISBI_Challenge_Loss
