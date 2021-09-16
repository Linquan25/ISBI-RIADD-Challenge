import torch.nn as nn
from metrics import *

from .base import BaseNet
from .vit_utils import configs as vit_configs
from .vit_utils.modeling import VisionTransformer


class ViT(BaseNet):
    def __init__(self, vit_config='ViT-B_16'):
        super().__init__()
        CONFIGS = {
            'ViT-B_16': vit_configs.get_b16_config(),
            'ViT-B_32': vit_configs.get_b32_config(),
            'ViT-L_16': vit_configs.get_l16_config(),
            'ViT-L_32': vit_configs.get_l32_config(),
            'ViT-H_14': vit_configs.get_h14_config(),
            'R50-ViT-B_16': vit_configs.get_r50_b16_config(),
            'testing': vit_configs.get_testing(),
        }
        

        self.model = VisionTransformer(CONFIGS[vit_config], img_size=224, 
                                       zero_head=True, num_classes=1)
        ckpt = np.load('../ViT-GTA/pretrained/imagenet21k+imagenet2012_ViT-B_16-224.npz')
        self.model.load_from(ckpt)
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        out, *_ = self.model(x)
        return out
