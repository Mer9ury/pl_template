import torch
import torch.nn as nn

from .backbones.resnet import *

class MyModel_default(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = resnet18(pretrained=False,num_classes=1)

    def forward(self, x):
        return self.backbone(x)

    

