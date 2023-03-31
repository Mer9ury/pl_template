import torch
import torch.nn as nn

from .backbones.resnet import resnet50

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet50(pretrained=True)
        self.classifier = nn.Linear(2048, 1)
    
    def forward(self, x):
        feature = self.backbone.get_feature(x)
        y = self.classifier(feature)

        return y


