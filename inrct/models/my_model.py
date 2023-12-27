import torch
import torch.nn as nn

from .backbones.resnet import resnet50

class NILUT(nn.Module):
    """
    Simple residual coordinate-based neural network for fitting 3D LUTs
    Official code: https://github.com/mv-lab/nilut
    """
    def __init__(self, configs):
        super().__init__()
        self.in_features = configs['in_features']
        self.hidden_features = configs['hidden_features']
        self.hidden_layers = configs['hidden_layers']
        self.out_features = configs['out_features']
        self.res = configs['res']

        self.net = []
        self.net.append(nn.Linear(self.in_features, self.hidden_features))
        self.net.append(nn.ReLU())
        
        for _ in range(self.hidden_layers):
            self.net.append(nn.Linear(self.hidden_features, self.hidden_features))
            self.net.append(nn.GELU())
        
        self.net.append(nn.Linear(self.hidden_features, self.out_features))
        if not self.res:
            self.net.append(torch.nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, intensity):
        output = self.net(intensity)
        if self.res:
            output = output + intensity
            # output = torch.clamp(output, 0.,1.)
        
        return output, intensity


