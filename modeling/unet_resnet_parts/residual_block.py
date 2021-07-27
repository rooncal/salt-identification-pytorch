import torch.nn as nn
from .conv_block import convBlock

class residualBlock(nn.Module):
    def __init__(self, in_channels, filters, size = 3, use_bn=True):
        super(residualBlock, self).__init__()

        self.norm = nn.BatchNorm2d(in_channels)
        self.conv1 = convBlock(in_channels, filters, size, use_bn=use_bn)
        self.conv2 = convBlock(filters, filters, size, activation=False, use_bn=use_bn)
        self.relu = nn.ReLU()
        self.use_bn = use_bn

    def forward(self, x):
        residual = x  
        x = self.relu(x)
        if self.use_bn:
          x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x 
    