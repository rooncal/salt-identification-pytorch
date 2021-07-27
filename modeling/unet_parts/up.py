import torch
import torch.nn as nn
from .double_conv import DoubleConv

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, use_bn=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(dropout)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_bn=use_bn)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.dropout(x)
        return self.conv(x)
