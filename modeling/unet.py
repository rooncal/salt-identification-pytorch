from .unet_parts.double_conv import DoubleConv
from .unet_parts.down import Down
from .unet_parts.out_conv import OutConv
from .unet_parts.up import Up

class GenericUnet(nn.Module):
    def __init__(self, n_channels, n_classes, dropout=0.2, use_bn=True):
        super(GenericUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64, use_bn=use_bn)
        self.down1 = Down(64, 128, dropout, use_bn=use_bn)
        self.down2 = Down(128, 256, dropout, use_bn=use_bn)
        self.down3 = Down(256, 512, dropout, use_bn=use_bn)
        self.down4 = Down(512, 512, dropout, use_bn=use_bn)
        self.up1 = Up(1024, 256, dropout, use_bn=use_bn)
        self.up2 = Up(512, 128, dropout, use_bn=use_bn)
        self.up3 = Up(256, 64, dropout, use_bn=use_bn)
        self.up4 = Up(128, 64, dropout, use_bn=use_bn)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits