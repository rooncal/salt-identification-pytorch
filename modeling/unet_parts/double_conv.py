class Nothing(nn.Module):
  def __init__(self,*args):
    super(Nothing, self).__init__()

  def forward(self, x):
    return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_bn=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_bn:
          BatchNorm = nn.BatchNorm2d
        else:
          BatchNorm = Nothing
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            BatchNorm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
