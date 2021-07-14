class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, use_bn=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            DoubleConv(in_channels, out_channels, use_bn=use_bn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
