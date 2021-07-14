class convBlock(nn.Module):
    def __init__(self, in_channels, filters, size, stride = 1, activation = True, use_bn=True):
        super(convBlock, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, filters, size, stride = stride, padding = size//2)
        self.norm = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
          x = self.norm(x)
        if self.activation:
            return self.relu(x)
        return x