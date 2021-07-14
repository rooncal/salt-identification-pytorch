class deconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 2, stride = 2):
        super(deconvBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride)

    def forward(self, x1, x2):
        xd = self.deconv(x1)
        x = torch.cat([xd, x2], dim = 1)
        return x