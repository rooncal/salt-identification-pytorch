from unet_resnet_parts import *

class UnetResnetModel(nn.Module):

    def __init__(self, filters = 16, dropout = 0.2, use_bn = True):
        super(UnetResnetModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, filters, 3, padding = 1),
            residualBlock(filters, filters, use_bn=use_bn),
            residualBlock(filters, filters, use_bn=use_bn),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout/2),
            nn.Conv2d(filters, filters * 2, 3, padding = 1),
            residualBlock(filters * 2, filters * 2, use_bn=use_bn),
            residualBlock(filters * 2, filters * 2, use_bn=use_bn),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            nn.Conv2d(filters * 2, filters * 4, 3, padding = 1),
            residualBlock(filters * 4, filters * 4, use_bn=use_bn),
            residualBlock(filters * 4, filters * 4, use_bn=use_bn),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            nn.Conv2d(filters * 4, filters * 8, 3, padding = 1),
            residualBlock(filters * 8, filters * 8, use_bn=use_bn),
            residualBlock(filters * 8, filters * 8, use_bn=use_bn),
            nn.ReLU()
        )
        
        self.middle = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout),
            nn.Conv2d(filters * 8, filters * 16, 3, padding = 3//2),
            residualBlock(filters * 16, filters * 16, use_bn=use_bn),
            residualBlock(filters * 16, filters * 16, use_bn=use_bn),
            nn.ReLU()
        )
        
        self.deconv4 = deconvBlock(filters * 16, filters * 8, 2)
        self.upconv4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 16, filters * 8, 3, padding = 1),
            residualBlock(filters * 8, filters * 8, use_bn=use_bn),
            residualBlock(filters * 8, filters * 8, use_bn=use_bn),
            nn.ReLU()
        )
  

        self.deconv3 = deconvBlock(filters * 8, filters * 4, 3)
        self.upconv3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 8, filters * 4, 3, padding = 1),
            residualBlock(filters * 4, filters * 4, use_bn=use_bn),
            residualBlock(filters * 4, filters * 4, use_bn=use_bn),
            nn.ReLU()
        )
        
        self.deconv2 = deconvBlock(filters * 4, filters * 2, 2)
        self.upconv2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 4, filters * 2, 3, padding = 1),
            residualBlock(filters * 2, filters * 2, use_bn=use_bn),
            residualBlock(filters * 2, filters * 2, use_bn=use_bn),
            nn.ReLU()
        )

        self.deconv1 = deconvBlock(filters * 2, filters, 3)
        self.upconv1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(filters * 2, filters, 3, padding = 1),
            residualBlock(filters, filters, use_bn=use_bn),
            residualBlock(filters, filters, use_bn=use_bn),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Conv2d(filters, 1, 3, padding = 1)
        )

    def forward(self, x):
        conv1 = self.conv1(x) 
        
        conv2 = self.conv2(conv1) 
        
        conv3 = self.conv3(conv2) 
        
        conv4 = self.conv4(conv3) 
        
        x = self.middle(conv4) 
        
       
        x = self.deconv4(x, conv4)
        x = self.upconv4(x)
      
        x = self.deconv3(x, conv3)
        x = self.upconv3(x)
     
        x = self.deconv2(x, conv2)
        x = self.upconv2(x)
    
        x = self.deconv1(x, conv1)
        x = self.upconv1(x)

        return x