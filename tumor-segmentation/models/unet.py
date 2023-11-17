import torch
import torch.nn as nn

import numpy as np

from augmentor import ImagePreprocessor, postprocessor, to_grayscale, preprocessor

#U-Net model definition
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_downsamples = 4, device = 'cpu'):
        super(UNet, self).__init__()

        self.DEVICE = device

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up3 = DoubleConv(512, 256)
        self.up2 = DoubleConv(256, 128)
        self.up1 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, out_channels, 1)

        self.to(self.DEVICE)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        x4 = self.dropout(x4)
        x = self.upconv3(x4)
        x = self.up3(torch.cat([x, x3], dim=1))

        x = self.upconv2(x)
        x = self.up2(torch.cat([x, x2], dim=1))

        x = self.upconv1(x)
        x = self.up1(torch.cat([x, x1], dim=1))

        logits = self.outc(x)
        return logits
    
    def predict(self, o_img):

        img = preprocessor(o_img)
        img = to_grayscale(img)
        img = torch.tensor(img, dtype=torch.float32, device = self.DEVICE)

        with torch.no_grad():
            logits = self.forward(img[None,None,:])
            logits = torch.sigmoid(logits[0][0])
            logits = logits > 0.5
            logits = logits*255
            logits = logits.cpu().numpy()
            logits = np.stack([logits,logits,logits], axis = -1)
            logits = postprocessor(logits, o_img.shape[:2])
        
        return logits
