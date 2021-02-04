import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBn, ConvPReLU, ConvLReLU, ConvTanh, ConvBnLReLU, ConvPixPReLU
from .blocks import ResidualBlock
    
class Generator(nn.Module):
    def __init__(self,
                 in_channels, hid_channels, out_channels):
        super().__init__()
        # Input Block
        self.InputBlock=ConvPReLU(in_channels, hid_channels,
                                  kernel_size=9, padding=4)
        
        # Residual Stage with 16 Residual Blocks
        self.ResStage=nn.Sequential(
            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),

            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),

            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),

            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),
            ResidualBlock(hid_channels),
            
            ConvBn(hid_channels, hid_channels))
        
        # PixelShuffle Stage with 2 Pixel Shuffle Blocks 
        self.PixStage=nn.Sequential(
            ConvPixPReLU(hid_channels, 4*hid_channels),
            ConvPixPReLU(hid_channels, 4*hid_channels))
        
        # Output Block
        self.OutBlock=ConvTanh(hid_channels, out_channels,
                                kernel_size=9, padding=4)
        
    def forward(self, x):
        x = self.InputBlock(x)
        fx = self.ResStage(x)
        x = self.PixStage(fx + x)
        x = self.OutBlock(x)
        return x
    
    
class Discriminator(nn.Module):
    def __init__(self,
                 in_channels, hid_channels, out_channels):
        super().__init__()
        self.InputBlock=nn.Sequential(
            ConvLReLU(in_channels, hid_channels,
                            kernel_size=3, stride=1, padding=1),
            ConvBnLReLU(hid_channels, hid_channels,
                            kernel_size=3, stride=2, padding=1))
        
        self.BaseStage=nn.Sequential(
            ConvBnLReLU(1* hid_channels, 2* hid_channels,
                            kernel_size=3, stride=1, padding=1),
            ConvBnLReLU(2* hid_channels, 2* hid_channels,
                            kernel_size=3, stride=2, padding=1),
                
            ConvBnLReLU(2* hid_channels, 4* hid_channels,
                            kernel_size=3, stride=1, padding=1),
            ConvBnLReLU(4* hid_channels, 4* hid_channels,
                            kernel_size=3, stride=2, padding=1),
        
            ConvBnLReLU(4* hid_channels, 8* hid_channels,
                            kernel_size=3, stride=1, padding=1),
            ConvBnLReLU(8* hid_channels, 8* hid_channels,
                            kernel_size=3, stride=2, padding=1))
        
        self.OutBlock=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvLReLU(8* hid_channels, 16* hid_channels,
                            kernel_size=1, stride=1, padding=0),
            nn.Conv2d(16* hid_channels, out_channels,
                            kernel_size=1, padding=0),
            nn.Flatten())
        
    def forward(self, x):
        x = self.InputBlock(x)
        x = self.BaseStage(x)
        x = self.OutBlock(x)
        return x
