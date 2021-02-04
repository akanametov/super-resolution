import torch
import torch.nn as nn
import torch.nn.functional as F

##################################
######### ConvBn block ###########
##################################
    
class ConvBn(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_bn=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        return self.conv_bn(x)
    
##################################
####### ConvLeakyReLU block ######
##################################

class ConvLReLU(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, alpha=0.2):
        super().__init__()
        self.conv_lrelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(alpha, inplace=True))
        
    def forward(self, x):
        return self.conv_lrelu(x)
    
##################################
######### ConvPReLU block ########
##################################

class ConvPReLU(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_prelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.PReLU())
        
    def forward(self, x):
        return self.conv_prelu(x)
    
##################################
######### ConvTanh block #########
##################################

class ConvTanh(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_th=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.Tanh())
        
    def forward(self, x):
        return self.conv_th(x)

##################################
####### ConvBnPReLU block ########
##################################

class ConvBnPReLU(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_bn_prelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.PReLU())
        
    def forward(self, x):
        return self.conv_bn_prelu(x)
    
##################################
##### ConvBnLeakyReLU block ######
##################################

class ConvBnLReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=4, stride=2, padding=1, alpha=0.2):
        super().__init__()
        self.conv_bn_lrelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha, inplace=True))
        
    def forward(self, x):
        return self.conv_bn_lrelu(x)
    
##################################
####### ConvPixPReLU block #######
##################################

class ConvPixPReLU(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_pix_prelu=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.PixelShuffle(2),
            nn.PReLU())
        
    def forward(self, x):
        return self.conv_pix_prelu(x)
    
##################################
######### Residual Block #########
##################################
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnPReLU(in_channels, in_channels, kernel_size, stride, padding),
            ConvBn(in_channels, in_channels, kernel_size, stride, padding))

    def forward(self, x):
        fx = self.block(x)
        return fx + x