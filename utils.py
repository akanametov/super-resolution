import torch
import torch.nn as nn
from torchvision.models import vgg16, vgg19
    
######################################
######## Adverserial BCE loss ########
######################################

class AdvLoss(nn.Module):
    '''BCE for True and False reals'''
    def __init__(self, alpha=1):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.alpha=alpha

    def forward(self, pred, target):
        return self.alpha* self.loss_fn(pred, target)
    

######################################
######### Pixel-wise MSE loss ########
######################################
    
class PixLoss(nn.Module):
    '''Pixel-wise MSE loss for images''' 
    def __init__(self, alpha=20):
        super().__init__()
        self.alpha=alpha

    def forward(self, fake, real):
        return self.alpha* torch.mean((fake - real)**2)
    
######################################
########   Model-based loss   ########
######################################

class ModelBasedLoss(nn.Module):
    '''Model based loss for generator'''
    def __init__(self, alpha=2, name='vgg19', device='cuda:0'):
        super().__init__()
        model = self.__loadModel__(name)
        self.model = self.__freeze__(model).to(device)
        self.alpha=alpha
        
    @staticmethod
    def __loadModel__(name='vgg19'):
        if name=='vgg16':
            model = vgg16(pretrained=True).features[:-1]
        elif name=='vgg19':
            model = vgg19(pretrained=True).features[:-1]
        return model.eval() 
    
    @staticmethod
    def __freeze__(model):
        for p in model.parameters():
            p.requires_grad = False
        return model
        
    def forward(self, fake, real):
        pred = self.model(fake)
        target = self.model(real)
        return self.alpha* torch.mean((pred - target)**2)
    
######################################
########   Generator loss   ##########
######################################

class GeneratorLoss(nn.Module):
    '''Generator loss'''
    def __init__(self, alpha=0.001, beta=0.006,
                 gamma=1, model='vgg19', device='cuda:0'):
        super().__init__()
        self.bce = AdvLoss(alpha)
        self.fb_mse = ModelBasedLoss(beta, model, device)
        self.mse = PixLoss(gamma)
        
    def forward(self, fake_pred, fake, real):
        fake_target = torch.ones_like(fake_pred)
        
        loss = (self.bce(fake_pred, fake_target)\
              + self.fb_mse(fake, real)\
              + self.mse(fake, real))#/3
        return loss

######################################
#######  Discriminator loss   ########
######################################

class DiscriminatorLoss(nn.Module):
    '''Discriminator loss'''
    def __init__(self, alpha=1):
        super().__init__()
        self.bce = AdvLoss(alpha)
        
    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        
        loss = (self.bce(fake_pred, fake_target)\
              + self.bce(real_pred, real_target))/2
        return loss
