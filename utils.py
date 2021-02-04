import torch
import torch.nn as nn

######################################
######### Pixel-wise MSE loss ########
######################################
    
class PixLoss(nn.Module):
    '''Pixel-wise MSE loss for images''' 
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha=alpha

    def forward(self, fake, real):
        return self.alpha* torch.mean((fake - real)**2)
    
######################################
######## Adverserial BCE loss ########
######################################

class AdvLoss(nn.Module):
    '''BCE for True and False reals'''
    def __init__(self, beta=1):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.beta=beta

    def forward(self, pred, target):
        return self.beta* self.loss_fn(pred, target)
    
######################################
########   Model-based loss   ########
######################################

class ModelBasedLoss(nn.Module):
    '''Model based loss function for generator'''
    def __init__(self, model, gamma=1, device='cuda:0'):
        super().__init__()
        self.model = model.to(device)
        self.gamma=gamma

    def forward(self, fake, real):
        pred = self.model(fake)
        target = self.model(real)
        return self.gamma* torch.mean((pred - target)**2)