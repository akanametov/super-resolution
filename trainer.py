import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from tqdm.notebook import tqdm
from IPython.display import clear_output

######################################
####### Function to show images  #####
######################################

def showImage(img_data, title):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    img_data = ((img_data + 1.)/2).detach().cpu()
    img_grid = make_grid(img_data[:4], nrow=4)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
######################################
###########    Predictor  ############
######################################
    
class Predictor():
    def __init__(self, model, transform):
        self.model=model.eval()
        self.transform=transform
        
    def predict(self, dataloader, path='test'):
        names = dataloader.dataset.images
        for i, img in enumerate(tqdm(dataloader)):
            hd_img_data = self.model(img)
            hd_img = self.transform(hd_img_data)
            hd_img.save(path+'/hd_'+names[i])
    
######################################
########   Generator`s WarmUp  #######
######################################

class GeneratorWarmUp():
    def __init__(self, generator, criterion, optimizer, device='cuda:0'):
        self.generator=generator.to(device)
        self.criterion=criterion.to(device)
        self.optimizer=optimizer
        
    def start(self, dataloader, epochs=500, device='cuda:0', name='SRResNet'):
        self.losses=[]
        for epoch in range(epochs):
            loss = 0.
            for x, real in tqdm(dataloader):
                x = x.to(device)
                real = real.to(device)
                
                with torch.cuda.amp.autocast(enabled=(device=='cuda:0')):
                    fake = self.generator(x)
                    loss = self.criterion(fake, real)
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                loss += loss.item()
            
            self.losses.append(loss/len(dataloader))
            clear_output(wait=True)
            print('Epoch {}: Generator`s ({}) loss: {:.5f}'.format(epoch+1, name, self.losses[-1]))
            showImage(2*x - 1., 'Input Low-Resolution Image')
            showImage(fake.to(real.dtype), 'Generated High-Resolution Image')
            showImage(real, 'Real High-Resolution Image')
            
            
######################################
###### Super-Resolution Trainer  #####
######################################
            
class Trainer():
    def __init__(self,
                 generator, discriminator,
                 g_criterion, d_criterion,
                 g_optimizer, d_optimizer, device='cuda:0'):
        
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.g_criterion = g_criterion.to(device)
        self.d_criterion = d_criterion.to(device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        
    def fit(self, dataloader, epochs=500, device='cuda:0'):
        g_losses=[]
        d_losses=[]
        for epoch in range(epochs):
            ge_loss=0.
            de_loss=0.
            for x, real in tqdm(dataloader):
                x = x.to(device)
                real = real.to(device)
                
                with torch.cuda.amp.autocast(enabled=(device=='cuda:0')):
                    # Generator`s loss
                    fake = self.generator(x)
                    fake_pred = self.discriminator(fake)
                    g_loss = self.g_criterion(fake_pred, fake, real)
                    
                    # Discriminator`s loss
                    fake = self.generator(x).detach()
                    fake_pred = self.discriminator(fake)
                    real_pred = self.discriminator(real)
                    d_loss = self.d_criterion(fake_pred, real_pred)
                    
                # Generator`s params update
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                
                # Discriminator`s params update
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
                ge_loss += g_loss.item()
                de_loss += d_loss.item()
                
            g_losses.append(ge_loss/len(dataloader))
            d_losses.append(de_loss/len(dataloader))
            
            clear_output(wait=True)
            print(f':::::::::::::::::  Epoch {epoch+1}  :::::::::::::::::')
            print(f'::::::::::: Generator loss: {g_losses[-1]:.3f} :::::::::::')
            print(f'::::::::: Discriminator loss: {d_losses[-1]:.3f} :::::::::')
            showImage(2*x - 1., 'Input Low-Resolution Image')
            showImage(fake.to(real.dtype), 'Generated High-Resolution Image')
            showImage(real, 'Real High-Resolution Image')
        self.data={'g_loss': g_losses, 'd_loss': d_losses}
