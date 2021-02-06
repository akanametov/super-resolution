import os
import torch
import torchvision
from PIL import Image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, ToPILImage

def NormalizeToImage(x):
    x_ = x[0].cpu()
    return ToPILImage()((x_ + 1)/2)

######################################
###### Customized STL10 Dataset ######
######################################

class DatasetSTL(torchvision.datasets.STL10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        LD_SIZE = [24, 24]
        HD_SIZE = [96, 96]
        self.transformInit = ToPILImage()
        self.transformLD = Compose([
                        Resize(LD_SIZE, interpolation=Image.BICUBIC),
                        ToTensor()
                        ])
        self.transformHD = Compose([
                        Resize(HD_SIZE),
                        ToTensor(),
                        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        
    def __getitem__(self, idx):
        img_data = torch.from_numpy(self.data[idx])
        img = self.transformInit(img_data)
        x = self.transformLD(img)
        y = self.transformHD(img)
        return x, y
    
######################################
#### Customized ImageNet Dataset #####
######################################
    
class DatasetImageNet(torchvision.datasets.ImageNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        LD_SIZE = [24, 24]
        HD_SIZE = [96, 96]
        self.transformInit = ToPILImage()
        self.transformLD = Compose([
                        Resize(LD_SIZE, interpolation=Image.BICUBIC),
                        ToTensor()
                        ])
        self.transformHD = Compose([
                        Resize(HD_SIZE),
                        ToTensor(),
                        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
        
    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        x = self.transformLD(img)
        y = self.transformHD(img)
        return x, y
    
######################################
###### DataSet for test images  ######
######################################

class DataSet(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 input_size=(24, 24),
                 transform=ToTensor()):
        super().__init__()
        self.path=path+'/'
        self.input_size=input_size
        self.images=self.__getImages__(path)
        self.transform=transform
    
    @staticmethod
    def __getImages__(path):
        files = os.listdir(path)
        isImage = lambda f: (f.endswith('.png')
                            or f.endswith('.jpg')
                            or f.endswith('.jpeg'))
        images = [f for f in files if isImage(f)]
        return sorted(images)
    
    def __len__(self,):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.path + self.images[idx]
        img = Image.open(img_path).convert('RGB').resize(self.input_size)
        return self.transform(img)