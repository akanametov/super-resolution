import torch
import torchvision
from PIL import Image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, ToPILImage

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