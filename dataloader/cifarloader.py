from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch import randperm

class CifarDataLoader:
    def __init__(self):
        self.cifar10data = None
        self.cifar10datatest = None
        self.rgbT = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224),
                # cifar10 normalization values (known)
                transforms.Normalize((.5,.5,.5), (.5,.5,.5))
            ])
        
    def get_test_data(self):
        if self.cifar10data is None:
            self.cifar10data = CIFAR10(root='./data', train=False, download=True, transform=self.rgbT)
        return self.cifar10data
    
    def get_test_data_iterable(self):
        cifar10data = self.get_test_data()
        return DataLoader(cifar10data, batch_size=64, shuffle=True)
    
    def get_random_test_slice(self, size=64):
        cifar10data = self.get_test_data()
        indices = randperm(len(cifar10data))[:size]
        dataSubSet = Subset(cifar10data, indices)
        return DataLoader(dataSubSet, batch_size=size, shuffle=True)
    
    def get_train_data(self):
        if self.cifar10datatest is None:
            self.cifar10datatest = CIFAR10(root='./data', train=False, download=True, transform=self.rgbT)
        return self.cifar10datatest
    
    def get_train_data_iterable(self):
        cifar10data = self.get_train_data()
        return DataLoader(cifar10data, batch_size=64, shuffle=True)