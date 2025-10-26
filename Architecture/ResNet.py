from torchvision.models import resnet18, ResNet18_Weights
from torch import cuda, device, nn, save, load
from torch.utils.data import DataLoader
from torch.optim import SGD
from pathlib import Path

dev = device("cuda" if cuda.is_available() else "cpu")


'''
Fast implementation of Resnet18, adapted from HW1 to CPGD
'''
class ResNet18(nn.Module):
    def __init__(self, data_type: str, num_classes: int):
        super(ResNet18, self).__init__()
        w = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=w)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.model.eval()
        self.state_path = "Models/ResNet_" + data_type.upper() + ".pth"
        self.lossfunc = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)
    
    def set_pretrain_path(self, path: str):
        self.state_path = path
    
    def forward(self, x):
        return self.model(x)
    
    def training_loop(self, data):
        for e in range(10):
            running_loss = 0.0
            for images, labels in data:
                self.optimizer.zero_grad()
                images, labels = images.to(dev), labels.to(dev)
                output = self.model(images)
                l = self.lossfunc(output, labels)
                l.backward()
                self.optimizer.step()
                running_loss += l.item()
            print(f"Epoch:{e} loss: {running_loss}")

    def getLoss(self):
        return self.lossfunc
    
    def isPretrainOnDisk(self):
        file_path = Path(self.state_path)
        return file_path.is_file()

    def save(self):
        save(self.model.state_dict(), self.state_path)

    def load(self):
        state_dict = load(self.state_path, map_location=dev)
        self.model.load_state_dict(state_dict)
        self.model.eval()