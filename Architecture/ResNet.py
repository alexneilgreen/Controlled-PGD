import os
import csv
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torch import cuda, device, nn, save, load
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
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
        self.state_path = "Output/Models/ResNet_" + data_type.upper() + ".pth"
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


'''
ResNet implementation from scratch (Alternative implementation)
'''
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetAlt(nn.Module):
    """
    ResNet18 implementation from scratch with training methods.
    """
    def __init__(self, data_type: str, num_classes: int, in_channels: int = 3):
        super(ResNetAlt, self).__init__()
        self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)
        self.state_path = "Output/Models/ResNet_" + data_type.upper() + ".pth"
        self.lossfunc = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.data_type = data_type
    
    def forward(self, x):
        return self.model(x)
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def training_loop(self, train_loader, epochs=10, lr=0.001, test_loader=None, batch_size=128, weight_decay=1e-4):
        """Train the model with mixed precision and early stopping."""
        device = next(self.model.parameters()).device
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler('cuda')
        
        best_acc = 0.0
        patience = 10
        patience_counter = 0
        early_stop_triggered = False
        epochs_completed = 0
        
        print(f"\nTraining ResNetAlt for {epochs} epochs with lr={lr}...")
        print(f"Using mixed precision training (FP16)...")
        
        final_train_acc = 0.0
        final_test_acc = 0.0
        
        for epoch in range(epochs):
            epochs_completed = epoch + 1
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with torch.amp.autocast('cuda'):
                    output = self.model(data)
                    loss = self.lossfunc(output, target)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | '
                        f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
            
            train_acc = 100. * correct / total
            final_train_acc = train_acc
            scheduler.step()
            
            # Validation phase
            if test_loader is not None:
                self.model.eval()
                test_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        
                        with torch.amp.autocast('cuda'):
                            output = self.model(data)
                            loss = self.lossfunc(output, target)
                        
                        test_loss += loss.item()
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                
                test_acc = 100. * correct / total
                final_test_acc = test_acc
                print(f'\nEpoch {epoch+1}/{epochs} Summary:')
                print(f'Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%\n')
                
                # Save best model and check early stopping
                if test_acc > best_acc:
                    best_acc = test_acc
                    patience_counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': test_acc,
                    }, self.state_path)
                    print(f'Model saved with accuracy: {test_acc:.2f}%')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'\nEarly stopping triggered after {epoch+1} epochs')
                        print(f'No improvement for {patience} consecutive epochs')
                        early_stop_triggered = True
                        break
            else:
                # No test loader, just save after each epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, self.state_path)
        
        if test_loader is not None:
            print(f'\nTraining completed! Best accuracy: {best_acc:.2f}%')
        else:
            print(f'\nTraining completed!')
        
        # Save metrics to CSV
        self._save_metrics_to_csv(
            model_arch="ResNet18",
            dataset=self.data_type,
            epochs=epochs,
            epochs_completed=epochs_completed,
            lr=lr,
            batch_size=batch_size,
            train_acc=final_train_acc,
            test_acc=final_test_acc,
            early_stop=early_stop_triggered
        )
    
    def getLoss(self):
        return self.lossfunc
    
    def isPretrainOnDisk(self):
        file_path = Path(self.state_path)
        return file_path.is_file()
    
    def save(self):
        """Save model state."""
        torch.save(self.model.state_dict(), self.state_path)
    
    def load(self):
        """Load model state."""
        checkpoint = torch.load(self.state_path, map_location=dev)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def _save_metrics_to_csv(self, model_arch, dataset, epochs, epochs_completed, lr, batch_size, train_acc, test_acc, early_stop):
        """Save training metrics to CSV file."""
        csv_path = "Output/Models/Training_Metrics.csv"
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow([
                    'Model Architecture', 'Dataset', 'Max Epochs', 'Epochs Completed', 
                    'Learning Rate', 'Batch Size', 'Training Accuracy', 'Test Accuracy', 
                    'Early Stop Triggered'
                ])
            
            # Write metrics
            writer.writerow([
                model_arch,
                dataset.upper(),
                epochs,
                epochs_completed,
                lr,
                batch_size,
                f"{train_acc:.2f}",
                f"{test_acc:.2f}",
                "Yes" if early_stop else "No"
            ])
        
        print(f"Metrics saved to {csv_path}")