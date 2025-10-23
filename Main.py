from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch import randperm, no_grad, device, cuda
from models.dnn import ResNet18For10class
from attack.attack_types import UntargetedAttack
dev = device("cuda" if cuda.is_available() else "cpu")

if __name__ == "__main__":
    #entry point of program
    model = ResNet18For10class("cifar10test")

    # move this to some data loader abstraction
    # DO NOT continue to pollute this file with implementation
    rgbT = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            # cifar10 normalization values (known)
            transforms.Normalize((.5,.5,.5), (.5,.5,.5))
        ])
    # Much of this code is adapted from HW1
    # There is a lot here that can be dropped.
    # since this is an initial test script, cleanliness is not needed
    # but it will be needed at some point
    cifar10data = CIFAR10(root='./data', train=False, download=True, transform=rgbT)
    testDataLoader = DataLoader(cifar10data, batch_size=64, shuffle=True)
    cifartraindata = CIFAR10("./data", train=True, download=True, transform=rgbT)
    indices = randperm(len(cifar10data))[:64]
    advData = Subset(cifar10data, indices)
    advDataLoader = DataLoader(advData, batch_size=64, shuffle=True)
    model = model.to(device=dev)
    tries = 0
    success = 0
    
    if model.isPretrainOnDisk():
        model.load()
    else:
        model.training_loop(cifartraindata, None)
        model.save()

    with no_grad():
        for image, label in testDataLoader:
            image = image.to(device=dev)
            label = label.to(device=dev)
            _, pred = model(image).max(1)
            success += (label == pred).sum()
            tries += pred.size(0)
    acc = 100 * (success/tries)
    print(f"clean accuracy={acc}")

    attack = UntargetedAttack(model, model.getLoss(), advDataLoader, lr=.001)
    attack.execute_attack()