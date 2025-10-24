from torch import no_grad, device, cuda
from models.dnn import ResNet18For10class
from models.vit import ViTFor10Class
from attack.attack_types import UntargetedAttack
from dataloader.cifarloader import CifarDataLoader
dev = device("cuda" if cuda.is_available() else "cpu")

if __name__ == "__main__":
    #entry point of program
    #model = ResNet18For10class("cifar10")
    model = ViTFor10Class("cifar10")
    data = CifarDataLoader()
    tries = 0
    success = 0
    
    if model.isPretrainOnDisk():
        model.load()
        model = model.to(device=dev)
    else:
        model = model.to(device=dev)
        training = data.get_train_data()
        model.training_loop(training)
        model.save()

    test = data.get_test_data_iterable()
    with no_grad():
        for image, label in test:
            image = image.to(device=dev)
            label = label.to(device=dev)
            _, pred = model(image).max(1)
            success += (label == pred).sum()
            tries += pred.size(0)
    acc = 100 * (success/tries)
    print(f"clean accuracy={acc}")

    adv = data.get_random_test_slice()
    attack = UntargetedAttack(model, model.getLoss(), adv, lr=.001)
    attack.execute_attack()