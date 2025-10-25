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

    acc = 100 * (success/tries)
    print(f"clean accuracy={acc}")

    # Prompt user for attack type
    print("\nSelect attack type:")
    print("\t1. PGD (Untargeted)")
    print("\t2. CPGD (Targeted)")
    
    while True:
        try:
            choice = int(input("Enter choice (1 or 2): "))
            if choice in [1, 2]:
                break
            else:
                print("Error: Please enter 1 or 2")
        except ValueError:
            print("Error: Please enter a valid integer")

    adv = data.get_random_test_slice()
    
    if choice == 1:
        attack = UntargetedAttack(model, model.getLoss(), adv, lr=.001)
    else:  # choice == 2
        from attack.attack_types import TargetedAttack
        attack = TargetedAttack(model, model.getLoss(), adv, num_classes=10, lr=.001)
    
    attack.execute_attack()