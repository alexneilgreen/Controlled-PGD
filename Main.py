from torch import no_grad, device, cuda
from models.dnn import ResNet18
from models.vit import ViT
from attack.attack_types import UntargetedAttack
from dataloader.Data_Loader import get_dataloader, get_random_test_slice
import argparse
dev = device("cuda" if cuda.is_available() else "cpu")

def determine_num_classes(dsname):
    if dsname == "cifar100":
        return 100
    return 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPGD main script")
    mg = parser.add_argument_group("model")
    dg = parser.add_argument_group("dataset")
    dg.add_argument("--dataset", type=str,choices=["cifar10","cifar100","mnist","stl10"],
                        default="cifar10",
                        help="Dataset to load. Correlates to a trained model.")
    dg.add_argument("--batchsize", type=int, default=64, help="batch size of dataset.")
    dg.add_argument("--workers", type=int, default=4, help="dataset workers.")
    mg.add_argument("--modeltype", type=str, required=True, choices=["resnet", "vit"],
                         help="The model architecture to use.")
    mg.add_argument("--pretrainedpath", type=str, required=False, default=None,
                        help="Overrides training a new model, allows for testing pretrained" \
                        " against new dataset. Optional.")
    args = parser.parse_args()

    classes = determine_num_classes(args.dataset)

    if args.modeltype == "resnet":
        model = ResNet18(args.dataset, classes)
    elif args.modeltype == "vit":
        model = ViT(args.dataset, classes)
    
    if args.pretrainedpath is not None:
        model.set_pretrain_path(args.pretrainedpath)
    
    #data = CifarDataLoader()
    tries = 0
    success = 0
    
    if model.isPretrainOnDisk():
        model.load()
        model = model.to(device=dev)
    else:
        model = model.to(device=dev)
        train = get_dataloader(args.dataset, batch_size=args.batchsize,
                                num_workers=args.workers)
        model.training_loop(train)
        model.save()

    
    test = get_dataloader(args.dataset, split="test", batch_size=args.batchsize,
                        num_workers=args.workers)
    with no_grad():
        for image, label in test:
            image = image.to(device=dev)
            label = label.to(device=dev)
            _, pred = model(image).max(1)
            success += (label == pred).sum()
            tries += pred.size(0)
    acc = 100 * (success/tries)
    print(f"clean accuracy={acc}")

    adv = get_random_test_slice(args.dataset, size=128, batch_size=args.batchsize,
                        num_workers=args.workers)
    attack = UntargetedAttack(model, model.getLoss(), adv, lr=.001)
    attack.execute_attack()