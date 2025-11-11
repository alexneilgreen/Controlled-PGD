import argparse
import os
import torch
import csv
from Architecture.ResNet import ResNetAlt
from Architecture.ViT import ViT
from Architecture.VLM import VLM
from Attack.Classes import UntargetedAttack, TargetedAttack
from Data_Loaders.Data_Loader import get_dataloader, get_num_classes, get_image_size_for_model, get_dataset_labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_class_map(path):
    mapping = {}
    with open(path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            mapping[row[0]] = row[1]
    return mapping

def main():
    parser = argparse.ArgumentParser(description='Controlled PGD Project')
    parser.add_argument('--train', type=bool, action='store_true')
    parser.add_argument('--model', type=str, choices=['resnet', 'vit', 'vlm'], default='resnet',
                       help='Model architecture to train')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar100', 'stl10'], 
                       default='mnist', help='Dataset to use')
    parser.add_argument('--model_path', type=str, help='The pretrained model on disk')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    # Attack arguments
    parser.add_argument('--type', type=str, choices=['PGD', 'CPGD'], default='PGD', help='Type of attack')
    parser.add_argument('--map', type=str, help='The class map on disk')
    parser.add_argument('--iterations', type=int, default=100, help='Number of attack iterations')
    parser.add_argument('--tolerance', type=float, default=0.000001, help='Attack convergence tolerance')
    parser.add_argument('--alpha', type=float, default=0.01, help='Attack step size')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Maximum perturbation (L-infinity norm)')
    
    args = parser.parse_args()
    
    num_classes = get_num_classes(args.dataset)
    img_size = get_image_size_for_model(args.model, args.dataset)
    
    if args.model == 'resnet':
        model = ResNetAlt(args.dataset, num_classes)
    elif args.model == 'vit':
        model = ViT(args.dataset, num_classes)
    elif args.model == 'vlm':
        model = VLM(args.dataset, get_dataset_labels(args.dataset))
    
    if args.train:
        os.makedirs('./Output/Models', exist_ok=True)
        
        train_loader = get_dataloader(
            dataset_name=args.dataset,
            split='train',
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            target_size=img_size
        )
        
        test_loader = get_dataloader(
            dataset_name=args.dataset,
            split='test',
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            target_size=img_size
        )
        
        model = model.to(device=device)
        model.training_loop(train_loader, epochs=args.epochs, lr=args.lr, test_loader=test_loader, batch_size=args.batch_size)
        model.save()
    else:
        model.load()
        model = model.to(device=device)
    
    loss_fn = model.getLoss()
    
    os.makedirs('./Output/Results', exist_ok=True)
    if args.type == 'PGD':
        print("\nExecuting PGD (Untargeted) Attack...")
        save_path = f"./Output/Results/{args.model}_{args.dataset}_pgd.txt"
        attack = UntargetedAttack(
            model=model,
            loss=loss_fn,
            dataloader=test_loader,
            save_path=save_path,
            iterations=args.iterations,
            tolerance=args.tolerance,
            epsilon=args.epsilon,
            alpha=args.alpha
        )
        attack.execute_attack()
    else:
        print("\nExecuting CPGD (Targeted) Attack...")
        mapping = parse_class_map(args.map)
        print(f"\nClass Mapping: {mapping}")
        
        save_path = f"./Output/Results/{args.model}_{args.dataset}_cpgd.txt"
        attack = TargetedAttack(
            model=model,
            loss=loss_fn,
            dataloader=test_loader,
            num_classes=num_classes,
            mapping=mapping,
            save_path=save_path,
            iterations=args.iterations,
            tolerance=args.tolerance,
            epsilon=args.epsilon,
            alpha=args.alpha
        )
        attack.execute_attack()


if __name__ == "__main__":
    main()