import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Architecture.ResNet import ResNetAlt
from Architecture.ViT import ViT
from Attack.Classes import UntargetedAttack, TargetedAttack
from Data_Loaders.Data_Loader import get_dataloader, get_available_datasets, get_num_classes, get_image_size_for_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_models_mode(args):
    """Handle training models mode."""
    print(f"Using device: {device}")
    
    os.makedirs('./Output/Models', exist_ok=True)
    
    if args.model == 'all':
        models_to_train = ['resnet', 'vit']
    else:
        models_to_train = [args.model]
    
    if args.dataset == 'all':
        datasets_to_train = get_available_datasets()
    else:
        datasets_to_train = [args.dataset]
    
    for model_name in models_to_train:
        for dataset_name in datasets_to_train:
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()} on {dataset_name.upper()}")
            print(f"{'='*60}")
            
            num_classes = get_num_classes(dataset_name)
            img_size = get_image_size_for_model(model_name, dataset_name)
            
            if model_name == 'resnet':
                model = ResNetAlt(dataset_name, num_classes)
            elif model_name == 'vit':
                model = ViT(dataset_name, num_classes)
            
            if model.isPretrainOnDisk() and not args.retrain:
                print(f"Model already exists. Skipping...")
                print("Use --retrain flag to retrain existing models.")
                continue
            
            print(f"Image size for {model_name}: {img_size}x{img_size}")
            
            train_loader = get_dataloader(
                dataset_name=dataset_name,
                split='train',
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                target_size=img_size
            )
            
            test_loader = get_dataloader(
                dataset_name=dataset_name,
                split='test',
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                target_size=img_size
            )
            
            model = model.to(device=device)
            model.training_loop(train_loader, epochs=args.epochs, lr=args.lr, test_loader=test_loader, batch_size=args.batch_size)
            model.save()

def get_available_models():
    """Get list of available trained models."""
    models_dir = './Output/Models'
    if not os.path.exists(models_dir):
        return []
    
    model_files = []
    for item in os.listdir(models_dir):
        item_path = os.path.join(models_dir, item)
        if item.endswith('.pth') or os.path.isdir(item_path):
            model_files.append(item)
    return model_files

def get_class_mapping(num_classes):
    """Prompt user for class mapping for CPGD."""
    print("\nPlease input Matrix Mapping Values")
    mapping = {}
    for i in range(num_classes):
        while True:
            try:
                target = int(input(f"Class {i} -> "))
                if 0 <= target < num_classes:
                    mapping[i] = target
                    break
                else:
                    print(f"Invalid target. Must be between 0 and {num_classes-1}")
            except ValueError:
                print("Invalid input. Please enter a number.")
    return mapping

def attack_models_mode(args):
    """Handle attacking models mode."""
    print(f"Using device: {device}")
    
    available_models = get_available_models()
    
    if not available_models:
        print("No trained models found. Please train models first.")
        return
    
    print("\nAvailable trained models:")
    for idx, model_file in enumerate(available_models):
        print(f"{idx+1}. {model_file}")
    
    while True:
        try:
            selection = int(input("\nSelect model number: ")) - 1
            if 0 <= selection < len(available_models):
                selected_model_file = available_models[selection]
                break
            else:
                print(f"Invalid selection. Please choose 1-{len(available_models)}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Parse model name and dataset from filename
    model_parts = selected_model_file.replace('.pth', '').split('_')
    if len(model_parts) >= 2:
        model_name = model_parts[0].lower()
        dataset_name = model_parts[1].lower()
    else:
        # Handle ViT directory format
        model_name = 'vit' if 'ViT' in selected_model_file else 'resnet'
        dataset_name = selected_model_file.split('_')[1].lower()
    
    print(f"\nSelected: {model_name.upper()} trained on {dataset_name.upper()}")
    
    num_classes = get_num_classes(dataset_name)
    img_size = get_image_size_for_model(model_name, dataset_name)
    
    if model_name == 'resnet':
        model = ResNetAlt(dataset_name, num_classes)
    elif model_name == 'vit':
        model = ViT(dataset_name, num_classes)
    
    model.load()
    model = model.to(device=device)
    print("Model loaded successfully!")
    
    test_loader = get_dataloader(
        dataset_name=dataset_name,
        split='test',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        target_size=img_size
    )
    
    print("\nSelect attack type:")
    print("1. PGD (Untargeted)")
    print("2. CPGD (Targeted)")
    
    while True:
        try:
            attack_choice = int(input("\nSelect attack (1 or 2): "))
            if attack_choice in [1, 2]:
                break
            else:
                print("Invalid selection. Please choose 1 or 2")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    loss_fn = model.getLoss()
    
    # Create Results directory
    os.makedirs('./Output/Results', exist_ok=True)
    
    if attack_choice == 1:
        print("\nExecuting PGD (Untargeted) Attack...")
        save_path = f"./Output/Results/{model_name}_{dataset_name}_pgd.txt"
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
        mapping = get_class_mapping(num_classes)
        print(f"\nClass Mapping: {mapping}")
        
        save_path = f"./Output/Results/{model_name}_{dataset_name}_cpgd.txt"
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

def main():
    parser = argparse.ArgumentParser(description='Controlled PGD Project')
    parser.add_argument('--mode', type=str, choices=['train', 'attack'], required=True,
                       help='Mode: train models or implement attacks')
    
    # Training arguments
    parser.add_argument('--model', type=str, choices=['resnet', 'vit', 'all'], default='all',
                       help='Model architecture to train')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar100', 'stl10', 'all'], 
                       default='all', help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--retrain', action='store_true', help='Retrain existing models')
    
    # Attack arguments
    parser.add_argument('--iterations', type=int, default=100, help='Number of attack iterations')
    parser.add_argument('--tolerance', type=float, default=0.000001, help='Attack convergence tolerance')
    parser.add_argument('--alpha', type=float, default=0.01, help='Attack step size')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Maximum perturbation (L-infinity norm)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_models_mode(args)
    elif args.mode == 'attack':
        attack_models_mode(args)

if __name__ == "__main__":
    main()