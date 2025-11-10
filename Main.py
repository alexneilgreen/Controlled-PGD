import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv
import matplotlib.pyplot as plt
import numpy as np

# from Architecture.ResNet import ResNet18
from Architecture.ResNet import ResNetAlt
from Architecture.ViT import ViT
from Attack.Classes import UntargetedAttack, TargetedAttack
from Data_Loaders.Data_Loader import get_dataloader, get_available_datasets, get_num_classes, get_image_size_for_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_class_map(path):
    """Parse class mapping from CSV file."""
    mapping = {}
    try:
        with open(path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                if len(row) >= 2:
                    mapping[int(row[0])] = int(row[1])
        return mapping
    except FileNotFoundError:
        print(f"Error: Mapping file '{path}' not found.")
        return None
    except ValueError:
        print(f"Error: Invalid mapping format in '{path}'. Expected integer values.")
        return None
    
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

def generate_attack_examples(model, test_loader, attack_obj, loss_fn, save_path, attack_type, num_examples=5):
    """Generate visualization comparing clean and adversarial images."""
    model.eval()
    
    # Get random samples
    all_images = []
    all_labels = []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
        if len(all_images) * images.size(0) >= num_examples:
            break
    
    all_images = torch.cat(all_images)[:num_examples]
    all_labels = torch.cat(all_labels)[:num_examples]
    
    # Move to device
    all_images = all_images.to(device)
    all_labels = all_labels.to(device)
    
    # Generate adversarial examples
    if attack_type == 'PGD':
        adv_images = attack_obj.pgd(all_images, all_labels, attack_obj.alpha, model, loss_fn)
    else:  # CPGD
        adv_images = attack_obj.cpgd(all_images, all_labels, attack_obj.alpha, model, loss_fn)
    
    # Get predictions
    with torch.no_grad():
        clean_outputs = model(all_images)
        adv_outputs = model(adv_images)
        _, clean_preds = clean_outputs.max(1)
        _, adv_preds = adv_outputs.max(1)
    
    # Move to CPU for visualization
    all_images = all_images.cpu()
    adv_images = adv_images.cpu()
    clean_preds = clean_preds.cpu()
    adv_preds = adv_preds.cpu()
    all_labels = all_labels.cpu()
    
    # Calculate perturbations (amplified for visibility)
    perturbations = adv_images - all_images
    perturbations_vis = (perturbations - perturbations.min()) / (perturbations.max() - perturbations.min() + 1e-8)
    
    # Create figure with 3x5 grid
    fig, axes = plt.subplots(3, num_examples, figsize=(15, 9))
    
    for i in range(num_examples):
        # Denormalize images from [-1, 1] to [0, 1]
        clean_img = (all_images[i].permute(1, 2, 0).numpy() + 1) / 2
        adv_img = (adv_images[i].permute(1, 2, 0).numpy() + 1) / 2
        pert_img = perturbations_vis[i].permute(1, 2, 0).numpy()
        
        # Clip to valid range
        clean_img = np.clip(clean_img, 0, 1)
        adv_img = np.clip(adv_img, 0, 1)
        
        # Top row: Clean images
        axes[0, i].imshow(clean_img)
        axes[0, i].set_title(f'True: {all_labels[i].item()}\nPred: {clean_preds[i].item()}', fontsize=9)
        axes[0, i].axis('off')
        
        # Middle row: Perturbations (amplified)
        axes[1, i].imshow(pert_img)
        axes[1, i].set_title('Perturbation\n(amplified)', fontsize=9)
        axes[1, i].axis('off')
        
        # Bottom row: Adversarial images
        axes[2, i].imshow(adv_img)
        axes[2, i].set_title(f'Adv Pred: {adv_preds[i].item()}', fontsize=9)
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.3, 0.5, 'Clean', transform=axes[0, 0].transAxes,
                    fontsize=12, va='center', ha='right', rotation=90, fontweight='bold')
    axes[1, 0].text(-0.3, 0.5, 'Perturbation', transform=axes[1, 0].transAxes,
                    fontsize=12, va='center', ha='right', rotation=90, fontweight='bold')
    axes[2, 0].text(-0.3, 0.5, 'Adversarial', transform=axes[2, 0].transAxes,
                    fontsize=12, va='center', ha='right', rotation=90, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    img_save_path = save_path.replace('.txt', '_examples.png')
    plt.savefig(img_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nExample images saved to: {img_save_path}")

def attack_single_model(model_file, args, attack_type, mapping_folder=None):
    """Attack a single model."""
    # Parse model name and dataset from filename
    model_parts = model_file.replace('.pth', '').split('_')
    if len(model_parts) >= 2:
        model_name = model_parts[0].lower()
        dataset_name = model_parts[1].lower()
    else:
        # Handle ViT directory format
        model_name = 'vit' if 'ViT' in model_file else 'resnet'
        dataset_name = model_file.split('_')[1].lower()
    
    print(f"\n{'='*60}")
    print(f"Attacking: {model_name.upper()} trained on {dataset_name.upper()}")
    print(f"{'='*60}")
    
    num_classes = get_num_classes(dataset_name)
    img_size = get_image_size_for_model(model_name, dataset_name)
    
    # Load appropriate mapping if CPGD
    mapping = None
    if attack_type == 'CPGD':
        if mapping_folder is None:
            print("Error: Mapping folder required for CPGD attacks.")
            print("Skipping this model...")
            return
        
        # Select correct mapping file based on num_classes
        if num_classes == 10:
            mapping_file = os.path.join(mapping_folder, '10class.csv')
        elif num_classes == 100:
            mapping_file = os.path.join(mapping_folder, '100class.csv')
        elif num_classes == 257:
            mapping_file = os.path.join(mapping_folder, '257class.csv')
        else:
            print(f"Error: No mapping available for {num_classes} classes.")
            print("Skipping this model...")
            return
        
        mapping = parse_class_map(mapping_file)
        if mapping is None:
            print("Skipping this model...")
            return
        
        if len(mapping) != num_classes:
            print(f"Error: Mapping has {len(mapping)} classes but dataset has {num_classes} classes.")
            print("Skipping this model...")
            return
    
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
    
    loss_fn = model.getLoss()
    
    # Create Results directory
    results_dir = f"./Output/Results/{args.iterations}_iterations_{args.alpha:.2f}_alpha"
    os.makedirs(results_dir, exist_ok=True)
    
    if attack_type == 'PGD':
        print("\nExecuting PGD (Untargeted) Attack...")
        save_path = f"{results_dir}/{model_name}_{dataset_name}_pgd.txt"
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
    else:  # CPGD
        print("\nExecuting CPGD (Targeted) Attack...")
        print(f"Using mapping: {mapping_file}")
        print(f"Class Mapping: {mapping}")
        
        save_path = f"{results_dir}/{model_name}_{dataset_name}_cpgd.txt"
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

def attack_models_mode(args):
    """Handle attacking models mode."""
    print(f"Using device: {device}")
    
    available_models = get_available_models()
    
    if not available_models:
        print("No trained models found. Please train models first.")
        return
    
    # Headless mode - attack all models
    if not args.attack_menu:
        print("\nHeadless Attack Mode")
        print(f"Attack Type: {args.type}")
        
        if args.type == 'CPGD':
            if not args.map:
                print("Error: --map argument is required for CPGD attacks in headless mode.")
                return
            
            if not os.path.isdir(args.map):
                print(f"Error: Mapping folder '{args.map}' not found.")
                return
            
            print(f"Using mapping folder: {args.map}")
        
        print(f"\nFound {len(available_models)} trained models")
        print("Attacking all models...")
        
        for model_file in available_models:
            attack_single_model(model_file, args, args.type, args.map if args.type == 'CPGD' else None)
        
        print("\n" + "="*60)
        print("All attacks completed!")
        print("="*60)
        return
    
    # Interactive menu mode
    else: 
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
        
        # Get mapping if CPGD
        mapping = None
        if attack_choice == 2:
            print("\nExecuting CPGD (Targeted) Attack...")
            mapping = get_class_mapping(num_classes)
            print(f"\nClass Mapping: {mapping}")
        
        # Prompt for example image generation
        while True:
            generate_examples = input("\nGenerate example images? (y/n): ").lower()
            if generate_examples in ['y', 'n', 'yes', 'no']:
                generate_examples = generate_examples.startswith('y')
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
        
        # Create Results directory with attack parameters
        results_dir = f"./Output/Results/{args.iterations}_iterations_{args.alpha:.2f}_alpha"
        os.makedirs(results_dir, exist_ok=True)
        
        if attack_choice == 1:
            print("\nExecuting PGD (Untargeted) Attack...")
            save_path = f"{results_dir}/{model_name}_{dataset_name}_pgd.txt"
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
            
            # Generate examples if requested
            if generate_examples:
                generate_attack_examples(model, test_loader, attack.pgd, loss_fn, save_path, 'PGD')
        else:
            save_path = f"{results_dir}/{model_name}_{dataset_name}_cpgd.txt"
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
            
            # Generate examples if requested
            if generate_examples:
                generate_attack_examples(model, test_loader, attack.cpgd, loss_fn, save_path, 'CPGD')

def main():
    parser = argparse.ArgumentParser(description='Controlled PGD Project')
    parser.add_argument('--mode', type=str, choices=['train', 'attack'], required=True,
                       help='Mode: train models or implement attacks')
    
    # Training arguments
    parser.add_argument('--model', type=str, choices=['resnet', 'vit', 'all'], default='all',
                       help='Model architecture to train')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'kmnist', 'cifar10', 'cifar100', 'stl10', 'svhn', 'all'], 
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

    # Headless arguments
    parser.add_argument('--attack_menu', action='store_true', help='Use interactive attack menu')
    parser.add_argument('--type', type=str, choices=['PGD', 'CPGD'], default='PGD', 
                       help='Type of attack (for headless mode)')
    parser.add_argument('--map', type=str, default='Attack/Mapping', 
                       help='Path to folder containing CPGD class mapping CSV files (for headless mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_models_mode(args)
    elif args.mode == 'attack':
        attack_models_mode(args)

if __name__ == "__main__":
    main()