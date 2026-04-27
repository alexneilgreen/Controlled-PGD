# Controlled PGD (CPGD) Attack Framework

A comprehensive PyTorch framework for training deep learning models and evaluating their robustness against PGD (Projected Gradient Descent) and CPGD (Controlled PGD) attacks.

## Project Structure

```
.
├── Main.py                          # Main entry point for training and attacking
├── requirements.txt                 # Python dependencies
│
├── Architecture/
│   ├── ResNet.py                   # ResNet18 implementation
│   └── ViT.py                      # Vision Transformer implementation
│
├── Attack/
│   ├── Classes.py                  # Attack wrapper classes
│   ├── PGD.py                      # PGD (untargeted) attack
│   ├── CPGD.py                     # CPGD (targeted) attack
│   └── Mapping/                    # Class mapping files for CPGD
│       ├── 10class.csv
│       ├── 100class.csv
│
├── Data_Loaders/
│   └── Data_Loader.py              # Dataset loading utilities
│
├── Output/
│   ├── Results/
│   │   └── Reporter.py             # Attack result reporting
│   └── Models/                     # Saved trained models
│       └── Training_Metrics.csv    # Training history
│
└── Data/                           # Downloaded datasets (auto-created)
```

## Installation

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

**Requirements:**
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- transformers >= 4.30.0
- evaluate >= 0.4.0
- datasets >= 2.12.0
- accelerate >= 0.26.0
- numpy >= 1.24.0
- pillow >= 9.5.0

## Quick Start

### Training Models

Basic command to train all models on all datasets:

```bash
python Main.py --mode train
```

Train a specific model on a specific dataset:

```bash
python Main.py --mode train --model resnet --dataset cifar10
```

### Attacking Models

Basic command to attack all trained models with PGD:

```bash
python Main.py --mode attack --type PGD
```

Attack all models with CPGD (targeted attack):

```bash
python Main.py --mode attack --type CPGD
```

Attack with interactive menu:

```bash
python Main.py --mode attack --attack_menu
```

## Training Mode Arguments

| Argument | Type | Choices | Default | Description |
|----------|------|---------|---------|-------------|
| `--mode` | str | train, attack | **Required** | Operating mode |
| `--model` | str | resnet, vit, all | all | Model architecture to train |
| `--dataset` | str | mnist, kmnist, cifar10, cifar100, stl10, svhn, all | all | Dataset to use |
| `--epochs` | int | - | 15 | Number of training epochs |
| `--lr` | float | - | 0.001 | Learning rate |
| `--batch_size` | int | - | 64 | Batch size for training |
| `--num_workers` | int | - | 4 | Number of dataloader workers |
| `--retrain` | flag | - | False | Force retrain existing models |

## Attack Mode Arguments

| Argument | Type | Choices | Default | Description |
|----------|------|---------|---------|-------------|
| `--mode` | str | train, attack | **Required** | Operating mode |
| `--iterations` | int | - | 100 | Number of attack iterations |
| `--tolerance` | float | - | 0.000001 | Attack convergence tolerance |
| `--alpha` | float | - | 0.01 | Attack step size |
| `--epsilon` | float | - | 0.3 | Maximum perturbation (L∞ norm) |
| `--attack_menu` | flag | - | False | Use interactive attack menu |
| `--type` | str | PGD, CPGD | PGD | Attack type (for headless mode) |
| `--map` | str | - | Attack/Mapping | Path to CPGD mapping folder |
| `--batch_size` | int | - | 64 | Batch size for evaluation |
| `--num_workers` | int | - | 4 | Number of dataloader workers |

## Supported Datasets

- **MNIST**: Handwritten digits (28×28, grayscale)
- **KMNIST**: Japanese characters (28×28, grayscale)
- **CIFAR-10**: Natural images, 10 classes (32×32, RGB)
- **CIFAR-100**: Natural images, 100 classes (32×32, RGB)
- **STL-10**: Natural images, 10 classes (96×96, RGB)
- **SVHN**: Street View House Numbers (32×32, RGB)

All datasets are automatically downloaded on first use.

## Model Architectures

### ResNet18
- Custom implementation from scratch
- Trained on 32×32 images (MNIST, KMNIST, CIFAR-10/100, SVHN)
- Trained on 96×96 images (STL-10)
- Mixed precision training (FP16)
- Early stopping with patience of 2 epochs

### Vision Transformer (ViT-Tiny)
- Based on `WinKawaks/vit-tiny-patch16-224`
- All images resized to 224×224
- Fine-tuned for each dataset
- Uses HuggingFace Trainer

## Attack Algorithms

### PGD (Projected Gradient Descent)
Untargeted white-box attack that maximizes classification loss:
- Iteratively perturbs input images
- Projects perturbations to L∞ ball of radius ε
- Goal: Cause misclassification to any wrong class

### CPGD (Controlled PGD)
Targeted white-box attack using predefined class mappings:
- Uses CSV mapping files to specify source → target class
- Minimizes loss for target class (reverse gradient)
- Goal: Force misclassification to specific target class
- Requires mapping file with appropriate number of classes

## Class Mapping Files

CPGD attacks require class mapping CSV files in the format:
```
0,5
1,3
2,7
...
```

Where each line maps `source_class,target_class`.

Provided mappings:
- `10class.csv`: For MNIST, KMNIST, CIFAR-10, STL-10, SVHN
- `100class.csv`: For CIFAR-100

## Output Files

### Training Metrics
Saved to `Output/Models/Training_Metrics.csv`:
- Model architecture
- Dataset
- Epochs completed
- Learning rate and batch size
- Training and test accuracy
- Early stopping indicator

### Attack Results
Saved to `Output/Results/{iterations}_iterations_{alpha}_alpha/`:
- Global Attack Success Rate (GASR)
- Model accuracy under attack
- Individual Attack Success Rate (IASR) per class
- For CPGD: Targeted success rates per class
- Example visualization images (when using interactive mode)

## Notes

- GPU acceleration is automatically used when available
- Models are saved after training and can be reused
- Use `--retrain` flag to overwrite existing models
- Attack results are organized by hyperparameters for easy comparison
- Interactive mode allows generating visualization examples of attacks

## Authors

**Alexander Green** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Master of Science in Computer Engineering (MSCpE)<br>
Dept. of Electrical and Computer Engineering, University of Central Florida<br>
[GitHub Home](https://github.com/alexneilgreen)

**Ernest Wheaton III** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Master of Science in Computer Science (MSCS)<br>
Dept. of Computer Science, University of Central Florida<br>
[GitHub Home](https://github.com/chivey-gnome)