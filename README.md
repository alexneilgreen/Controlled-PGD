<!-- SHOWCASE: true -->

# Controlled PGD (CPGD) Attack Framework

> A PyTorch framework for training ResNet18 and Vision Transformer models and evaluating their robustness against untargeted PGD and targeted CPGD adversarial attacks.

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Language](https://img.shields.io/badge/language-Python-blue)
![Semester](https://img.shields.io/badge/semester-Fall%202025-orange)

---

## Course Information

| Field                  | Details                                                                                                                                                                                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Course Title           | Special Topics: Trustworthy Machine Learning                                                                                                                                                                                                       |
| Course Number          | CAP 6938                                                                                                                                                                                                                                           |
| Semester               | Fall 2025                                                                                                                                                                                                                                          |
| Assignment Title       | Course Final Project                                                                                                                                                                                                                               |
| Assignment Description | Final course project on ML + Attack/Defense. Students select a topic within adversarial machine learning, ideally aligned with their own research, and deliver a proposal followed by a final presentation and report. Groups of up to 4 students. |

---

## Project Description

This framework trains ResNet18 and ViT-Tiny classifiers on six image datasets (MNIST, KMNIST, CIFAR-10, CIFAR-100, STL-10, SVHN) and evaluates their robustness against white-box adversarial attacks. The untargeted PGD attack iteratively perturbs inputs under an L-infinity constraint to induce misclassification, while the novel CPGD (Controlled PGD) attack extends this by minimizing loss toward a predefined target class using a CSV class-mapping file. Results are saved as structured reports containing global and per-class attack success rates, enabling direct comparison across architectures, datasets, and hyperparameter configurations.

---

## Screenshots / Demo

> _No screenshot available. Add one with: `![Demo](docs/your-image.png)`_

---

## Results

When run correctly, the framework produces per-model attack result files under `Output/Results/{iterations}_iterations_{alpha}_alpha/`. Each file reports:

```
=== PGD Attack Results ===
Attack Parameters: iterations=100, alpha=0.01, epsilon=0.3, tolerance=1e-06

Overall Accuracy Under Attack: 12.34%
Global Attack Success Rate (GASR): 87.66%

Per-Class Attack Success Rate (IASR):
  Class 0: 91.2%
  Class 1: 84.7%
  ...
```

For CPGD runs, an additional targeted success report shows how often each class was forced to its specified target. Training metrics for all model/dataset combinations are appended to `Output/Models/Training_Metrics.csv`, with columns for architecture, dataset, epochs completed, learning rate, batch size, train/test accuracy, and whether early stopping triggered.

Key indicators: a GASR above ~80% at default hyperparameters indicates a successful attack. If accuracy under attack remains high, try increasing `--iterations` or `--epsilon`. If training accuracy is unexpectedly low, verify GPU availability and inspect `Training_Metrics.csv` for early-stop events.

---

## Key Concepts

`adversarial-attacks` `projected-gradient-descent` `targeted-attacks` `l-infinity-norm` `white-box-attacks` `residual-networks` `vision-transformers` `image-classification` `mixed-precision-training` `early-stopping`

---

## Languages & Tools

- **Language:** Python 3.10+
- **Framework/SDK:** PyTorch, HuggingFace Transformers, HuggingFace Evaluate
- **Hardware:** CUDA-capable GPU (automatic CPU fallback supported)
- **Build System:** pip / requirements.txt

---

## File Structure

```
.
├── Main.py                          # CLI entry point: train and attack modes
├── requirements.txt                 # Python dependencies
│
├── Architecture/
│   ├── ResNet.py                    # Custom ResNet18 with mixed-precision training and early stopping
│   └── ViT.py                       # ViT-Tiny fine-tuning via HuggingFace Trainer
│
├── Attack/
│   ├── Classes.py                   # UntargetedAttack and TargetedAttack orchestration wrappers
│   ├── PGD.py                       # PGD untargeted attack implementation
│   ├── CPGD.py                      # CPGD targeted attack implementation
│   └── Mapping/
│       ├── 10class.csv              # Source->target class mapping for 10-class datasets
│       └── 100Class.csv             # Source->target class mapping for CIFAR-100
│
├── Data_Loaders/
│   └── Data_Loader.py               # Dataset download and dataloader utilities
│
├── Output/
│   ├── Results/
│   │   └── Reporter.py              # Attack result reporting (GASR, IASR, targeted success)
│   └── Models/                      # Saved model weights and Training_Metrics.csv
│
└── Data/                            # Downloaded datasets (auto-created on first run)
```

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (CPU fallback supported but significantly slower)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/alexneilgreen/UCF-CAP6938-CPGDAttackFramework.git
cd UCF-CAP6938-CPGDAttackFramework

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models on all datasets
python Main.py --mode train

# 4. Attack all trained models with PGD (untargeted, headless)
python Main.py --mode attack --type PGD

# 5. Attack all trained models with CPGD (targeted, headless)
python Main.py --mode attack --type CPGD --map Attack/Mapping

# 6. Interactive menu (select model and attack type, optionally save example images)
python Main.py --mode attack --attack_menu
```

### Controls

| Argument        | Default          | Description                                                         |
| --------------- | ---------------- | ------------------------------------------------------------------- |
| `--mode`        | required         | `train` or `attack`                                                 |
| `--model`       | `all`            | `resnet`, `vit`, or `all`                                           |
| `--dataset`     | `all`            | `mnist`, `kmnist`, `cifar10`, `cifar100`, `stl10`, `svhn`, or `all` |
| `--epochs`      | `15`             | Number of training epochs                                           |
| `--lr`          | `0.001`          | Learning rate                                                       |
| `--batch_size`  | `64`             | Batch size                                                          |
| `--retrain`     | `False`          | Force retrain existing models                                       |
| `--type`        | `PGD`            | Attack type in headless mode: `PGD` or `CPGD`                       |
| `--iterations`  | `100`            | Number of attack iterations                                         |
| `--alpha`       | `0.01`           | Attack step size                                                    |
| `--epsilon`     | `0.3`            | Maximum L-inf perturbation                                          |
| `--tolerance`   | `1e-6`           | Convergence tolerance                                               |
| `--map`         | `Attack/Mapping` | Path to CPGD mapping folder (headless mode)                         |
| `--attack_menu` | `False`          | Enable interactive attack menu                                      |

---

## Contributors

| Name               | Role         | GitHub                                             |
| ------------------ | ------------ | -------------------------------------------------- |
| Alexander Green    | Co-developer | [@alexneilgreen](https://github.com/alexneilgreen) |
| Ernest Wheaton III | Co-developer | [@chivey-gnome](https://github.com/chivey-gnome)   |

---

## Academic Integrity

This repository is publicly available for **portfolio and reference purposes only**.
Please do not submit any part of this work as your own for academic coursework.
