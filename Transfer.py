import argparse
import os
import sys
import csv
import torch
from PIL import Image
from torchvision import transforms
from Architecture.ResNet import ResNetAlt
from Architecture.ViT import ViT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Attack')
    parser.add_argument('--model', type=str, choices=['resnet', 'vit'], default='vit', help='Model architecture to attack')
    parser.add_argument('--model_dataset', type=str, help='Model to attack')
    parser.add_argument('--classes', type=int, default=10, help='Number of classes in attack')
    parser.add_argument('--adversarial', type=str, default='transfer/', 
                       help='Path to folder containing CPGD class mapping CSV files (for headless mode)')
    args = parser.parse_args()
    with open(os.path.join(args.adversarial, "attack.csv")) as f:
        reader = csv.reader(f)

        model = None
        if args.model == 'resnet':
            model = ResNetAlt(args.model_dataset, args.classes)
        elif args.model == 'vit':
            model = ViT(args.model_dataset, args.classes)

        if model is None:
            print(f'Bad model name: {args.model}')
            sys.exit()
        model.load()
        num_img = [0] * 10
        adv_miss = [0] * 10
        gen_miss = [0] * 10
        true_hit = [0] * 10
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        model.to(device=device)
        for truelbl, advlbl, imgpath in reader:
            img = transform(Image.open(imgpath).convert("RGB")).to(device=device)
            img = img.unsqueeze(0)
            pred = model(img).argmax(dim=-1).item()
            num_img[pred] = num_img[pred] + 1
            iscorrect = pred == truelbl
            isadvcorrect = not iscorrect and pred == advlbl
            if iscorrect:
                true_hit[pred] = true_hit[pred] + 1
            elif isadvcorrect:
                adv_miss[pred] = adv_miss[pred] + 1
            else:
                gen_miss[pred] = gen_miss[pred] + 1

        
        for label in range(10):
            acc = 100 * (true_hit[label] / num_img[label]) if true_hit[label] > 0 else 0
            miss = 100 * (gen_miss[label] / num_img[label]) if gen_miss[label] > 0 else 0
            advmiss = 100 * (adv_miss[label] / num_img[label]) if adv_miss[label] > 0 else 0
            print(f"Class: {label}")
            print(f"Number of images classified to label: {num_img[label]}\n")
            print(f"Accuracy: {acc}\n")
            print(f"Misclassification (non-adv): {miss}\n")
            print(f"Misclassification (adv): {advmiss}\n\n")



