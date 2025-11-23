import argparse
import os
import sys
import csv
from PIL import Image
from torchvision import transforms
from Architecture.ResNet import ResNetAlt
from Architecture.ViT import ViT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Attack')
    parser.add_argument('--model', type=str, choices=['resnet', 'vit'], default='vit', help='Model architecture to attack')
    parser.add_argument('--model_dataset', type=str, help='Model to attack')
    parser.add_argument('--classes', type=int, default=10, help='Number of classes in attack')
    parser.add_argument('--adversarial', type=str, default='transfer/', 
                       help='Path to folder containing CPGD class mapping CSV files (for headless mode)')
    args = parser.parse_args()
    reader = csv.reader(os.path.join(args.adversarial, "attack.csv"))

    model = None
    if args.model == 'resnet':
        model = ResNetAlt(args.model_dataset, args.classes)
    elif args.model == 'vit':
        model = ViT(args.model_dataset, args.classes)

    if model is None:
        print(f'Bad model name: {args.model}')
        sys.exit()
    model.load()
    num_img = 0, adv_miss = 0, gen_miss = 0, true_hit = 0
    transform = transforms.ToTensor()
    for row in reader:
        num_img = num_img + 1
        truelbl = row[0]
        advlbl = row[1]
        img = transform(Image.open(row[2]))
        pred = model(img)
        iscorrect = pred[0] == truelbl
        isadvcorrect = not iscorrect and pred[0] == advlbl
        if iscorrect:
            true_hit = true_hit + 1
        elif isadvcorrect:
            adv_miss = adv_miss + 1
        else:
            gen_miss = gen_miss + 1
    acc = 100 * (true_hit / num_img)
    miss = 100 * (gen_miss / num_img) if gen_miss > 0 else 0
    advmiss = 100 * (adv_miss / num_img) if adv_miss > 0 else 0
    print(f"Number of images in adversarial dataset: {num_img}\n")
    print(f"Accuracy: {acc}\n")
    print(f"Misclassification (non-adv): {miss}\n")
    print(f"Misclassification (adv): {advmiss}\n")



