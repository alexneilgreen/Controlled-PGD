import os
import csv
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image

class TransferOutput:
    def __init__(self, output_path):
        os.mkdir(output_path)
        self.path = os.path.join(output_path, "attack.csv")
        self.img_path = os.join(output_path, "img")
        os.mkdir(self.img_path)
        self.img_counter = 0

    def output(self, true_label, adv_label, images):
        WIDTH = 12
        with open(self.path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            for i in range(len(true_label)):
                row = []
                row.append(true_label[i].item())
                row.append(adv_label[i].item())
                img = to_pil_image(images[i])
                out_name = os.join(self.img_path, f"{self.img_counter:0{WIDTH}d}.jpg")
                self.img_counter = self.img_counter + 1
                img.save(out_name)
                row.append(out_name)
                writer.writerow(row)
            
