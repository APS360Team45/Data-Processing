
import PIL
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random

file_path = "avid_picked_fruits"

def load_data(folder): 

  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  data = []

  for fruit in os.listdir(folder):
    if fruit == "Strawberry":
       continue
    n_path = os.path.join(folder, fruit)
    if ".ini" in n_path:
      continue
    for ripeness, file in enumerate(os.listdir(n_path)):
        i_path = os.path.join(n_path, file)
        if ".ini" in i_path:
          continue
        label = ripeness

        img_list = os.listdir(i_path)

        random.shuffle(img_list)

        for count, img in enumerate(img_list):
               
            if ".ini" in img:
                continue

            f_path = os.path.join(i_path, img)
            image = PIL.Image.open(f_path).convert("RGB")

            aspect_ratio = image.width / image.height

            if aspect_ratio < 1:
                new_width = 256
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = 256
                new_width = int(new_height * aspect_ratio)

            resized_image = image.resize((new_width, new_height))
            padded_image = PIL.ImageOps.pad(resized_image, (256, 256), color="black")
            hsv_image = padded_image.convert("HSV")
              
            data.append((transform(hsv_image), torch.tensor(label)))

  random.shuffle(data)

  return data

test = load_data(file_path)

torch.save(test, 'test_dataset_extra(avid).pth')