import PIL
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random

file_path = "Fruits"

def load_data(folder, set_fruit): 

  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  data = []

  for fruit in os.listdir(folder):
    if fruit.lower() != set_fruit.lower():
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

        original_size = len(img_list)

        if original_size > 800:
          img_list = img_list[:800]

        while len(img_list) < 800:
          mod = random.randint(0, 5)
          img_list.append(mod)

        print(fruit, ripeness)
        print(len(img_list))

        for count, img in enumerate(img_list):
              
            if count >= original_size:

              temp = random.randint(0, original_size-1)
              temp_str = img_list[temp]
              while ".ini" in temp_str:
                temp = random.randint(0, original_size-1)
                temp_str = img_list[temp]

              f_path = os.path.join(i_path, temp_str)
              temp_image = PIL.Image.open(f_path).convert("RGB")
              
              match img:
                case 0:
                  rot = random.randint(0, 180) + 90
                  mod_img = temp_image.rotate(rot)                  
                case 1:
                  mod_img = temp_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                case 2:
                  blr_rad = random.randint(3, 5)
                  mod_img = temp_image.filter(PIL.ImageFilter.GaussianBlur(radius=blr_rad))
                case 3:
                  if random.randint(0,1) == 1:
                    brightness_factor = random.uniform(0.3, 0.5)
                  else:
                    brightness_factor = random.uniform(1.5, 1.8)
                  enhancer = PIL.ImageEnhance.Brightness(temp_image)
                  mod_img = enhancer.enhance(brightness_factor)
                case 4:
                  temp_array = np.array(temp_image)
                  noise = np.random.normal(0, 1, temp_array.shape[:2])
                  noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
                  noise*=10
                  noise = noise.astype(np.uint8)
                  noise = np.clip(noise, 0, 255)
                  mod_array = temp_array[:, :, :] + noise
                  mod_array = np.clip(mod_array, 0, 255)
                  mod_img = PIL.Image.fromarray(mod_array.astype(np.uint8))
                case 5:
                  shear_x = 0
                  shear_y = 0
                  if random.randint(0,1) == 1:
                    shear_x = random.randint(1, 3)
                  else:
                    shear_y = random.randint(1, 3)
                  mod_img = temp_image.transform(temp_image.size, PIL.Image.AFFINE, 
                                                 (1, shear_x, -(temp_image.size[1]/2)*shear_x, 1, shear_y, -(temp_image.size[0]/2)*shear_y))
              image = mod_img

            else:       
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

fruit = "Banana"
data = load_data(file_path, fruit)

print(f"Dataset Length: {len(data)}")

torch.save(data, f'test_dataset_{fruit}.pth')
