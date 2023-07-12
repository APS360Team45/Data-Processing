import PIL
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random

file_path = "Fruits"

def load_data(folder): 

  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  data = []

  for fruit in os.listdir(folder):
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

        if original_size > 1000:
          img_list = img_list[:1000]

        while len(img_list) < 1000:
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

                  mod_img = temp_image.transform(temp_image.size, PIL.Image.AFFINE, (1, shear_x, -(temp_image.size[1]/2)*shear_x, 1, shear_y, -(temp_image.size[0]/2)*shear_y))

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

######################################################
###### Getting train loaders for training ############

def load_training_data(images, train_p=.8, val_p = .1, batch_size=64): # This function takes in the raw shuffled data and outputs object of type dataloader with specified batches

  total_size = len(images)

  train_size = int(train_p * total_size)
  val_size = int(val_p * total_size + train_size)

  # splits data into training, validation, and testing datasets
  train = images[:train_size]
  val = images[train_size:val_size]
  test = images[val_size:]

  # creates and returns a dataloader from each of the previously split datasets
  trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=1)
  if len(val) > 0: valloader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=1)
  else: valloader = 0
  if len(test) > 0: testloader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=1)
  else: testloader = 0

  return trainloader, valloader, testloader

data = load_data(file_path)

train_loader, val_loader, test_loader = load_training_data(data)