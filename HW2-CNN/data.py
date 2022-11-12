import os 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
import pandas as pd
from PIL import Image

class P1dataset(Dataset):
    def __init__(self, txt_path, img_dir, transform=None):
        self.imgdir = img_dir
        self.transform = transform
        self.df = pd.read_csv(os.path.join(os.getcwd(),"HW2_data/train.csv"))
        self.files = [f for f in os.listdir(self.imgdir)]

    def __len__(self):
        files = [f for f in os.listdir(self.imgdir)]
        return len(files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img = Image.open(os.path.join(self.imgdir, img_name))
        mask = self.df["id"]==int(img_name.split(".")[0])
        
        label = int(self.df[mask]["label"].values)
        
        if self.transform:
            img = self.transform(img)
        return img, label

class P1valid_dataset(Dataset):
    def __init__(self, img_dir):
        self.imgdir = img_dir
        
    def __len__(self):
        files = [f for f in os.listdir(self.imgdir)]
        return len(files)

    def __getitem__(self, idx):
        files = [f for f in os.listdir(self.imgdir)][idx]
        img = read_image(os.path.join(self.imgdir, files))
        name = files.split(".")[0]
        return img, name

# class P1dataset(Dataset):
#     def __init__(self, txt_path, img_dir, transform=None):
    
#         df = pd.read_csv(txt_path)[:-1]
#         self.img_dir = img_dir
#         self.txt_path = txt_path
#         self.img_names = df["id"].values
#         self.y = df['label'].values
#         self.transform = transform

#     def __getitem__(self, index):
#         img = Image.open(os.path.join(self.img_dir, str(self.img_names[index])+".jpg"))
        
#         if self.transform is not None:
#             img = self.transform(img)
        
#         label = self.y[index]
#         return img, label

#     def __len__(self):
#         return self.y.shape[0]
