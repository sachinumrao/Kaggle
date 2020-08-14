import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.transforms.transforms import Normalize, Resize, ToTensor
import config
from config import IMG_SIZE 

class AptosDataset(Dataset):

    def __init__(self, df):

        self.data = df
        self.data_folder = config.TRAIN_IMG_FOLDER
        self.img_transform = transforms.Compose([
                                transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

    def __len__(self):
        N = self.data.shape[0]
        return N

    def __getitem__(self, idx):
        y = self.data['diagnosis'].iloc[idx]
        img_name = self.data['id_code'].iloc[idx]
        x = self.get_image(img_name)
        x = self.img_transform(x)
        
        y = torch.tensor(y)

        data = (x, y)
        return data

    def get_image(self, img_name):
        img_path = self.data_folder + img_name + '.png'
        img = Image.open(img_path)

        return img


if __name__ == "__main__":
    idx = 10
    data = AptosDataset()
    print("Dataset Length: ", data.__len__())
    sample = data[idx]
    img = sample['img']
    target = sample['target']
    print("Shape of Input: ", img.shape)
    print("Target: ", target)