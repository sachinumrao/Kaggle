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

    def __init__(self, df, isTest=False):

        self.data = df
        self.isTest = isTest
        if isTest:
            self.data_folder = config.TEST_IMG_FOLDER
        else:
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
        
        if not self.isTest:
            y = self.data['diagnosis'].iloc[idx]
        else:
            y = 0
            
        img_name = self.data['id_code'].iloc[idx]
        img_path = self.data_folder + img_name + '.png'
        x = self.get_image(img_path)
        x = self.img_transform(x)
        
        y = torch.tensor(y)

        if not self.isTest:
            data = (x, y)
        else:
            data = (x, y, img_name)
            
        return data

    def get_image(self, img_path):
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