import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.transforms.transforms import Normalize, Resize, ToTensor
import config 

class AptosDataset(Dataset):

    def __init__(self):
        self.train_data = pd.read_csv(config.TRAIN_CSV_FILE)
        self.train_folder = config.TRAIN_IMG_FOLDER
        self.img_transform = transforms.Compose([
                                transforms.Resize(config.IMG_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        N = self.train_data.shape[0]
        return N

    def __getitem__(self, idx):
        y = self.train_data['diagnosis'].iloc[idx]
        img_name = self.train_data['id_code'].iloc[idx]
        x = self.get_image(img_name)
        x = self.img_transform(x)
        
        y = torch.tensor(y)

        data = {
            'x': x,
            'y': y
        }
        return data

    def get_image(self, img_name):
        img_path = config.TRAIN_IMG_FOLDER + img_name + '.png'
        img = Image.open(img_path)

        return img


if __name__ == "__main__":
    idx = 10
    data = AptosDataset()
    print("Dataset Length: ", data.__len__())
    sample = data[idx]
    img = sample['x']
    target = sample['y']
    print("Shape of Input: ", img.shape)
    print("Target: ", target)