import torch
import torch.nn as nn
import torch.nn.functional as F 


class BaseCNNModel(nn.Module):

    def __init__(self, input_shape, num_classes):
        super(BaseCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1)

        n = self.get_dims(input_shape)
        self.fc1 = nn.Linear(n, 128)


    def forward(self, x):
        pass

    def get_dims(self, x):
        pass