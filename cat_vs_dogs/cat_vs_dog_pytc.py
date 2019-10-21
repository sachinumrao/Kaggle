# import system libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import torch 
import torch
import torchvision
import torchvision.transforms as transforms

# improt nueral net module from torch
import torch.nn as nn
import torch.nn.functional as F

# import optimizer from torch
import torch.optim as optim

# define tranform for dataset images and convert them into tensor
transform = transforms.Compose([transforms.Resize((150,150)),
                                       transforms.ToTensor()
                                       ])

# load train dataset
train_data_path = '~/Data/Kaggle/dogs-vs-cats/train_data/'
train_set = torchvision.datasets.ImageFolder(
        root=train_data_path,
        transform=transform
    )

# create iterable data object for train set
train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=32,
    shuffle=True,
    num_workers=4)

# load validation set
val_data_path = '~/Data/Kaggle/dogs-vs-cats/val/'
val_set = torchvision.datasets.ImageFolder(
        root=val_data_path,
        transform=transform
    )

# create iterab;e data object for validation set
val_loader = torch.utils.data.DataLoader(val_set,
    batch_size=32,
    shuffle=True,
    num_workers=4)

# define convolutional network
model = torch.nn.Sequential(
    nn.Conv2d(3,64,3),
    nn.MaxPool2d(2),
    nn.Conv2d(32,16,3),
    nn.MaxPool2d(2,2),
    nn.Linear(16*36*36,10),
    nn.ReLU(),
    nn.Linear(10,1),
    nn.Sigmoid()

)

# define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# train the network in epochs
loss_records = []
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i,data in enumerate(train_loader, 0):
        inputs, labels = data

        # reset the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        #print(outputs.shape)
        # calculate loss
        loss = criterion(outputs, labels)

        # backward pass / calculate gradients
        loss.backward()

        # take one grad step
        optimizer.step()

        # store loss 
        loss_records.append(loss.item())

        # print stats
        if (i+1)%100 == 0:
            running_loss = loss.item()
            print("Epoch : ", epoch+1, " , Step : ", i+1, " , Loss : ",running_loss)

# test the model
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct +=  (predicted == labels).sum().item()


print("Accuracy : ", correct/total)

# draw loss value during training
import matplotlib.pyplot as plt
plt.plot(loss_records)
plt.show()