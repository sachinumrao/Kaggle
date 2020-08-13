import torch
import torch.nn as nn
import torch.nn.functional as F 


class BaseCNNModel(nn.Module):

    def __init__(self, num_classes):
        super(BaseCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1)

        self.conv3 = nn.Conv2d(32, 64, 5, 1)
        self.conv4 = nn.Conv2d(64, 64, 5, 1)

        self.conv5 = nn.Conv2d(64, 128, 3, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1)

        self.pool = nn.MaxPool2d(2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.drop = nn.Dropout(0.15)

        n = 15488
        self.fc1 = nn.Linear(n, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # extract batch size
        bs = x.shape[0]

        # first conv block
        out = self.bn1(self.pool(F.relu(self.conv2(self.conv1(x)))))
        # print("Shape after first conv block: ", out.shape)

        # second conv block
        out = self.bn2(self.pool(F.relu(self.conv4(self.conv3(out)))))
        # print("Shape after second conv block: ", out.shape)

        # third conv block
        out = self.bn3(self.pool(F.relu(self.conv6(self.conv5(out)))))
        # print("Shape after third conv block: ", out.shape)

        # flatten the input
        out = out.view(bs, -1)
        # print("Shape after flattening: ", out.shape)

        # linear block
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out


if __name__ == "__main__":
    num_classes = 5
    model = BaseCNNModel(num_classes)
    x = torch.rand((1, 3, 256, 256))
    output_shape = model(x)