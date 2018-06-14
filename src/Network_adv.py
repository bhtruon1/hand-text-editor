import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout_rate = 0.8

        # 64*64
        self.conv1 = nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)

        # 32*32
        self.conv2 = nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)

        # 16*16
        self.conv3 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        # 8*8
        self.conv4 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        # 4*4
        self.fc1 = nn.Linear(4 * 4 * 32, 1024)
        self.fcbn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 38)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # size 3 x 64 x 64
        x = self.bn1(self.conv1(x))
        x = F.relu(self.pool(x)) 
        x = self.bn2(self.conv2(x)) 
        x = F.relu(self.pool(x)) 
        x = self.bn3(self.conv3(x))
        x = F.relu(self.pool(x))
        x = self.bn4(self.conv4(x))
        x = F.relu(self.pool(x))
        # Flatten
        x = x.view(-1, 4 * 4 * 32)

        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
