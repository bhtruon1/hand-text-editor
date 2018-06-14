import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.num_channels = 4  # output channel for the first layer
        self.dropout_rate = 0.8

        # 32*32
        self.conv1 = nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)

        # 16*16
        self.conv2 = nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)

        # 8*8
        self.conv3 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        # 4*4
        self.fc1 = nn.Linear(4 * 4 * 16, 128)
        self.fcbn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 38)

    def forward(self, s):
        # size 3 x 64 x 64
        s = self.bn1(self.conv1(s))  # batch_size x num_channels x 64 x 64
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 32 x 32
        s = self.bn2(self.conv2(s))  # batch_size x num_channels*2 x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*2 x 16 x 16
        s = self.bn3(self.conv3(s))  # batch_size x num_channels*4 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*4 x 8 x 8

        # Flatten
        s = s.view(-1, 4 * 4 * 16)  # batch_size x 8*8*num_channels*4

        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                      p=self.dropout_rate, training=self.training)  # batch_size x self.num_channels*4
        s = self.fc2(s)  # batch_size x 6

        return F.log_softmax(s, dim=1)
