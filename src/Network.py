from __future__ import division
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


# class Net(torch.nn.Module):
#     # Our batch shape for input x is (3, 32, 32)
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
#         self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
#         self.fc2 = torch.nn.Linear(64, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = x.view(-1, 18 * 16 * 16)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return (x)


class Net(torch.nn.Module):
    # Our batch shape for input x is (3, 32, 32)

    def __init__(self):
        super(Net, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        #
        # self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # max pooling (kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

        # fully conected layers:
        self.fc6 = nn.Linear(8 * 8 * 128, 1024)
        # self.fc7 = nn.Linear(4096, 256)
        self.fc8 = nn.Linear(1024, 38)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        # x = F.relu(self.conv3_1(x))
        # x = F.relu(self.conv3_2(x))
        # x = F.relu(self.conv3_3(x))
        # x = self.pool(x)

        # x = F.relu(self.conv4_1(x))
        # x = F.relu(self.conv4_2(x))
        # x = F.relu(self.conv4_3(x))
        # x = self.pool(x)
        # x = F.relu(self.conv5_1(x))
        # x = F.relu(self.conv5_2(x))
        # x = F.relu(self.conv5_3(x))
        # x = self.pool(x)
        x = x.view(-1, 8 * 8 * 128)
        x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x
