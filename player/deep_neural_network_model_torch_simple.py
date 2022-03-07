from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)     # He initialization
        torch.nn.init.zeros_(m.bias)

    elif type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class DnnModel(nn.Module):
    def __init__(self, h, w, num_actions):
        super(DnnModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(4, 1), stride=(1, 1), padding=(2, 0))
        init_weights(self.conv1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 1), stride=None)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(1, 1), padding=(2, 0))
        init_weights(self.conv2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 1), stride=None)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(4, 1), stride=(1, 1), padding=(2, 0))
        init_weights(self.conv3)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((2, 1), stride=None)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(4, 1), stride=(1, 1), padding=(2, 0))
        init_weights(self.conv4)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d((1, 1), stride=None)

        self._flatten = nn.Flatten()

        def conv2d_size_out(size: tuple, padding: tuple, stride: tuple, kernel_size=(4, 1)):
            h = (size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) // stride[0] + 1
            w = (size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) // stride[1] + 1
            size = (h, w)
            return size

        def maxpool2d_size_out(size: tuple, pooling_size: tuple):
            h = size[0] // pooling_size[0]
            w = size[1] // pooling_size[1]
            size = (h, w)
            return size

        input = (h, w)
        cal_h_w = maxpool2d_size_out(conv2d_size_out(input, padding=(2, 0), stride=(1, 1)), pooling_size=(2, 1))
        cal_h_w = maxpool2d_size_out(conv2d_size_out(cal_h_w, padding=(2, 0), stride=(1, 1)), pooling_size=(2, 1))
        cal_h_w = maxpool2d_size_out(conv2d_size_out(cal_h_w, padding=(2, 0), stride=(1, 1)), pooling_size=(2, 1))
        cal_h_w = maxpool2d_size_out(conv2d_size_out(cal_h_w, padding=(2, 0), stride=(1, 1)), pooling_size=(1, 1))

        linear_input_size = cal_h_w[0] * cal_h_w[1] * 64
        self.actor = nn.Linear(linear_input_size, num_actions)
        self.critic = nn.Linear(linear_input_size, 1)
        init_weights(self.critic)
        init_weights(self.actor)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value
