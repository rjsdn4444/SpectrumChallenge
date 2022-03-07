import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv_layer2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv_layer3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv_layer1(x)))
        out = self.bn2(self.conv_layer2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv_layer2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv_layer3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv_layer1(x)))
        out = F.relu(self.bn2(self.conv_layer2(out)))
        out = self.bn3(self.conv_layer3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels, num_max_unit_packets):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv_layer1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.actor_channel = nn.Linear(512 * 32, num_channels)
        self.actor_packet = nn.Linear(512 * 32, num_max_unit_packets)
        self.critic = nn.Linear(512 * 32, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv_layer1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        channel = self.actor_channel(out)
        packet = self.actor_packet(out)
        packet_probs = self.softmax(packet)
        # print(packet_probs)
        channel_probs = self.sigmoid(channel)
        value = self.critic(out)
        return channel_probs, packet_probs, value


def ResNet18(num_channels, num_max_unit_packets):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_channels=num_channels, num_max_unit_packets=num_max_unit_packets)


def ResNet34(num_channels, num_max_unit_packets):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_channels=num_channels, num_max_unit_packets=num_max_unit_packets)


def ResNet50(num_channels, num_max_unit_packets):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_channels=num_channels, num_max_unit_packets=num_max_unit_packets)


def ResNet101(num_channels, num_max_unit_packets):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_channels=num_channels, num_max_unit_packets=num_max_unit_packets)


def ResNet152(num_channels, num_max_unit_packets):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_channels=num_channels, num_max_unit_packets=num_max_unit_packets)

