from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
from scipy.special import factorial


__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes, num_partitionings, use_bias=False):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+236_comb_fromZeroNoise'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.use_bias = use_bias
        self.num_partitionings = num_partitionings
        self.num_classes = num_classes

        self.register_buffer('k_dist', torch.tensor(self._build_k_distribution()))
        self.register_buffer('partitionings', torch.zeros(num_partitionings, num_classes, dtype=torch.long))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _build_k_distribution(self):
        k_range = np.arange(1, self.num_classes+1, dtype='longdouble')
        k_weights = np.power(k_range, self.num_classes) / factorial(k_range, exact=True)
        k_probs = k_weights / k_weights.sum()

        return k_probs.astype('float32')

    def _sample_partitionings(self):
        ks = torch.multinomial(self.k_dist, self.num_partitionings, True)
        for i, k in enumerate(ks):
            self.partitionings[i].random_(0, k+1)

    def _construct_partitioning_params(self, partitioning):
        weight = []
        bias = [] if self.use_bias else None

        zero_weight = self.fc.weight.new_zeros(1, self.fc.in_features)
        zero_bias = self.fc.bias.new_zeros(1,1)

        for i in range(self.num_classes):
            partition = (partitioning == i)
            if partition.long().sum().item() == 0:
                weight.append(zero_weight)
                if self.use_bias:
                    bias.append(zero_bias)
                continue
            idx = torch.nonzero(partition).view(-1)
            weight.append(self.fc.weight.index_select(0, idx).mean(0, keepdim=True))
            if self.use_bias:
                bias.append(self.fc.bias.index_select(0, idx).sum(0, keepdim=True))
            torch.cuda.synchronize()

        weight = torch.cat(weight, 0)
        if self.use_bias:
            bias = torch.cat(bias, 0)

        return weight, bias

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        self._sample_partitionings()

        output = x.new_zeros(x.size(0), self.fc.out_features)
        for partitioning in self.partitionings:
            weight, bias = self._construct_partitioning_params(partitioning)
            partitioning_output = F.log_softmax(F.linear(x, weight, bias), dim=1)
            output.add_(partitioning_output.index_select(1, partitioning))

        return output


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
