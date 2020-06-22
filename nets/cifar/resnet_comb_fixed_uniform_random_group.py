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

    def __init__(self, depth, num_classes, num_partitionings, group_size=64, include_normal_head=False, use_bias=False):
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

        self.use_bias = use_bias
        self.include_normal_head = include_normal_head
        self.num_partitionings = num_partitionings
        self.num_classes = num_classes

        self.register_buffer('partitionings', torch.zeros(num_partitionings, num_classes, dtype=torch.long))
        self.register_buffer('group_map', torch.zeros(num_partitionings, group_size, dtype=torch.long))
        self.register_buffer('score_masks', torch.zeros(num_partitionings, num_classes, dtype=torch.uint8))
        self._sample_partitionings(torch.tensor(self._build_k_distribution()))
        self._sample_group_map(64 * block.expansion, num_partitionings, group_size)
        self._build_score_masks()

        ## Change fc into a 1d conv with group
        self.fc = nn.Conv1d(num_partitionings * group_size, num_partitionings * num_classes,
                            kernel_size=1, groups=num_partitionings)

        if self.include_normal_head:
            self.fc_normal = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _build_score_masks(self):
        for i, partitioning in enumerate(self.partitionings):
            for j in range(partitioning.size(0)):
                self.score_masks[i][j] = ((partitioning == j).long().sum() == 0)

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

    def _sample_partitionings(self, k_dist):
        num_partitionings = self.num_partitionings
        ks = torch.multinomial(k_dist, num_partitionings, True)
        for i, k in enumerate(ks):
            self.partitionings[i].random_(0, k+1)
            self.partitionings[i].add_(i * self.num_classes)

    def _sample_group_map(self, feat_size, num_partitionings, group_size):
        for i in range(num_partitionings):
            self.group_map[i] = torch.randperm(feat_size)[:group_size]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x_ = x.index_select(1, self.group_map.view(-1))

        outputs = self.fc(x_.unsqueeze(2)).view(-1, self.num_partitionings, self.num_classes)
        outputs.masked_fill_(self.score_masks.unsqueeze(0).expand_as(outputs), -1e10)
        outputs = F.log_softmax(outputs, dim=2)
        outputs = outputs.view(-1, self.num_partitionings * self.num_classes)
        outputs = outputs.index_select(1, self.partitionings.view(-1))
        outputs = outputs.view(-1, self.num_partitionings, self.num_classes)

        normal_head_output = None
        other_head_output = None

        output = outputs.sum(1)

        if self.include_normal_head:
            normal_head_output = F.log_softmax(self.fc_normal(x), dim=1)
            other_head_output = output

            output = normal_head_output + other_head_output

        return output, normal_head_output, other_head_output


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)
