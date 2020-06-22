from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.mca import MSA_ST

from math import *
import numpy as np
import copy


class CombinatorialClassifier(nn.Module):
    partition_weight = None

    def __init__(self, num_classes, num_partitionings, num_partitions, feature_dim, additive=False, attention=False,
                 mode='softmax', combination='logit'):
        super(CombinatorialClassifier, self).__init__()
        self.classifiers = nn.Linear(feature_dim, num_partitions * num_partitionings)

        self.num_classes = num_classes
        self.num_partitionings = num_partitionings
        self.num_partitions = num_partitions
        self.attention = attention
        self.mode = mode
        self.combination = combination
        #self.layer_norm = nn.LayerNorm(num_classes, eps=1e-6, elementwise_affine=False)
        if self.attention:

            self.AtModule = nn.Sequential(
                nn.Linear(feature_dim, num_partitionings // 4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(num_partitionings // 4, num_partitionings, bias=False),
                #nn.Softmax()
            )
            print("attention module activated")
        #Adds a persistent buffer to the module.
        #This is typically used to register a buffer that should not to be considered a model parameter.
        #For example, BatchNorm’s running_mean is not a parameter, but is part of the persistent state.

        self.register_buffer('partitionings', -torch.ones(num_partitionings, num_classes).long())

        self.additive = additive
        print("mode : ", self.mode, 'combination : ', self.combination)

    def set_partitionings(self, partitionings_map):
        self.partitionings.copy_(torch.LongTensor(partitionings_map).t())
        arange = torch.arange(self.num_partitionings).view(-1, 1).type_as(self.partitionings)
        #arange를 더해준다.? -> 01110, 23332
        self.partitionings.add_(arange * self.num_partitions)

    def rescale_grad(self):
        for params in self.classifiers.parameters():
            if self.partition_weight is None:
                params.grad.mul_(self.num_partitionings)
            else:
                params.grad.mul_(self.partition_weight.sum())

    def forward(self, input, output_sum=True, return_meta_dist=False, with_feat=False):
        assert self.partitionings.sum() > 0, 'Partitionings is never given to the module.'

        if self.attention:
            weight = self.AtModule(input)
            weight = weight.view(-1, self.num_partitionings).unsqueeze(2)

            if self.mode == 'softmax':
                weight = F.softmax(weight, dim=1)
            else:
                weight = torch.sigmoid(weight)


            self.partition_weight = weight
            #weight = weight.view(1, -1, 1)
            #output = output * weight
            #self.partition_weight = weight

        all_output = self.classifiers(input)
        all_output = all_output.view(-1, self.num_partitionings, self.num_partitions)
        # all_output = F.log_softmax(all_output, dim=2)

        if self.attention:
            if self.combination == 'prob':
                all_output = F.softmax(all_output, dim=2)
                all_output = all_output * weight
                norm = torch.log_(all_output.sum(2, keepdim=True) + 1e-10)
                all_output = torch.log_(all_output + 1e-10)
                all_output = all_output - norm
            else:
                all_output = F.log_softmax(all_output, dim=2)
                all_output = all_output * weight
        else:
            all_output = F.log_softmax(all_output, dim=2)

        # if self.combination == 'prob':
        #     all_output = F.softmax(all_output, dim=2)
        #     if self.attention:
        #         all_output = all_output * weight
        #         norm = torch.log_(all_output.sum(2, keepdim=True) + 1e-10)
        #
        #     all_output = torch.log_(all_output + 1e-10)
        #     all_output = all_output - norm
        # else:
        #     all_output = F.log_softmax(all_output, dim=2)
        #     if self.attention:
        #         all_output = all_output * weight

        if return_meta_dist:
            return all_output
        all_output = all_output.view(-1, self.num_partitionings * self.num_partitions)
        output = all_output.index_select(1, self.partitionings.view(-1))
        output = output.view(-1, self.num_partitionings, self.num_classes)

        # if self.attention:
        #     output = output * weight


        if output_sum:
            output = output.sum(1)
            #norm = torch.log(torch.exp(output).sum(1, keepdim=True) + 1e-20)

            #output =  output - norm
            #output = F.normalize(output, p=2, dim=1, eps=1e-12)
            #output = self.layer_norm(output)

        if with_feat:
            return input, output
        return output



class CombinatorialClassifierWithSA(CombinatorialClassifier):

    def __init__(self, num_classes, num_partitionings, num_partitions, feature_dim, additive=False, attention=False,
                 mode='softmax', combination='logit'):
        super(CombinatorialClassifierWithSA, self).__init__(num_classes, num_partitionings, num_partitions, feature_dim, additive=additive, attention=attention,
                 mode=mode, combination=combination)
        #self.classifiers = nn.Linear(feature_dim, num_partitions)
        self.classifiers = torch.nn.ModuleList([nn.Linear(feature_dim, num_partitions) for i in range(num_partitionings)])
        if attention:
            print('attention module is ready')

    def rescale_grad(self):
        for params in self.classifiers.parameters():
            params.grad.mul_(self.num_partitionings)

    def forward(self, input, attention_weight=None, output_sum=True, return_meta_dist=False, with_feat=False):
        assert self.partitionings.sum() > 0, 'Partitionings is never given to the module.'

        if self.attention:
            weight = attention_weight
            #weight = weight.view(-1, self.num_partitionings).unsqueeze(2)

            #if self.mode == 'softmax':
            #    weight = F.softmax(weight, dim=1)


            self.partition_weight = weight
            #print(self.partition_weight.sum())

        #all_output = self.classifiers(input)
        all_output = [head(input[:, i, :]) for i, head in enumerate(self.classifiers)]
        all_output = torch.cat(all_output, dim=1)
        all_output = all_output.view(-1, self.num_partitionings, self.num_partitions)
        # all_output = F.log_softmax(all_output, dim=2)

        if self.attention:
            if self.combination == 'prob':
                all_output = F.softmax(all_output, dim=2)
                all_output = all_output * weight
                #norm = torch.log_(all_output.sum(2, keepdim=True) + 1e-10)
                all_output = torch.log_(all_output + 1e-10)
                #all_output = all_output - norm
                #all_output = torch.exp(all_output)
            else:
                all_output = F.log_softmax(all_output, dim=2)
                all_output = all_output * weight
        else:
            all_output = F.log_softmax(all_output, dim=2)


        if return_meta_dist:
            return all_output
        all_output = all_output.view(-1, self.num_partitionings * self.num_partitions)
        output = all_output.index_select(1, self.partitionings.view(-1))
        output = output.view(-1, self.num_partitionings, self.num_classes)

        # if self.attention:
        #     output = output * weight


        if output_sum:
            output = output.sum(1)
            #output = F.normalize(output, p=2, dim=1, eps=1e-12)
            #output = self.layer_norm(output)

        if with_feat:
            return input, output
        return output


class CombinatorialClassifierEnsemble(nn.Module):
    partition_weight = None

    def __init__(self, num_classes, num_partitionings, num_partitions, feature_dim, additive=False, attention=False):
        super(CombinatorialClassifierEnsemble, self).__init__()
        self.classifiers = nn.Linear(feature_dim, num_partitions * num_partitionings)
        self.AtModule = nn.Linear(feature_dim, num_partitionings)
        self.standard = nn.Linear(feature_dim, num_classes)
        self.num_classes = num_classes
        self.num_partitionings = num_partitionings
        self.num_partitions = num_partitions
        self.attention = attention
        if self.attention:
            print("attention module activated")
        #Adds a persistent buffer to the module.
        #This is typically used to register a buffer that should not to be considered a model parameter.
        #For example, BatchNorm’s running_mean is not a parameter, but is part of the persistent state.

        self.register_buffer('partitionings', -torch.ones(num_partitionings, num_classes).long())

        self.additive = additive

    def set_partitionings(self, partitionings_map):
        self.partitionings.copy_(torch.LongTensor(partitionings_map).t())
        arange = torch.arange(self.num_partitionings).view(-1, 1).type_as(self.partitionings)
        #arange를 더해준다.? -> 01110, 23332
        self.partitionings.add_(arange * self.num_partitions)

    def rescale_grad(self):
        for params in self.classifiers.parameters():
            if self.partition_weight is None:
                params.grad.mul_(self.num_partitionings)
            else:
                params.grad.mul_(self.partition_weight.sum())

    def forward(self, input, weight=None, output_sum=True, return_meta_dist=False, with_feat=False):
        assert self.partitionings.sum() > 0, 'Partitionings is never given to the module.'

        if self.attention:
            weight = self.AtModule(input)
            weight = weight.view(-1, self.num_partitionings, 1)
            weight = F.softmax(weight, dim=1)
            #weight = weight.view(1, -1, 1)
            #output = output * weight
            #self.partition_weight = weight

        standard_output = self.standard(input)
        all_output = self.classifiers(input)
        all_output = all_output.view(-1, self.num_partitionings, self.num_partitions)
        #all_output = F.log_softmax(all_output, dim=2)
        all_output = F.log_softmax(all_output, dim=2)
        all_output = all_output * weight
        #all_output = torch.log(all_output)

        if return_meta_dist:
            return all_output
        all_output = all_output.view(-1, self.num_partitionings * self.num_partitions)
        output = all_output.index_select(1, self.partitionings.view(-1))
        output = output.view(-1, self.num_partitionings, self.num_classes)


        if weight is not None:
            weight = weight.view(1, -1, 1)
            output = output * weight
            self.partition_weight = weight

        if output_sum:
            output = output.sum(1)

        if with_feat:
            return input, output
        return output


class EnsembleClassifier(nn.Module):
    def __init__(self, num_classes, num_ensembles, feature_dim, additive=False):
        super(EnsembleClassifier, self).__init__()
        self.classifiers = nn.Linear(feature_dim, num_classes * num_ensembles)
        self.num_classes = num_classes
        self.num_ensembles = num_ensembles

        self.additive = additive

    def rescale_grad(self):
        for params in self.classifiers.parameters():
            params.grad.mul_(self.num_ensembles)

    def forward(self, input, weight=None):
        all_output = self.classifiers(input)
        if self.additive and False:
            raise NotImplementedError

            all_output = all_output.view(-1, self.num_partitionings, self.num_partitions)
            all_output = F.softmax(all_output, dim=2).view(-1, self.num_partitionings * self.num_partitions)
            output = all_output.index_select(1, self.partitionings.view(-1))

            output = output.view(-1, self.num_partitionings, self.num_classes)
            _sum = output.sum(dim=2, keepdim=True)
            output /= _sum.detach()

            # output = all_output.index_select(1, self.partitionings.view(-1))
            # output = F.softmax(output.view(-1, self.num_partitionings, self.num_classes), dim=236_comb_fromZeroNoise)
            if weight is None:
                output = output.sum(1) / self.num_partitionings
            else:
                weight = weight.view(1, -1, 1)
                output = output * weight
                output = output.sum(1)

            output = torch.log(output)
        else:
            all_output = all_output.view(-1, self.num_ensembles, self.num_classes)
            output = F.log_softmax(all_output, dim=2).sum(1)

        return output

class CombinatorialClassifierSplit(nn.Module):
    partition_weight = None

    def __init__(self, num_classes, num_partitionings, num_partitions, feature_dim, additive=False, attention=False,
                 mode='softmax', combination='logit'):
        super(CombinatorialClassifierSplit, self).__init__()
        #self.classifiers = nn.Linear(feature_dim, num_partitions * num_partitionings)
        self.split = int(feature_dim / num_partitionings)
        self.classifiers = nn.ModuleList([nn.Linear(self.split, num_partitions) for i in range(num_partitionings)])
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_partitionings = num_partitionings
        self.num_partitions = num_partitions
        self.attention = attention
        self.mode = mode
        self.combination = combination

        #Adds a persistent buffer to the module.
        #This is typically used to register a buffer that should not to be considered a model parameter.
        #For example, BatchNorm’s running_mean is not a parameter, but is part of the persistent state.

        self.register_buffer('partitionings', -torch.ones(num_partitionings, num_classes).long())

        self.additive = additive
        print("mode : ", self.mode, 'combination : ', self.combination)

    def set_partitionings(self, partitionings_map):
        self.partitionings.copy_(torch.LongTensor(partitionings_map).t())
        arange = torch.arange(self.num_partitionings).view(-1, 1).type_as(self.partitionings)
        #arange를 더해준다.? -> 01110, 23332
        self.partitionings.add_(arange * self.num_partitions)

    def rescale_grad(self):
        for params in self.classifiers.parameters():
            if self.partition_weight is None:
                params.grad.mul_(self.num_partitionings)
            else:
                params.grad.mul_(self.partition_weight.sum())

    def forward(self, input, output_sum=True, return_meta_dist=False, with_feat=False):
        assert self.partitionings.sum() > 0, 'Partitionings is never given to the module.'


        #all_output = self.classifiers(input)
        all_output = []
        channel_idx = 0
        for linear in self.classifiers:
            output = linear(input[:, channel_idx:channel_idx+self.split])
            all_output.append(output)
            channel_idx += self.split
        all_output = torch.cat(all_output, dim=1 )
        all_output = all_output.view(-1, self.num_partitionings, self.num_partitions)
        # all_output = F.log_softmax(all_output, dim=2)


        all_output = F.log_softmax(all_output, dim=2)

        # if self.combination == 'prob':
        #     all_output = F.softmax(all_output, dim=2)
        #     if self.attention:
        #         all_output = all_output * weight
        #         norm = torch.log_(all_output.sum(2, keepdim=True) + 1e-10)
        #
        #     all_output = torch.log_(all_output + 1e-10)
        #     all_output = all_output - norm
        # else:
        #     all_output = F.log_softmax(all_output, dim=2)
        #     if self.attention:
        #         all_output = all_output * weight

        if return_meta_dist:
            return all_output
        all_output = all_output.view(-1, self.num_partitionings * self.num_partitions)
        output = all_output.index_select(1, self.partitionings.view(-1))
        output = output.view(-1, self.num_partitionings, self.num_classes)

        # if self.attention:
        #     output = output * weight


        if output_sum:
            output = output.sum(1)

        if with_feat:
            return input, output
        return output