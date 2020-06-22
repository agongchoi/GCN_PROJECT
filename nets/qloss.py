from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLoss(nn.Module):
    def __init__(self, q=0.8, k=0.1):
        super(QLoss, self).__init__()
        self.q = q
        self.k = k

    def forward(self, logit, target):
        prob = F.softmax(logit, dim=1)
        target_prob = prob[range(len(target)), target]

        mask = (target_prob >= self.k)

        qloss = (1 - target_prob[mask].pow(self.q)) / self.q
        loss_of_k = (1 - pow(self.k, self.q)) / self.q

        inv_mask = 1 - mask.float()
        truncated_loss = (qloss.sum() + (loss_of_k * inv_mask.sum())) / logit.size(0)

        return truncated_loss


class QLossCombLogProb(nn.Module):
    def __init__(self, q=0.8, k=0.1):
        super(QLossCombLogProb, self).__init__()
        self.q = q
        self.k = k

    def forward(self, comb_logprob, target):
        target_logprob = comb_logprob[range(comb_logprob.size(0)), :, target]
        target_prob = torch.exp(target_logprob)

        mask = (target_prob >= self.k)

        qloss = (1 - target_prob[mask].pow(self.q)) / self.q
        loss_of_k = (1 - pow(self.k, self.q)) / self.q

        inv_mask = 1 - mask.float()
        truncated_loss = (qloss.sum() + (loss_of_k * inv_mask.sum())) / target_logprob.numel()

        return truncated_loss
