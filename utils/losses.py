import torch
import torch.nn as nn
import torch.nn.functional as F




class EntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits):
        p = F.softmax(logits, dim=1)
        elementwise_entropy = -p * F.log_softmax(logits, dim=1)
        if self.reduction == 'none':
            return elementwise_entropy

        sum_entropy = torch.sum(elementwise_entropy, dim=1)
        if self.reduction == 'sum':
            return sum_entropy

        return torch.mean(sum_entropy)
