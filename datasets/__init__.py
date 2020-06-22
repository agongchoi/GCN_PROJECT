from .cub200 import CUB200 as CUB200
from .cub200_split import CUB200 as CUB200Split
from .cub200_multisplit import CUB200 as CUB200MultiSplit
from .cub200_noisy import CUB200 as NoisyCUB200
from .cub200_nnnoisy import CUB200 as NNNoisyCUB200
from .cub200_intranoisy import CUB200 as IntraNoisyCUB200
from .CIFAR10_intranoisy import cifar as IntraNoisyCIFAR10
from .CIFAR100_intranoisy import cifar as IntraNoisyCIFAR100
from .CIFAR100_intranoisy2 import cifar as IntraNoisyCIFAR1002
from .CIFAR100_OpenUni import cifar as OpenUniCIFAR100
from .CIFAR100_split import cifar as SplitCIFAR100
from .CIFAR100_OpenNN import cifar as OpenNNCIFAR100
from .CIFAR100_ClosedNN import cifar as ClosedNNCIFAR100

__all__ = ('CUB200', 'CUB200Split', 'CUB200MultiSplit',
           'NoisyCUB200', 'NNNoisyCUB200', 'IntraNoisyCUB200',
           'IntraNoisyCIFAR10', 'IntraNoisyCIFAR100',
           'IntraNoisyCIFAR1002', 'OpenUniCIFAR100', 'SplitCIFAR100',
           'OpenNNCIFAR100', 'ClosedNNCIFAR100')
