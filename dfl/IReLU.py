#!/usr/bin/env python
import torch, math
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

class IReLU(torch.nn.Module):
    __constants__ = ['negative_slope', 'positive_slope']
    negative_slope: float
    positive_slope: float

    def __init__(self, negative_slope=torch.tan(math.pi/8), positive_slope=torch.tan(3*math.pi/8)):
        super(IReLU, self).__init__()

        self.negative_slope = negative_slope
        self.positive_slope = positive_slope

    def forward(self, x):
        return max(0,x)*self.positive_slope + min(0,x)*self.negative_slope

    def inv(self, y):
        return max(0,x)/self.positive_slope + min(0,x)/self.negative_slope