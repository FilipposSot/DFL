#!/usr/bin/env python
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

class LearnedDFL(torch.nn.Module):
    def __init__(self, D_x, D_eta, D_u, H):
        super(LearnedDFL, self).__init__()

        self.D_x = D_x
        D_xi = D_x + D_eta + D_u

        self.g = torch.nn.Sequential(
            torch.nn.Linear(D_x,H),
            torch.nn.ReLU(),
            torch.nn.ReLU(),
            torch.nn.ReLU(),
            torch.nn.Linear(H,D_eta)
        )

        self.A = torch.nn.Linear(D_xi,D_x)
        self.H = torch.nn.Linear(D_xi,D_eta)

    def forward(self, x_star):
        x_tm1 = x_star[:,:self.D_x]
        u_tm1 = x_star[:,self.D_x:]

        eta_tm1 = self.g(x_tm1)

        xi_tm1 = torch.cat((x_tm1,eta_tm1,u_tm1), 1)

        x_t = self.A(xi_tm1)
        eta_t = self.H(xi_tm1)

        return x_t, eta_t