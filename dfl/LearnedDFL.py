#!/usr/bin/env python

import torch

class LearnedDFL(torch.nn.Module):
    def __init__(self, D_x, D_z, D_e, D_u, H):
        super(LearnedDFL, self).__init__()

        D_xi = D_x + D_z + D_e + D_u

        self.g = torch.nn.Sequential(
            torch.nn.Linear(D_x+D_z,H),
            torch.nn.ReLU(),
            torch.nn.Linear(H,H),
            torch.nn.ReLU(),
            torch.nn.Linear(H,D_e),
            torch.nn.ReLU()
        )

        self.A = torch.nn.Linear(D_xi,D_x)
        self.Z = torch.nn.Linear(D_xi,D_z)
        self.H = torch.nn.Linear(D_xi,D_e)

    def forward(self, x, z, u):
        xs = torch.cat((x,z), 1)

        eta = self.g(xs)

        xi = torch.cat((xs,eta,u), 1)

        x_tp1    = self.A(xi)
        zeta_tp1 = self.Z(xi)
        eta_tp1  = self.H(xi)

        return x_tp1, zeta_tp1, eta_tp1