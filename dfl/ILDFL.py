#!/usr/bin/env python
import torch, math
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

class ILDFL(torch.nn.Module):
    def __init__(self, D_x, D_zeta, D_eta, D_u, H):
        super(ILDFL, self).__init__()

        self.D_x = D_x
        D_xi = D_x + D_zeta + D_eta + D_u

        # eta = g(x)
        self.g = torch.nn.Sequential(
            torch.nn.Linear(D_x+D_zeta,H),
            torch.nn.ReLU(),
            torch.nn.Linear(H,H),
            torch.nn.ReLU(),
            torch.nn.Linear(H,D_eta),
            torch.nn.ReLU()
        )

        # Linear Dynamic Model
        self.A = torch.nn.Linear(D_xi,D_x   )
        self.Z = torch.nn.Linear(D_xi,D_zeta)
        self.H = torch.nn.Linear(D_xi,D_eta )

        self.g_u = torch.nn.Sequential(
            ILinear(D_u,D_u),
            IReLU(),
            ILinear(D_u,D_u),
            IReLU(),
            ILinear(D_u,D_u),
            IReLU()
        )

        self.D = torch.nn.Linear(D_u, D_zeta)

    def forward(self, x, zeta, u):
        nu        = self.g_u(u)
        zeta_star = zeta-self.D(nu)
        xs        = torch.cat((x,zeta_star), 1)
        eta       = self.g(xs)
        xi        = torch.cat((xs,eta,u), 1)

        x_tp1 = self.A(xi)
        zeta_star_tp1 = self.Z(xi)
        eta_tp1 = self.H(xi)

        return x_tp1, zeta_star_tp1, eta_tp1

    def g_u_inv(self, nu):
        z = nu
        for l in range(len(self.g_u)-1, -1, -1):
            z = self.g_u[l].inv(z)
        return z

class IReLU(torch.nn.Module):
    __constants__ = ['negative_slope', 'positive_slope']
    negative_slope: float
    positive_slope: float

    def __init__(self, negative_slope=math.tan(math.pi/8), positive_slope=math.tan(3*math.pi/8)):
        super(IReLU, self).__init__()

        self.negative_slope = negative_slope
        self.positive_slope = positive_slope

    def forward(self, x):
        return torch.clamp(x,min=0)*self.positive_slope + torch.clamp(x,max=0)*self.negative_slope

    def inv(self, y):
        return torch.clamp(y,min=0)/self.positive_slope + torch.clamp(y,max=0)/self.negative_slope

class ILinear(torch.nn.Linear):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(ILinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

    def inv(self, y: torch.Tensor) -> torch.Tensor:
        # breakpoint()
        return torch.matmul(y-self.bias, torch.pinverse(torch.transpose(self.weight,0,1)))