#!/usr/bin/env python
import torch, math
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

class ILDFL(torch.nn.Module):
    def __init__(self, D_x, D_eta, D_u, H):
        super(ILDFL, self).__init__()

        self.D_x = D_x
        D_xi = D_eta + D_u

        self.g = torch.nn.Sequential(
            ILinear(D_x,H),
            IReLU(),
            ILinear(H,H),
            IReLU(),
            ILinear(H,D_eta),
            IReLU()
        )

        self.H = torch.nn.Linear(D_xi,D_eta)

    def forward(self, x_star):
        x_tm1 = x_star[:,:self.D_x]
        u_tm1 = x_star[:,self.D_x:]

        eta_tm1 = self.g(x_tm1)

        xi_tm1 = torch.cat((eta_tm1,u_tm1), 1)

        eta_t = self.H(xi_tm1)

        return eta_t

    def inv(self, eta):
        z = eta
        for l in range(len(self.g)-1, -1, -1):
            z = self.g[l].inv(z)
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