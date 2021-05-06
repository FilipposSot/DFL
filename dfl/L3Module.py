#!/usr/bin/env python

import torch, math

class LearnedDFL(torch.nn.Module):
    def __init__(self, D_x: int, D_z: int, D_e: int, D_u: int, H: int):
        super(LearnedDFL, self).__init__()

        # Store dimensions in instance variables
        self.D_x = D_x
        self.D_z = D_z
        self.D_e = D_e
        self.D_u = D_u

        # Compute dimension of data vector
        D_xi = D_x + D_z + D_e + D_u

        # N/N model to compute augmented state, eta = g((x,zeta))
        self.g = torch.nn.Sequential(
            torch.nn.Linear(D_x+D_z,H),
            torch.nn.ReLU(),
            torch.nn.Linear(H,H),
            torch.nn.ReLU(),
            torch.nn.Linear(H,D_e),
            torch.nn.ReLU()
        )

        # Linear dynamic model matrices
        self.A = torch.nn.Linear(D_xi, D_x, bias=False)
        self.Z = torch.nn.Linear(D_xi, D_z, bias=False)
        self.H = torch.nn.Linear(D_xi, D_e, bias=False)

        # Anticausal filter matrix, D
        self.D = torch.zeros(D_u, D_z, requires_grad=False)

    def forward(self, x: torch.Tensor, zeta: torch.Tensor, u: torch.Tensor):
        zeta-= torch.matmul(u,self.D)
        xs   = torch.cat((x,zeta), 1)
        eta  = self.g(xs)
        xi   = torch.cat((xs,eta,u), 1)

        x_tp1, zeta_tp1, eta_tp1 = self.ldm(xi)

        return x_tp1, zeta_tp1, eta_tp1

    def ldm(self, xi: torch.Tensor):
        return self.A(xi), self.Z(xi), self.H(xi)

    def regress_D_matrix(self, u: torch.Tensor, zeta: torch.Tensor):
        self.D = torch.lstsq(torch.transpose(torch.matmul(torch.transpose(zeta, 0,1), u), 0,1), torch.transpose(torch.matmul(torch.transpose(u, 0,1), u), 0,1)).solution

    def _filter_linear_module(self, M: torch.nn.Linear):
        # Extract matrix from linear module
        A = M.weight

        # Partition matrix into zeta and u components
        A_z = A[:, self.D_x:self.D_x+self.D_z]
        A_u = A[:, self.D_x+self.D_z+self.D_e:]

        # Add filter to u component
        A_u+= torch.matmul(A_z, torch.transpose(self.D, 0,1))

        # Reassemble and return
        A[:, self.D_x+self.D_z+self.D_e:] = A_u
        M.weight.data = A
        return M

    def filter_linear_model(self):
        self.A = self._filter_linear_module(self.A)
        self.Z = self._filter_linear_module(self.Z)
        self.H = self._filter_linear_module(self.H)

class ILDFL(LearnedDFL):
    def __init__(self, D_x: int, D_zeta: int, D_eta: int, D_u: int, H: int):
        super(ILDFL, self).__init__(D_x, D_z, D_e, D_u, H)

        self.g_u = torch.nn.Sequential(
            ILinear(D_u,D_u),
            IReLU(),
            ILinear(D_u,D_u),
            IReLU(),
            ILinear(D_u,D_u),
            IReLU()
        )

        self.D = torch.nn.Linear(D_u, D_z, bias=False)

    def forward(self, x: torch.Tensor, zeta: torch.Tensor, u: torch.Tensor):
        nu        = self.g_u(u)
        zeta_star = zeta-self.D(nu)
        xs        = torch.cat((x,zeta_star), 1)
        eta       = self.g(xs)
        xi        = torch.cat((xs,eta,u), 1)

        x_tp1, zeta_star_tp1, eta_tp1 = self.ldm(xi)

        return x_tp1, zeta_star_tp1, eta_tp1

    def g_u_inv(self, nu: torch.Tensor):
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
        return torch.matmul(y-self.bias, torch.pinverse(torch.transpose(self.weight,0,1)))