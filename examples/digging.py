#!/usr/bin/env python

from dfl.dfl import *
from dfl.dynamic_system import *
from dfl.mpc import *
# from dfl.Transpose import Transpose

import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.autograd import Variable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from scipy import signal

m = 1.0
k11 = 0.2
k13 = 2.0
b1  = 3.0

class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        
        self.n_x = 6
        self.n_zeta = 6
        self.n_eta = 7
        self.n_u = 3

        self.n = self.n_x + self.n_eta

        # User defined matrices for DFL
        self.A_cont_x  = np.zeros((self.n_x,self.n_x))

        self.A_cont_eta = np.zeros((self.n_x,self.n_eta))

        self.B_cont_x = np.zeros((self.n_x,self.n_u))

        # Limits for inputs and states
        self.x_min = np.array([-2.0,-2.0])
        self.x_max = np.array([2.0 ,2.0])

        self.u_min = np.array([-2.5])
        self.u_max = np.array([ 2.5])

    # functions defining constituitive relations for this particular system
    # @staticmethod
    # def phi_c1(q):
    #     e = k11*q + k13*q**3
    #     return e

    # @staticmethod
    # def phi_r1(f):
    #     e = b1*np.sign(f)*f**2
    #     return e
    
    # nonlinear state equations
    def f(self,t,x,u):

        x_dot = np.zeros(x.shape)
        q,v = x[0],x[1]
        x_dot[0] = v
        x_dot[1] = -self.phi_r1(v) -self.phi_c1(q) + u 

        return x_dot

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        # q,v = x[0], x[1]
        # y = np.array([q,v])
        return np.copy(x)
    
    # @staticmethod
    # def gkoop1(t,x,u):
    #     q,v = x[0], x[1]
    #     y = np.array([q,v,Plant1.phi_c1(q), Plant1.phi_r1(v)])
    #     return y  
    
    # @staticmethod
    # def gkoop2(t,x,u):
    #     q,v = x[0],x[1]

    #     y = np.array([q,v,q**2,q**3,q**4,q**5,q**6,q**7,
    #                   v**2,v**3,v**4,v**5,v**6,v**7,v**9,v**11,v**13,v**15,v**17,v**19,
    #                   v*q,v*q**2,v*q**3,v*q**4,v*q**5,
    #                   v**2*q,v**2*q**2,v**2*q**3,v**2*q**4,v**2
    #                   *q**5,
    #                   v**3*q,v**3*q**2,v**3*q**3,v**3*q**4,v**3*q**5])
    #     return y 

    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        q,v = x[0],x[1]
        
        eta = np.zeros(self.n_eta)
        eta[0] = self.phi_c1(q)
        eta[1] = self.phi_r1(v)

        return eta

###########################################################################################

#Dummy forcing laws
def zero_u_func(y,t):
    return 1 

def rand_u_func(y,t):
    return np.random.normal(0.0,0.3)

def sin_u_func(y,t):
    return np.sin(3*t)

def square_u_func(y,t):
    return 0.5*signal.square(3 * t)

if __name__== "__main__":
    ################# DFL MODEL TEST ##############################################
    plant1 = Plant1()
    dt_data = 0.01
    dfl1 = DFL(plant1, dt_data = dt_data, dt_control = dt_data)
    driving_fun = square_u_func

    # dfl1.generate_data_from_random_trajectories( t_range_data = 5.0, n_traj_data = 100 )
    dfl1.generate_data_from_file('data_nick_not_flat.npz')
    dfl1.learn_eta_fn()
    # dfl1.generate_DFL_disc_model()
    dfl1.regress_K_matrix()

    driving_fun = lambda y,t : dfl1.u_data_test[round(t/dt_data)]
    x_0 = dfl1.x_data_test[0]
    x_0_aug = np.concatenate((x_0,dfl1.zetas_data_test[0]),0)

    # t, u_nonlin, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0, driving_fun, T)
    t = dfl1.t_data_test
    u_nonlin = dfl1.u_data_test
    x_nonlin = dfl1.x_data_test
    y_nonlin = dfl1.y_data_test
    T = dfl1.t_data_test[-1]
    # breakpoint()
    # t, u_dfl, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, driving_fun, T, continuous = False)
    t, u_lrn, x_lrn, y_lrn = dfl1.simulate_system_learned(x_0_aug, driving_fun, T)
    t, u_koop, x_koop, y_koop = dfl1.simulate_system_koop(x_0, driving_fun, T)

    sse_kop = np.sum((x_nonlin-x_koop[:,:6])**2)
    sse_lrn = np.sum((x_nonlin-x_lrn [:,:6])**2)
    print(sse_kop)
    print(sse_lrn)

    matplotlib.rcParams.update({'font.size': 18})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(t, y_nonlin[:,0], 'k', label='True')
    axs[0].plot(t, y_koop[:,0] ,'g-.', label='Koopman')
    # axs[0].plot(t, y_dfl[:,0] ,'r-.', label='DFL')
    axs[0].plot(t, y_lrn[:,0] ,'b-.', label='L3')
    axs[0].legend(ncol=3,bbox_to_anchor=(0, 1, 1, 0))
    axs[0].set_ylim(-5,5)
    # axs[0].tick_params(labelbottom = False, bottom = False)

    axs[1].plot(t, y_nonlin[:,1],'k')
    axs[1].plot(t, y_koop[:,1] ,'g-.')
    # axs[1].plot(t, y_dfl[:,1],'r-.')
    axs[1].plot(t, y_lrn[:,1],'b-.')
    axs[1].set_ylim(-2,0.5)
  
    axs[2].plot(y_nonlin[:,0], y_nonlin[:,1],'k')
    axs[2].plot(y_koop[:,0], y_koop[:,1],'g-.')
    axs[2].plot(y_lrn[:,0], y_lrn[:,1],'b-.')
    axs[2].set_xlim(-3.5,1.8)
    axs[2].set_ylim(-2,0.5)

    axs[1].set_xlabel('time')
    
    axs[0].set_ylabel('x')
    axs[1].set_ylabel('y')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')

    fig.subplots_adjust(hspace=0.5)

    plt.show()