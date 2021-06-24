#!/usr/bin/env python

import numpy as np

from dfl.dfl.dfl import *
from dfl.dfl.dynamic_system import *
from dfl.dfl.mpc import *

import matplotlib.pyplot as plt

# T_RANGE_DATA = 1.0
# DT_DATA = 0.05
# N_TRAJ_DATA = 20
# X_INIT_MIN = np.array([0.0,0.0,0.0])
# X_INIT_MAX = np.array([1.0,1.0,1.0])

mu = -1
lam = -10

class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        # Linear part of states matrices


        self.n_x = 2
        self.n_eta = 2
        self.n_u = 1
        self.N_y = 4

        self.n = self.n_x + self.n_eta

        # User defined matrices for DFL
        self.A_cont_x  = np.array([[mu, 0.0],
                                   [0.0, lam]])

        self.A_cont_eta = np.array([[0.0,0.0], [-lam,-lam]])

        self.B_cont_x = np.array([[0.0],[0.0]])

        # self.B_x = np.array([[0.0],[1.0]])

        self.x_min = np.array([-2.0,-2.0])
        self.x_max = np.array([2.0 ,2.0])

        self.u_min = np.array([-2.5])
        self.u_max = np.array([ 2.5])


    # # functions defining constituitive relations for this particular system
    # def phi_c1(self,q):
    #     e = k11*q + k13*q**3
    #     return e

    def P1(self,x):
        y = x**2
        return y
   
    def P2(self,x):
        y = x**3
        return y
    # nonlinear state equations
    def f(self,t,x,u):

        x_dot = np.zeros(x.shape)
        x1,x2 = x[0], x[1]
        x_dot[0] = mu*x1
        x_dot[1] = lam*(x2-self.P1(x1)-self.P2(x1)) 
        return x_dot

    # nonlinear observation equations
    def g(self,t,x,u):
        x1, x2 = x[0],x[1]
        y = np.array([x1,x2,x1**2,x1**3])
        return y 
    
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        q,v = x[0],x[1]
        eta = np.zeros(self.n_eta)
        # eta[0] = self.phi_c1(q) + self.phi_r1(v)
        # eta[1] = 0.0
        eta[0] = self.P1(q)
        eta[1] = self.P2(q)
        return eta

def zero_u_func(y,t):
    return 0 

def rand_u_func(y,t):
    return np.random.normal(0.0,0.3) 

def sin_u_func(y,t):
    return np.sin(3*t) 

if __name__== "__main__":
    plant = Plant1()
    dfl = DFL(plant)
    dfl.generate_data_from_random_trajectories()
    # dfl.generate_H_matrix()
    dfl.regress_K_matrix()

    # dfl.generate_disrete_time_system()


    # print('Hx:')
    # print(dfl.H_x)

    # print('Heta:')
    # print(dfl.H_eta)
    # # w, v = np.linalg.eig(dfl.H_eta)
    # # print(np.abs(w))
    
    # print('Hu:')
    # print(dfl.H_u)

    # A_full = np.block([[plant.A_x,plant.A_eta],
                #      [dfl.H_x,dfl.H_eta]])
    # w, v = np.linalg.eig(plant.A_x)
    # print(plant.A_x)
    
    t_f = 5.0
    # x_0 = np.array([1.0,1.0])
    error_koopman = np.array([0,0])
    # error_dfl = np.array([0,0])
    
    fig, axs = plt.subplots(3, 1)

    for i in range(100):

        x_0 = np.random.uniform(plant.x_min,plant.x_max)

        # t, u, x_dfl = dfl.simulate_system_dfl(x_0, zero_u_func, t_f)
        t, u, x_nonlin ,y_nonlin= dfl.simulate_system_nonlinear(x_0, sin_u_func, t_f)
        t, u, x_koop,y_koop = dfl.simulate_system_koop(x_0, sin_u_func, t_f)

        print(y_koop.shape)
        
        axs[0].plot(y_koop[:,0], y_koop[:,1], 'b')
        axs[1].plot(y_koop[:,1], y_koop[:,2], 'b')
        axs[2].plot(y_koop[:,2], y_koop[:,3], 'b')

        # error_koopman =+ np.mean(np.power(x_nonlin - x_koop[:,0:2],2),axis = 0)
        # error_dfl     =+ np.mean(np.power(x_nonlin -  x_dfl[:,0:2],2),axis = 0)


    # print('Koopman Error',error_koopman)
    # print('DFL Error', error_dfl)

    fig, axs = plt.subplots(2, 1)
    
    axs[0].plot(t, x_nonlin[:,0], 'b')
    # axs[0].plot(t, x_dfl[:,0] ,'b--')
    axs[0].plot(t, x_koop[:,0] ,'b.')

    axs[0].plot(t, x_nonlin[:,1],'r')
    # axs[0].plot(t, x_dfl[:,1],'r--')
    axs[0].plot(t, x_koop[:,1] ,'r.')

    axs[0].set_xlim(0, t_f)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('states')
    axs[0].grid(True)

    axs[1].plot(t, u)
    axs[1].set_ylabel('input')

    fig.tight_layout()
    plt.show()
