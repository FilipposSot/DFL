#!/usr/bin/env python

import numpy as np
from dfl import *
from dynamic_system import *
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


        self.N_x = 2
        self.N_eta = 2
        self.N_u = 1
        self.N_y =4

        self.N = self.N_x + self.N_eta

        # self.A_x  = np.array([[0.0, 1.0],
        #                       [0.0, 0.0]])

        # self.A_eta = np.array([[0.0, 0.0],
        #                        [-1/m,-1/m]])

        # self.A_eta = np.array([[0.0, 0.0],
        #                        [-1/m,0.0]])

        # self.B_x = np.array([[0.0],[1.0]])

        self.x_init_min = np.array([-2.0,-2.0])
        self.x_init_max = np.array([2.0 ,2.0])




    # # functions defining constituitive relations for this particular system
    # def phi_c1(self,q):
    #     e = k11*q + k13*q**3
    #     return e

    def P(self,x):
        y = x**2
        return y

    # nonlinear state equations
    def f(self,t,x,u):

        x_dot = np.zeros(x.shape)
        x1,x2 = x[0], x[1]
        x_dot[0] = mu*x1
        x_dot[1] = lam*(x2-self.P(x1)) 
        return x_dot

    # nonlinear observation equations
    def g(self,t,x,u):
        x1, x2 = x[0],x[1]
        y = np.array([x1,x2,x1**2])
        return y 
    
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        q,v = x[0],x[1]
        eta = np.zeros(self.N_eta)
        # eta[0] = self.phi_c1(q) + self.phi_r1(v)
        # eta[1] = 0.0
        eta[0] = self.P(q)
        eta[1] = self.P(q)
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
    dfl.generate_K_matrix()

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

    for i in range(1):

        x_0 = np.random.uniform(plant.x_init_min,plant.x_init_max)

        # t, u, x_dfl = dfl.simulate_system_dfl(x_0, zero_u_func, t_f)
        t, u, x_nonlin = dfl.simulate_system_nonlinear(x_0, sin_u_func, t_f)
        t, u, x_koop = dfl.simulate_system_koop(x_0, sin_u_func, t_f)

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
