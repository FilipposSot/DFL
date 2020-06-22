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

m = 1.0
k11 = 1.1

k13 = 1.4
k21 = 1.0
k23 = 1.2
b1  = 1.0
b2  = 1.5

class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        # Linear part of states matrices


        self.N_x = 3
        self.N_eta = 4
        self.N_u = 1
        self.N = self.N_x + self.N_eta

        self.A_x  = np.array([[0.0, 0.0, -1.0/m],
                              [0.0, 0.0,  1.0/m],
                              [0.0, 0.0,   0.0]])

        self.A_eta = np.array([[0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0],
                               [1.0, 1.0,-1.0,-1.0]])

        self.B_x = np.array([[1.0],[0.0],[0.0]])

        self.x_init_min = np.array([-1.0,-1.0,-1.0])
        self.x_init_max = np.array([1.0 ,1.0 ,1.0])

    # functions defining constituitive relations for this particular system
    def phi_c1(self,q):
        e = k11*q + k13*q**3
        return e

    def phi_c2(self,q):
        e = k21*q + k23*q**3
        return e

    def phi_r1(self,f):
        e = b1*np.sign(f)*np.abs(f)*np.abs(f)
        return e

    def phi_r2(self,f):
        e = b2*np.sign(f)*np.abs(f)*np.abs(f)
        return e

    # nonlinear state equations
    def f(self,t,x,u):

        x_dot = np.zeros(x.shape)
        q1,q2,p = x[0],x[1],x[2]
        x_dot[0] = u - (1.0/m)*p
        x_dot[1] = (1.0/m)*p
        x_dot[2] = self.phi_c1(q1) + self.phi_r1(u-p/m) - self.phi_c2(q2) - self.phi_r2(p/m)
        return x_dot

    # nonlinear observation equations
    def g(self,x,u,t):
        y = np.array([x[0],x[1],x[2]])
        return y 
    
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        q1,q2,p = x[0],x[1],x[2]
        eta = np.zeros(self.N_eta)
        eta[0] = self.phi_c1(q1)
        eta[1] = self.phi_r1(u - p/m)
        eta[2] = self.phi_c2(q2)
        eta[3] = self.phi_r2(p/m)
        return eta

    # def phi_dot(self,t,x,u):
    #     '''
    #     outputs the analytical (exact) values of the time derivatives 
    #     of the auxiliary variables
    #     '''
    #     q1,q2,p = x[0],x[1],x[2]
        
    #     q1_dot = u - (1.0/m)*p
    #     q2_dot = (1.0/m)*p
    #     p_dot = self.phi_c1(q1) + self.phi_r1(u-p/m) - self.phi_c2(q2) - self.phi_r2(p/m)
        

    #     eta_dot = np.zeros(self.N_eta)
    #     eta = np.zeros(self.N_eta)
        
    #     eta[0] = self.phi_c1(q1)
    #     eta[1] = self.phi_r1(u - p/m)
    #     eta[2] = self.phi_c2(q2)
    #     eta[3] = self.phi_r2(p/m)

    #     eta_dot[0] = (k11 + 3*k13*q1**2)*q1_dot
        



def zero_u_func(y,t):
    return 0 


if __name__== "__main__":
    plant = Plant1()
    dfl = DFL(plant)
    dfl.generate_data_from_random_trajectories()
    dfl.generate_H_matrix()

    print('Hx:')
    print(dfl.H_x)

    print('Heta:')
    print(dfl.H_eta)
    w, v = np.linalg.eig(dfl.H_eta)
    print(np.abs(w))
    
    print('Hu:')
    print(dfl.H_u)

    A_full = np.block([[plant.A_x,plant.A_eta],
					   [dfl.H_x,dfl.H_eta]])
    w, v = np.linalg.eig(plant.A_x)
    print(plant.A_x)

    x_0 = np.array([1.0,0.0,0.0])
    t_f = 5.0

    t, u, x_dfl = dfl.simulate_system_dfl(x_0, zero_u_func, t_f)
    t, u, x_nonlin = dfl.simulate_system_nonlinear(x_0, zero_u_func, t_f)
    
    fig, axs = plt.subplots(2, 1)
    
    axs[0].plot(t, x_dfl[:,0] ,'b')
    axs[0].plot(t, x_nonlin[:,0], 'b--')
    axs[0].plot(t, x_dfl[:,1],'r')
    axs[0].plot(t, x_nonlin[:,1],'r--')
    axs[0].plot(t, x_dfl[:,2],'g')
    axs[0].plot(t, x_nonlin[:,2],'g--')

    axs[0].set_xlim(0, t_f)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('states')
    axs[0].grid(True)

    axs[1].plot(t, u)
    axs[1].set_ylabel('input')

    fig.tight_layout()
    plt.show()
