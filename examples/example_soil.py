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

class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        # Linear part of states matrices


        self.N_x = 3
        self.N_eta = 4
        self.N_u = 1
        self.N = self.N_x + self.N_eta

        self.x_init_min = np.array([-1.0,-1.0,-1.0,-1.0, 0.0])
        self.x_init_max = np.array([1.0 , 1.0, 1.0, 1.0, 0.0])
        self.u_min = np.array([0.5, -1.0])
        self.u_max = np.array([1.0, -0.5])

    def get_s(self,x):
        '''
        Returns the soil surface parameters at a particular x value
        '''
        return 0.0

    def get_I(self,gamma,D):
        '''
        Calculates variable system inertia
        '''
        return np.diag([1,1])

    @staticmethod
    def Phi_soil(D, v_x, v_z):
        '''
        place hold soil force
        will be replaced by FEE
        '''
        F =  np.array([0,D]) + -D*np.array([1.0,0.0])*v_x + -D*np.array([0.0,1.0])*v_z # max(v_x,0)*(np.abs(np.array([v_x,v_z])*D)) +
        F = F + 10*max(v_x,0)*-D
        return F

    # nonlinear state equations
    def f(self,t,xi,u):

        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        s = self.get_s(x)
        D = s-z
        I = self.get_I(gamma,D)

        x_dot = v_x
        z_dot = v_z

        # print(D)
        F_soil = self.Phi_soil(D,v_x,v_z)
        v_dot = np.linalg.inv(I).dot(F_soil + u)

        gamma_dot = v_x*D

        xi_dot = np.array([x_dot, z_dot, v_dot[0], v_dot[1], gamma_dot]) 

        return xi_dot

        # nonlinear observation equations
    def g(self,x,u,t):
        
        return x
   
    def gkoop1(self,t,x,u):
        x, z, v_x, v_z, gamma = x[0], x[1], x[2], x[3], x[4]
        
        s = self.get_s(x)
        D = s-z
        F = self.Phi_soil(D,v_x,v_z)
        y = np.array([x,z,v_x,v_z,gamma, F[0],F[1]])
        return y  
    
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        return 0

def zero_u_func(y,t):
    u = np.array([0.0,0.0])
    u[0] = 1.0
    u[1] = -1.0
    return u

def control_u_func(y,t):
    u = np.array([0.0,0.0])
    # print(u)
    u[0] = 1.0
    u[1] = 2*(-0.5-y[1])
    return u

if __name__== "__main__":
    plant = Plant1()
    dfl = DFL(plant)

    setattr(plant, "g", plant.gkoop1)

    dfl.generate_data_from_random_trajectories()
    dfl.generate_K_matrix()

   
    x_0 = np.array([0.0,0.0,0.0,0.0,0.0])
    t_f = 5.0

    t, u, x_nonlin, y_nonlin= dfl.simulate_system_nonlinear(x_0, zero_u_func, t_f)
    t, u, x_koop1, y_koop = dfl.simulate_system_koop(x_0,zero_u_func, t_f)

    fig, axs = plt.subplots(3, 1)   
    
    axs[0].plot(t, x_nonlin[:,0],'r', t, x_nonlin[:,1],'b')
    axs[0].plot(t, x_koop1[:,0],'r--',  t, x_koop1[:,1],'b--')
    axs[0].set_xlim(0, t_f)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('position states')
    axs[0].grid(True)

    axs[1].plot(t, x_nonlin[:,2],'r', t, x_nonlin[:,3],'b')
    axs[1].plot(t, x_koop1[:,2],'r--',  t, x_koop1[:,3],'b--')
    axs[1].set_xlim(0, t_f)
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('velocity states')
    axs[2].grid(True)

    axs[2].plot(t, x_nonlin[:,4])
    axs[2].set_xlim(0, t_f)
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('bucket filling')
    axs[2].grid(True)

    fig.tight_layout()
    plt.show()
