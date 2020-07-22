#!/usr/bin/env python

import numpy as np
from dfl.dfl.dfl import *
from dfl.dfl.dynamic_system import *
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

        self.x_min = np.array([-1.0,-1.0,-1.0,-1.0, 0.0])
        self.x_max = np.array([1.0 , 1.0, 1.0, 1.0, 0.0])
        self.u_min = np.array([0.5, -1.0])
        self.u_max = np.array([1.0, -0.5])

    def get_s(self,x):
        '''
        Returns the soil surface parameters at a particular x value
        '''
        s = 0.5*np.sin(x) + 0.5*x
        s_dash = 0.5*np.cos(x) + 0.5
        s_dash_dash = -0.5*np.sin(x)
        return s, s_dash, s_dash_dash

    def get_I(self,gamma,D):
        '''
        Calculates variable system inertia
        '''
        return np.diag([1,1])

    @staticmethod
    def Phi_soil(D, x, z, v_x, v_z):
        '''
        place hold soil force
        will be replaced by FEE
        '''
        F = (-D**2 + -D*np.sqrt(x) -10*D*v_x)*np.array([1.0,0.0])
        F = F + ( D + D**3 + -10*D*v_z)*np.array([0.0,1.0])

        # F = np.array([0,D+D**3]) + -5*D*np.array([0.0,1.0])*v_z # max(v_x,0)*(np.abs(np.array([v_x,v_z])*D)) +
        # F = F + 5*max(v_x,0)*-D*np.array([1.0,0.0]) - 0.5*D*np.array([1.0,0.0])

        return F

    # nonlinear state equations
    def f(self,t,xi,u):

        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        s, s_dash, s_dash_dash = self.get_s(x)
        D = s-z
        I = self.get_I(gamma,D)

        x_dot = v_x
        z_dot = v_z

        # print(D)
        F_soil = self.Phi_soil(D, x, z, v_x, v_z)
        F_noise =  np.random.normal(np.array([0.0,0.0]),np.array([0.1,0.1]))
        # noise_ratio_0 = np.abs(F_soil[0]/F_noise[0])
        # noise_ratio_1 = np.abs(F_soil[1]/F_noise[1])

        # if noise_ratio_0 > 10 and noise_ratio_1 > 10:
        #     F_soil = F_soil + F_noise

        # print('Noise/Signal',noise_ratio_0,noise_ratio_1)

        
        # if np.abs(F_noise[0]) > 0.1 and np.abs(F_noise[1]) > 0.1:
            # F_soil = F_soil + np.abs(F_noise[0])*F_noise
        # F_soil = F_soil + F_noise
        # np.random.uniform(np.array([-0.05,0.0]),np.array([0.0,0.05]))
        # print(F_soil.shape)
        # print(np.random.normal(np.array([0.0,0.0]),np.array([0.01,0.01])))

        v_dot = np.linalg.inv(I).dot(F_soil + u) 

        gamma_dot = v_x*D

        xi_dot = np.array([x_dot, z_dot, v_dot[0], v_dot[1], gamma_dot]) 

        return xi_dot

        # nonlinear observation equations
    def g(self,x,u,t):
        
        return x
   
    def gkoop1(self,t,x,u):
        x, z, v_x, v_z, gamma = x[0], x[1], x[2], x[3], x[4]
        
        s, s_dash, s_dash_dash = self.get_s(x)
        D = s-z
        F = self.Phi_soil(D,x, z,v_x,v_z)
        # y = np.array([x,z,v_x,v_z,gamma, F[0],F[1],s, s_dash, s_dash_dash])
        y = np.array([x,z,v_x,v_z,gamma, F[0],F[1]]) #, s_dash, s_dash_dash])

        return y  
    
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        return 0

def zero_u_func(y,t):
    u = np.array([0.0,0.0])
    if t < 2:
        u[0] = 1.0
        u[1] = -1.0
    if t >= 2:
        u[0] = 2.0
        u[1] = -1.0
    if t >= 2:
        u[0] = 1.0
        u[1] = -0.2
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

    x_0 = np.array([0.0,0.0,0.0,0.0,0.0])
    dfl.generate_data_from_random_trajectories(t_range_data = 5.0, n_traj_data = 100, x_0 = x_0, plot_sample = True)
    dfl.generate_K_matrix()

   
    x_0 = np.array([0.0,0.0,0.0,0.0,0.0])
    t_f = 5.0

    t, u, x_nonlin, y_nonlin= dfl.simulate_system_nonlinear(x_0, zero_u_func, t_f)
    t, u, x_koop1, y_koop = dfl.simulate_system_koop(x_0,zero_u_func, t_f)

    fig, axs = plt.subplots(4, 1)   
    
    axs[0].plot(t, x_nonlin[:,0],'b', t, x_nonlin[:,1],'r')
    axs[0].plot(t, x_koop1[:,0],'b--',  t, x_koop1[:,1],'r--')
    axs[0].set_xlim(0, t_f)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('position states')
    axs[0].grid(True)

    axs[1].plot(t, x_nonlin[:,2],'b', t, x_nonlin[:,3],'r')
    axs[1].plot(t, x_koop1[:,2],'b--',  t, x_koop1[:,3],'r--')
    axs[1].set_xlim(0, t_f)
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('velocity states')
    axs[1].grid(True)

    axs[2].plot(t, x_nonlin[:,4],'g')
    axs[2].plot(t, x_koop1[:,4],'g--')
    axs[2].set_xlim(0, t_f)
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('bucket filling')
    axs[2].grid(True)

    print(u.shape)
    axs[3].plot(t, u[:,0,0],'k')
    axs[3].plot(t, u[:,0,1],'k--')
    axs[3].set_xlim(0, t_f)
    axs[3].set_ylim(-3, 3)
    axs[3].set_xlabel('time')
    axs[3].set_ylabel('input')
    axs[3].grid(True)

    fig.tight_layout()
    plt.show()
