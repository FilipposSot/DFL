#!/usr/bin/env python

import numpy as np
from dfl.dfl.dfl_soil import *
from dfl.dfl.dynamic_system import *
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splrep, splev, splint


# T_RANGE_DATA = 1.0
# DT_DATA = 0.05
# N_TRAJ_DATA = 20
# X_INIT_MIN = np.array([0.0,0.0,0.0])
# X_INIT_MAX = np.array([1.0,1.0,1.0])

class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        # Linear part of states matrices

        self.N_x = 5
        self.N_eta = 3
        self.N_u = 2
        self.N = self.N_x + self.N_eta

        # User defined matrices for DFL
        self.A_cont_x  = np.array([[ 0., 0., 1., 0., 0.],
                                   [ 0., 0., 0., 1., 0.],
                                   [ 0., 0., 0., 0., 0.],
                                   [ 0., 0., 0., 0., 0.],
                                   [ 0., 0., 0., 0., 0.]])

        self.A_cont_eta = np.array([[ 0., 0., 0.],
                                    [ 0., 0., 0.],
                                    [ 1., 0., 0.],
                                    [ 0., 1., 0.],
                                    [ 0., 0., 1.]])

        self.B_cont_x = np.array([[0.0,0.0],
                                  [0.0,0.0],
                                  [1.0,0.0],
                                  [0.0,1.0],
                                  [0.0,0.0],])

        self.x_min = np.array([-1.0,-1.0,-1.0,-1.0, 0.0])
        self.x_max = np.array([ 1.0, 1.0, 1.0, 1.0, 0.0])
        self.u_min = np.array([0.5, -1.0])
        self.u_max = np.array([1.0, -0.5])

    def set_soil_surf(self, x, y):

        self.tck_sigma = splrep(x, y, s = 0)

    def soil_surf_eval(self, x):
        # Evaluate the spline soil surface and its derivatives
        
        surf     = splev(x, self.tck_sigma, der = 0 )
        surf_d   = splev(x, self.tck_sigma, der = 1 )
        surf_dd  = splev(x, self.tck_sigma, der = 2 )
        surf_ddd = splev(x, self.tck_sigma, der = 3 )

        return surf, surf_d, surf_dd, surf_ddd

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
        F = (-D**2 -10*D*v_x)*np.array([1.0,0.0])
        F = F + ( D + D**3 + -10*D*v_z)*np.array([0.0,1.0])

        # F = np.array([0,D+D**3]) + -5*D*np.array([0.0,1.0])*v_z # max(v_x,0)*(np.abs(np.array([v_x,v_z])*D)) +
        # F = F + 5*max(v_x,0)*-D*np.array([1.0,0.0]) - 0.5*D*np.array([1.0,0.0])

        return F

    # nonlinear state equations
    def f(self,t,xi,u):

        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        s, s_dash, s_dash_dash, s_dash_dash_dash = self.soil_surf_eval(x)
        D = s-z
        I = self.get_I(gamma,D)

        x_dot = v_x
        z_dot = v_z
        F_soil = self.Phi_soil(D, x, z, v_x, v_z)
        # F_noise =  np.random.normal(np.array([0.0,0.0]),np.array([0.01,0.01]))
        
        v_dot = np.linalg.inv(I).dot(F_soil + u) 

        gamma_dot = v_x*D

        xi_dot = np.array([x_dot, z_dot, v_dot[0], v_dot[1], gamma_dot]) 

        return xi_dot

        # nonlinear observation equations
    def g(self,x,u,t):
        
        return x

    # observation including soil surface shape
    def g_state_and_surface(self,t,xi,u):
        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        s, s_dash, s_dash_dash, s_dash_dash_dash = self.soil_surf_eval(x)
        D = s-z
        # F = self.Phi_soil(D,x, z,v_x,v_z)
        y = np.array([x, z, v_x, v_z, gamma])

        return y  
    
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,xi,u):
        '''
        outputs the values of the auxiliary variables
        '''
        eta = np.zeros(self.N_eta)
        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        s, s_dash, s_dash_dash, s_dash_dash_dash = self.soil_surf_eval(x)
        D = s - z
        F_soil = self.Phi_soil(D, x, z, v_x, v_z)
        
        eta[0] = F_soil[0]
        eta[1] = F_soil[1]
        eta[2] = v_x*D

        return eta

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
    x = np.array([0,0.25,0.5,0.75,1.])
    y = np.array([0,0.2 ,0.2,0.2 ,0])
    plant.set_soil_surf(x, y)
    dfl = DFLSoil(plant)
    
    setattr(plant, "g", plant.g_state_and_surface)
    
    x_0 = np.array([0.0,-0.2,0.0,0.0,0.0])
    dfl.generate_data_from_random_trajectories(t_range_data = 5.0, n_traj_data = 5, x_0 = x_0, plot_sample = True)
    dfl.regress_model()
    dfl.linearize_soil_dynamics(np.array([0.0,-0.2,0.0,0.0,0.0]))

    ##############################################################
    # plant = Plant1()
    # dfl = DFL(plant)

    # setattr(plant, "g", plant.g_state_and_surface)

    # x_0 = np.array([0.0,0.0,0.0,0.0,0.0])
    # dfl.generate_data_from_random_trajectories(t_range_data = 5.0, n_traj_data = 2, x_0 = x_0, plot_sample = True)
    # # dfl.generate_K_matrix()
    # print(dfl.Y_minus.shape)
   
    # x_0 = np.array([0.0,0.0,0.0,0.0,0.0])
    # t_f = 5.0

    # t, u, x_nonlin, y_nonlin= dfl.simulate_system_nonlinear(x_0, zero_u_func, t_f)
    # # t, u, x_koop1, y_koop = dfl.simulate_system_koop(x_0,zero_u_func, t_f)

    # fig, axs = plt.subplots(4, 1)   
    
    # axs[0].plot(t, x_nonlin[:,0],'b', t, x_nonlin[:,1],'r')
    # # axs[0].plot(t, x_koop1[:,0],'b--',  t, x_koop1[:,1],'r--')
    # axs[0].set_xlim(0, t_f)
    # axs[0].set_xlabel('time')
    # axs[0].set_ylabel('position states')
    # axs[0].grid(True)

    # axs[1].plot(t, x_nonlin[:,2],'b', t, x_nonlin[:,3],'r')
    # # axs[1].plot(t, x_koop1[:,2],'b--',  t, x_koop1[:,3],'r--')
    # axs[1].set_xlim(0, t_f)
    # axs[1].set_xlabel('time')
    # axs[1].set_ylabel('velocity states')
    # axs[1].grid(True)

    # axs[2].plot(t, x_nonlin[:,4],'g')
    # # axs[2].plot(t, x_koop1[:,4],'g--')
    # axs[2].set_xlim(0, t_f)
    # axs[2].set_xlabel('time')
    # axs[2].set_ylabel('bucket filling')
    # axs[2].grid(True)

    # axs[3].plot(t, u[:,0,0],'k')
    # axs[3].plot(t, u[:,0,1],'k--')
    # axs[3].set_xlim(0, t_f)
    # axs[3].set_ylim(-3, 3)
    # axs[3].set_xlabel('time')
    # axs[3].set_ylabel('input')
    # axs[3].grid(True)

    # fig.tight_layout()
    # plt.show()
