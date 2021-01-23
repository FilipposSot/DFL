#!/usr/bin/env python
import sys

#import system id toolboxes
sys.path.insert(0, "/home/filippos/repositories/pyN4SID")
import ssid
from sippy import *
import control

from abc import ABC, abstractmethod 
import numpy as np 
from numpy.linalg import lstsq
from scipy.linalg import expm
from numpy.linalg import inv

from scipy.integrate import ode
from scipy.signal import savgol_filter
from scipy.signal import cont2discrete
from scipy.linalg import eigvals
import copy

import matplotlib.pyplot as plt

np.set_printoptions(precision = 4)
np.set_printoptions(suppress = True)

class DFLSoil():
    
    def __init__(self, dynamic_plant,
                       dt_data = 0.05,
                       dt_control = 0.05):
        
        self.plant = dynamic_plant
        self.dt_data = dt_data 
        self.dt_control = dt_control 
        self.n_s = 3

        self.H_disc_x   = np.zeros((self.plant.n_eta,self.plant.n_x))
        self.H_disc_eta = np.zeros((self.plant.n_eta,self.plant.n_eta)) 
        self.H_disc_s   = np.zeros((self.plant.n_eta,self.n_s)) 
        self.H_disc_u   = np.zeros((self.plant.n_eta,self.plant.n_u)) 
        self.A_disc_x   = np.zeros((self.plant.n_x,self.plant.n_x))
        self.A_disc_eta = np.zeros((self.plant.n_x,self.plant.n_eta))
        self.B_disc_x   = np.zeros((self.plant.n_x,self.plant.n_u))

    def regress_model(self):
        '''
        regress the H matrix for DFL model
        '''
        omega = np.concatenate((self.X_minus.reshape(-1, self.X_minus.shape[-1]),
                                self.Eta_minus.reshape(-1, self.Eta_minus.shape[-1]),
                                self.S_minus.reshape(-1, self.S_minus.shape[-1]),
                                self.U_minus.reshape(-1, self.U_minus.shape[-1])),axis=1).T
        
        Y = self.Eta_plus.reshape(-1, self.Eta_plus.shape[-1]).T

        H_disc = lstsq(omega.T,Y.T,rcond=None)[0].T
        
        self.H_disc_x   = H_disc[:,:self.plant.n_x]
        self.H_disc_eta = H_disc[:, self.plant.n_x                                : self.plant.n_x + self.plant.n_eta]
        self.H_disc_s   = H_disc[:, self.plant.n_x + self.plant.n_eta             : self.plant.n_x + self.plant.n_eta + self.n_s]
        self.H_disc_u   = H_disc[:, self.plant.n_x + self.plant.n_eta  + self.n_s : self.plant.n_x + self.plant.n_eta + self.n_s + self.plant.n_u]

        (self.A_disc_x, self.B_disc_x,_,_,_) = cont2discrete((self.plant.A_cont_x, self.plant.B_cont_x, 
                                                            np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                            self.dt_data)
        
        (_,self.A_disc_eta ,_,_,_)   = cont2discrete((self.plant.A_cont_x, self.plant.A_cont_eta, 
                                                    np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                    self.dt_data)

    def regress_model_no_surface(self):
        '''
        regress the H matrix for DFL model
        '''
        omega = np.concatenate((self.X_minus.reshape(-1, self.X_minus.shape[-1]),
                                self.Eta_minus.reshape(-1, self.Eta_minus.shape[-1]),
                                self.U_minus.reshape(-1, self.U_minus.shape[-1])),axis=1).T
        
        Y = self.Eta_plus.reshape(-1, self.Eta_plus.shape[-1]).T

        H_disc = lstsq(omega.T,Y.T,rcond=None)[0].T
        
        self.H_disc_x   = H_disc[:,:self.plant.n_x]
        self.H_disc_eta = H_disc[:, self.plant.n_x : self.plant.n_x + self.plant.n_eta]
        self.H_disc_u   = H_disc[:, self.plant.n_x + self.plant.n_eta: self.plant.n_x + self.plant.n_eta +  self.plant.n_u]

        (self.A_disc_x, self.B_disc_x,_,_,_) = cont2discrete((self.plant.A_cont_x, self.plant.B_cont_x, 
                                                            np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                            self.dt_data)
        
        (_,self.A_disc_eta ,_,_,_)   = cont2discrete((self.plant.A_cont_x, self.plant.A_cont_eta, 
                                                    np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                    self.dt_data)

    def linearize_soil_dynamics_no_surface(self, x_nom):
      
        A_lin =  np.block([[self.A_disc_x, self.A_disc_eta],
                           [self.H_disc_x, self.H_disc_eta]])

        B_lin = np.block([[self.B_disc_x],
                          [self.H_disc_u]])

        # constant bias term
        K_lin = np.concatenate((np.zeros(self.plant.n_x),np.zeros(self.plant.n_eta)))

        return A_lin, B_lin, K_lin


    def linearize_soil_dynamics(self, x_nom):

        s_nom, s_dash_nom, s_dash_dash_nom, s_dash_dash_dash_nom = self.plant.soil_surf_eval(x_nom[0])

        sigma_zero   =  np.array([s_nom, s_dash_nom, s_dash_dash_nom]) 
        sigma_zero_d =  np.array([s_dash_nom, s_dash_dash_nom, s_dash_dash_dash_nom])

        T = np.zeros((self.n_s,self.plant.n_x))
        T[:,0] = 1.0

        H_disc_x_surf = self.H_disc_x + self.H_disc_s.dot(np.diag(sigma_zero_d)).dot(T)

        
        A_lin =  np.block([[self.A_disc_x , self.A_disc_eta],
                           [H_disc_x_surf , self.H_disc_eta]])

        B_lin = np.block([[self.B_disc_x],
                          [self.H_disc_u]])

        # constant bias term
        K_lin_eta = self.H_disc_s.dot(sigma_zero) - self.H_disc_s.dot(sigma_zero_d)*x_nom[0]
        K_lin = np.concatenate((np.zeros(self.plant.n_x),K_lin_eta))

        return A_lin, B_lin, K_lin

    def f_cont_dfl(self,t,xi,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        x   = xi[:self.plant.n_x]
        eta = xi[self.plant.n_x:self.plant.n_x + self.plant.n_eta]
        
        x_dot   = np.dot(self.plant.A_cont_x,x) +  np.dot(self.plant.A_cont_eta, eta) + np.dot(self.plant.B_cont_x,u)
        eta_dot = np.dot(self.H_cont_x,x) +  np.dot(self.H_cont_eta, eta) + np.dot(self.H_cont_u,u)

        return np.concatenate((x_dot,eta_dot))

    def f_disc_dfl(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        A_lin =  np.block([[self.A_disc_x, self.A_disc_eta],
                           [self.H_disc_x, self.H_disc_eta]])

        B_lin = np.block([[self.B_disc_x],
                          [self.H_disc_u]])
        
        y_plus = np.dot(A_lin,x) + np.dot(B_lin, u)

        return y_plus
        
    def g_disc_hybrid(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])

        if len(u.shape) > 1:
            u = u.flatten()

        eta = np.dot(self.C_til, x[self.plant.n_x:]) + np.dot(self.D_til_1, x[:self.plant.n_x]) + np.dot(self.D_til_2, u)

        y = np.concatenate((x[:self.plant.n_x],eta))

        # print(y.shape)
        return y
    
    def simulate_system(self, x_0, u_minus, t_f, dt, u_func, dt_control, f_func, g_func, continuous = True):
        '''
        Simulate a system in continuous time
        Arguments:
        x_0: initial state
        u_func: control function
        t_f: final time

        '''
        # initial time and input
        t = 0.0

        # create numerical integration object
        if continuous:
            r = ode(f_func).set_integrator('vode', method = 'bdf', max_step = 0.001)
            r.set_initial_value(x_0,t)

        t_array = []
        x_array = []
        u_array = []
        y_array = []
        s_array = []

        # initial state and 
        x_t = copy.copy(x_0)
        y_t = g_func(t, x_t, u_minus)

        u_t = u_func(y_t, t)

        t_array.append(t)
        x_array.append(x_t)
        u_array.append([u_t])
        y_array.append(g_func(t,x_t,u_minus))
        s_array.append(self.plant.soil_surf_eval(x_t[0]))

        D_t = s_array[-1][0] - x_array[-1][1]

        t_control_last = 0
        
        #Simulate the system
        while t < t_f:
            
            if (D_t < 0.01) and (x_t[3] > 0.0):
                break

            if continuous:
                r.set_f_params(u_t).set_jac_params(u_t)
                x_t = r.integrate(r.t + dt)
            else:
                x_t = f_func(t, x_t, u_t)

            t = t + dt
            y_t = g_func(t, x_t, u_t)
            
            if t - t_control_last > dt_control:
                t_control_last = copy.copy(t) 
                u_t = u_func(g_func(t, x_t, u_t), t)

            t_array.append(t)
            x_array.append(x_t)
            u_array.append([u_t])
            y_array.append(y_t)
            s_array.append(self.plant.soil_surf_eval(x_t[0]))

            D_t = s_array[-1][0] - x_array[-1][1]
        # print(len(y_array))
        return np.array(t_array), np.array(u_array), np.array(x_array), np.array(y_array)

    def generate_data_from_random_trajectories(self, t_range_data = 10.0, n_traj_data = 50, x_0 = None, plot_sample = False):
        '''
        create random data to train DFL and other dynamic system models
        '''
        t_data = []
        x_data = []
        u_data = []
        eta_data = []
        eta_dot_data = []

        
        X_minus_data = []
        U_minus_data = []
        Y_minus_data = []
        Eta_minus_data = []
        S_minus_data = []

        X_plus_data  = []
        U_plus_data  = []
        Y_plus_data  = []
        Eta_plus_data  = []
        S_plus_data = []

        for i in range(n_traj_data):
            
            x_surf, y_surf = self.plant.generate_random_surface()
            self.plant.set_soil_surf(x_surf, y_surf)

            # define initial conitions and range of time
            t_0 = 0.0
            t_f = t_range_data

            if x_0 is None:
                x_0 = np.random.uniform(self.plant.x_min,self.plant.x_max)

            #initialize the ode integrator
            r = ode(self.plant.f).set_integrator('dopri5')
            r.set_initial_value(x_0,t_0)

            t_array = []
            x_array = []
            u_array = []
            eta_array = []
            s_array = []
            y_array = []

            t_control_last = -10000000
            u_t = np.zeros(self.plant.n_u)  

            D_t = 0

            #simulate the system
            while r.successful() and r.t < t_f:

                #define the control input. A random value is used for data generation
                r.set_f_params(u_t).set_jac_params(u_t)
                x_t = r.integrate(r.t + self.dt_data)             

                if r.t - t_control_last > self.dt_control:
                    
                    t_control_last = r.t 
                    
                    if D_t > 0.15:
                        u_t =  np.random.uniform(low = self.plant.u_min , high = self.plant.u_max + np.array([0.0,0.7]))
                    else:
                        u_t =  np.random.uniform(low = self.plant.u_min , high = self.plant.u_max)

                # these are the inherent variables if the system ie input and state
                t_array.append(r.t)
                x_array.append(x_t)
                u_array.append([u_t])

                # these describe additional observations such as auxiliary variables or measurements
                eta_array.append(self.plant.phi(r.t,x_t,u_t))
                s_array.append(self.plant.soil_surf_eval(x_t[0]))
                y_array.append(self.plant.g(r.t,x_t,u_t))

                D_t = s_array[-1][0] - x_array[-1][1]

            # eta_dot_array = np.gradient(np.array(eta_array),self.dt_data)[0]

            # eta_dot_array2 = savgol_filter(np.array(eta_array),
                                           # window_length = 5, polyorder = 3,
                                           # deriv = 1, axis=0)/self.dt_data

            if plot_sample and i == 0:
                fig, axs = plt.subplots(5, 1)
                axs[0].plot(np.array(t_array), np.array(x_array)[:,0], 'k')
                axs[1].plot(np.array(t_array), np.array(x_array)[:,1], 'k')
                axs[2].plot(np.array(t_array), np.array(x_array)[:,2], 'k')
                axs[3].plot(np.array(t_array), np.array(x_array)[:,3], 'k')
                axs[4].plot(np.array(t_array), np.array(u_array)[:,0], 'k')
                
                axs[4].set_xlabel('time')

                axs[0].set_ylabel('q1')
                axs[1].set_ylabel('q2')
                axs[2].set_ylabel('p1')
                axs[3].set_ylabel('p2')
                axs[4].set_ylabel('u')

                fig, axs = plt.subplots(3, 1)
                axs[0].plot(np.array(t_array), np.array(eta_array)[:,0], 'k')
                axs[1].plot(np.array(t_array), np.array(eta_array)[:,1], 'k')
                axs[2].plot(np.array(t_array), np.array(eta_array)[:,2], 'k')
                
                axs[2].set_xlabel('time')

                axs[0].set_ylabel('F_x')
                axs[1].set_ylabel('F_z')
                axs[2].set_ylabel('gamma_dot')

                fig, axs = plt.subplots(1, 1)
                axs.plot(x_surf, y_surf, 'k')
                axs.plot(np.array(x_array)[:,0],np.array(x_array)[:,1], 'b')
                axs.axis('equal')

                plt.show()

            Y_minus_data.append(y_array[:-1])
            U_minus_data.append(u_array[:-1])
            X_minus_data.append(x_array[:-1])
            Eta_minus_data.append(eta_array[:-1])
            S_minus_data.append(s_array[:-1])

            Y_plus_data.append(y_array[1:])
            U_plus_data.append(u_array[1:])
            X_plus_data.append(x_array[1:])
            Eta_plus_data.append(eta_array[1:])
            S_plus_data.append(s_array[1:])


            t_data.append(t_array)
            x_data.append(x_array)
            u_data.append(u_array)
            # eta_data.append(eta_array)
            # eta_dot_data.append(eta_dot_array2)
        
        self.t_data = np.array(t_data) 
        self.x_data = np.array(x_data)
        self.u_data = np.array(u_data)
        self.eta_data = np.array(eta_data)
        # self.eta_dot_data = np.array(eta_dot_data)

        self.Y_minus = np.array(Y_minus_data)
        self.U_minus = np.array(U_minus_data)
        self.X_minus = np.array(X_minus_data)
        self.Eta_minus = np.array(Eta_minus_data)
        self.S_minus = np.array(S_minus_data)

        self.Y_plus = np.array(Y_plus_data)
        self.U_plus = np.array(U_plus_data)
        self.X_plus = np.array(X_plus_data)
        self.Eta_plus = np.array(Eta_plus_data)
        self.S_plus = np.array(S_plus_data)

    def simulate_system_nonlinear(self, x_0, u_func, t_f):

        u_minus = np.zeros((self.plant.n_u,1))
        t,u,x,y = self.simulate_system(x_0, u_minus, t_f, self.dt_data,
                                        u_func, self.dt_control, self.plant.f, self.plant.g,
                                        continuous = True)
        
        return t, u, x, y
        
    def simulate_system_dfl(self, x_0, u_func, t_f, continuous = True):

        u_minus = np.zeros((self.plant.n_u,1))
        eta_0 = self.plant.phi(0.0, x_0, u_minus)
        xi_0 = np.concatenate((x_0,eta_0))
        
        if continuous == True:
            t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                            u_func, self.dt_control, self.f_cont_dfl, self.plant.g,
                                            continuous = True)
        else:
            t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                            u_func, self.dt_control, self.f_disc_dfl, self.plant.g,
                                            continuous = False)
            
        return t, u, xi, y 

    def simulate_system_koop(self, x_0, u_func, t_f):

        u_minus = np.zeros((self.plant.N_u,1))
        xi_0 = self.plant.g(0.0, x_0, u_minus)

        t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                        u_func,self.dt_control, self.f_koop, self.plant.g,
                                        continuous = False)
        return t, u, xi, y 