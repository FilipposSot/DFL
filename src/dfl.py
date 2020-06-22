#!/usr/bin/env python
import sys
sys.path.insert(0, "/home/filippos/repositories/pyN4SID")
import ssid

from abc import ABC, abstractmethod 
import numpy as np 
from numpy.linalg import lstsq
from scipy.linalg import expm
from numpy.linalg import inv

from scipy.integrate import ode
from scipy.signal import savgol_filter
import copy

import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

class DFL():
    
    def __init__(self,dynamic_plant,t_range_data = 1.0,
                 dt_data = 0.05,n_traj_data = 300):
        
        self.plant = dynamic_plant
        self.t_range_data = t_range_data
        self.dt_data = dt_data 
        self.n_traj_data = n_traj_data

    def generate_disrete_time_system(self):

        A_full = np.block([[self.plant.A_x, self.plant.A_eta],
                           [self.H_x,   self.H_eta]])

        B_full = np.block([[self.plant.B_x],
                           [self.H_u]])

        A_disc = expm(A_full*self.dt_data)

        B_disc = inv(A_full).dot(expm(A_full*self.dt_data) - np.eye(A_full.shape[0])).dot(B_full)

        print(A_disc)
        print(B_disc)

    def generate_data_from_random_trajectories(self):
        '''
        create random data to train DFL
        '''
        t_data = []
        x_data = []
        u_data = []
        eta_data = []
        eta_dot_data = []

        Y_minus_data = []
        Y_plus_data  = []
        U_minus_data = []

        for i in range(self.n_traj_data):
            
            # define initial conitions and range of time
            t_0 = 0.0
            t_f = self.t_range_data
            x_0 = np.random.uniform(self.plant.x_init_min,self.plant.x_init_max)

            #initialize the ode integrator
            r = ode(self.plant.f).set_integrator('vode', method='bdf')
            r.set_initial_value(x_0,t_0)

            t_array = []
            x_array = []
            u_array = []
            eta_array = []
            y_array = []

            #simulate the system
            while r.successful() and r.t < t_f:
                
                #define the control input. A random value is used for data generation
                u_t =  np.random.uniform(low = -.01, high = .01)
                r.set_f_params(u_t).set_jac_params(u_t)
                x_t = r.integrate(r.t+self.dt_data)
                
                #these are the inherent variables if the system ie input and state
                t_array.append(r.t)
                x_array.append(x_t)
                u_array.append([u_t])

                #these describe additional observations such as auxiliary variables or measurements
                eta_array.append(self.plant.phi(r.t,x_t,u_t))
                y_array.append(self.plant.g(r.t,x_t,u_t))


            eta_dot_array = np.gradient(np.array(eta_array),self.dt_data)[0]

            eta_dot_array2 = savgol_filter(np.array(eta_array),
                                           window_length = 5, polyorder = 3,
                                           deriv = 1, axis=0)/self.dt_data

            Y_minus_data.append(y_array[:-1])
            U_minus_data.append(u_array[:-1])
            Y_plus_data.append(y_array[1:])

            t_data.append(t_array)
            x_data.append(x_array)
            u_data.append(u_array)
            eta_data.append(eta_array)
            eta_dot_data.append(eta_dot_array)
        
        self.t_data = np.array(t_data) 
        self.x_data = np.array(x_data)
        self.u_data = np.array(u_data)
        self.eta_data = np.array(eta_data)
        self.eta_dot_data = np.array(eta_dot_data)

        self.Y_minus = np.array(Y_minus_data)
        self.Y_plus = np.array(Y_plus_data)
        self.U_minus = np.array(U_minus_data)

    def generate_H_matrix(self,indices = None):
        '''
        regress the H matrix
        '''
        if indices is None:

            Xi = np.concatenate((self.x_data.reshape(-1, self.x_data.shape[-1]),
                                 self.eta_data.reshape(-1, self.eta_data.shape[-1]),
                                 self.u_data.reshape(-1, self.u_data.shape[-1])),axis=1)
            
            Eta_dot = self.eta_dot_data.reshape(-1, self.eta_dot_data.shape[-1])

        self.H = lstsq(Xi,Eta_dot,rcond=None)[0].T
        self.H_x   = self.H[:,:self.plant.N_x]
        self.H_eta = self.H[:,self.plant.N_x:self.plant.N_x+self.plant.N_eta]
        self.H_u   = self.H[:,self.plant.N_x+self.plant.N_eta:self.plant.N_x+self.plant.N_eta+self.plant.N_u]

    def generate_K_matrix(self):

        omega = np.concatenate((self.Y_minus.reshape(-1, self.Y_minus.shape[-1]),
                                self.U_minus.reshape(-1, self.U_minus.shape[-1])),axis=1).T
        
        Y = self.Y_plus.reshape(-1, self.Y_plus.shape[-1]).T

        # U_til, s_til, V_til_conj = np.linalg.svd(omega,full_matrices=False)
        # V_til = V_til_conj.T.conj()

        # U_til1 = U_til[:self.Y_plus.shape[-1], :]
        # U_til2 = U_til[self.Y_plus.shape[-1]:, :]

        # U_hat, s_hat, V_hat_conj = np.linalg.svd(Y,full_matrices=False)
        # V_hat = V_hat_conj.T.conj()

        # # _basis = Ur

        # _Atilde = U_hat.T.conj().dot(Y).dot(V_til).dot(np.diag(np.reciprocal(s_til))).dot(U_til1.T.conj()).dot(U_hat)
        # _Btilde = U_hat.T.conj().dot(Y).dot(V_til).dot(np.diag(np.reciprocal(s_til))).dot(U_til2.T.conj())
        
        # self.A_koop = _Atilde 
        # self.B_koop = _Btilde 
        # print(self.A_koop)
        
        G = lstsq(omega.T,Y.T,rcond=None)[0].T
        
        self.A_koop = G[: , :self.Y_plus.shape[-1]] 
        self.B_koop = G[: , self.Y_plus.shape[-1]:]
        # print(self.A_koop)

        # print(np.linalg.eig(_Atilde))

        # print(_Atilde)
        # print(_Btilde)

        # print(_B)

    def generate_N4SID_model(self):

        U = self.U_minus.reshape(-1, self.U_minus.shape[-1]).T
        Y = self.Y_plus.reshape(-1, self.Y_plus.shape[-1]).T
        
        NumURows = 10
        NumUCols = 5600
        
        print('u',U.shape)
        print('y',Y.shape)
        print('NumURows',NumURows)
        print('NumUCols',NumUCols)



        AID,BID,CID,DID,CovID,S = ssid.N4SID(U,Y,NumURows,NumUCols,4)
        print(AID,BID,CID,DID)
        plt.plot(S / S.sum())
        plt.show()

    def simulate_system_nonlinear(self, x_0, u_func, t_f):
        '''
        Simulate the full nonlinear system
        Arguments:
        x_0: initial state
        u_func: control function
        t_f: final time

        '''
        t_0 = 0.0

        #initialize the ode integrator
        r = ode(self.plant.f).set_integrator('vode', method='bdf')
        r.set_initial_value(x_0,t_0)

        t_array = []
        x_array = []
        u_array = []
        eta_array = []

        # initial state and 
        x_t = copy.copy(x_0)
        u_t = u_func(self.plant.g(r.t,x_0,0.0),r.t)

        #simulate the system
        while r.successful() and r.t < t_f:
            
            x_t_minus = copy.copy(x_t)
            u_t_minus = copy.copy(u_t)

            #define the control input. A random value is used for data generation
            u_t = u_func(self.plant.g(r.t,x_t_minus,u_t_minus),r.t)

            r.set_f_params(u_t).set_jac_params(u_t)
            x_t = r.integrate(r.t+self.dt_data)
            
            t_array.append(r.t)
            x_array.append(x_t)
            u_array.append([u_t])

            # eta_array.append(self.plant.phi(r.t,x_t,u_t))

        # eta_dot_array = savgol_filter(np.array(eta_array),
        #                               window_length = 5, polyorder = 3,
        #                               deriv = 1, axis=0)

        return np.array(t_array), np.array(u_array), np.array(x_array)
    
    def f_dfl(self,t,xi,u):

        u = np.array([u])
        x   = xi[:self.plant.N_x]
        eta = xi[self.plant.N_x:self.plant.N_x + self.plant.N_eta]
        x_dot   = np.dot(self.plant.A_x,x) +  np.dot(self.plant.A_eta, eta) + np.dot(self.plant.B_x,u) # temporary fix.
        eta_dot = np.dot(self.H_x,x) +  np.dot(self.H_eta, eta) + np.dot(self.H_u,u)

        return np.concatenate((x_dot,eta_dot))

    def f_koop(self,t,x,u):
        
        u = np.array([u])
        y_dot = np.dot(self.A_koop,x) + np.dot(self.B_koop,u)

        return y_dot

    def simulate_system_dfl(self, x_0, u_func, t_f):
        '''
        Simulate the full nonlinear system
        Arguments:
        x_0: initial state
        u_func: control function
        t_f: final time

        '''
        t_0 = 0.0
        # initial state and 
        u_0 = u_func(self.plant.g(t_0,x_0,0.0),t_0)

        #initialize the ode integrator
        eta_0 = self.plant.phi(t_0,x_0,u_0)
        xi_0 = np.concatenate((x_0,eta_0))

        r = ode(self.f_dfl).set_integrator('vode', method='bdf')
        r.set_initial_value(xi_0,t_0)

        # print(dir(r))
        # print(r.y)
        t_array = []
        xi_array = []
        u_array = []
        eta_array = []
        
        # initial state and 
        x_t = copy.copy(x_0)
        u_t = 0.0
        xi_t = copy.copy(xi_0)
        #simulate the system
        while r.successful() and r.t < t_f:
            
            x_t_minus = copy.copy(xi_t[:self.plant.N_x])
            u_t_minus = copy.copy(u_t)

            #define the control input. A random value is used for data generation
            u_t = u_func(self.plant.g(r.t,x_t_minus,u_t_minus),r.t)

            r.set_f_params(u_t).set_jac_params(u_t)

            xi_t = r.integrate(r.t+self.dt_data)

            t_array.append(r.t)
            xi_array.append(xi_t)
            u_array.append([u_t])

            # eta_array.append(self.plant.phi(r.t,x_t,u_t))

        # eta_dot_array = savgol_filter(np.array(eta_array),
        #                               window_length = 5, polyorder = 3,
        #                               deriv = 1, axis=0)

        return np.array(t_array), np.array(u_array), np.array(xi_array)

    def simulate_system_koop(self, x_0, u_func, t_f):
        '''
        Simulate the full nonlinear system
        Arguments:
        x_0: initial state
        u_func: control function
        t_f: final time

        '''
        t_0 = 0.0
        # initial state and 
        u_0 = u_func(self.plant.g(t_0,x_0,0.0),t_0)

        #initialize the ode integrator
        xi_0 = self.plant.g(t_0,x_0,u_0)

        t_array = []
        xi_array = []
        u_array = []
        eta_array = []
        
        # initial state and 
        x_t = copy.copy(x_0)
        u_t = 0.0
        xi_t = copy.copy(xi_0)
        #simulate the system
        t=0.0

        while True and t < t_f:
            
            t = t + self.dt_data
            xi_t_minus = copy.copy(xi_t)
            u_t_minus = copy.copy(u_t)

            #define the control input. A random value is used for data generation
            u_t = u_func(self.plant.g(t,xi_t_minus[:self.plant.N_x],u_t_minus),t)

            xi_t = self.f_koop(t, xi_t_minus, u_t)
            t_array.append(t)
            xi_array.append(xi_t)
            u_array.append([u_t])

        return np.array(t_array), np.array(u_array), np.array(xi_array)