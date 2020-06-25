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
from scipy.signal import cont2discrete
from scipy.linalg import eigvals
import copy

import matplotlib.pyplot as plt

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

class DFL():
    
    def __init__(self,dynamic_plant, t_range_data = 5.0,
                 dt_data = 0.05, n_traj_data = 100):
        
        self.plant = dynamic_plant
        self.t_range_data = t_range_data
        self.dt_data = dt_data 
        self.n_traj_data = n_traj_data

    def generate_disrete_time_DFL_system(self):

        A_full = np.block([[self.plant.A_x, self.plant.A_eta],
                           [self.H_x,   self.H_eta]])

        B_full = np.block([[self.plant.B_x],
                           [self.H_u]])

        A_disc = expm(A_full*self.dt_data)

        print(A_disc)

        B_disc = inv(A_full).dot(expm(A_full*self.dt_data) - np.eye(A_full.shape[0])).dot(B_full)

    def generate_data_from_random_trajectories(self):
        '''
        create random data to train DFL
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

        X_plus_data  = []
        U_plus_data  = []
        Y_plus_data  = []
        Eta_plus_data  = []

        for i in range(self.n_traj_data):
            
            # define initial conitions and range of time
            t_0 = 0.0
            t_f = self.t_range_data
            x_0 = np.random.uniform(self.plant.x_init_min,self.plant.x_init_max)

            #initialize the ode integrator
            r = ode(self.plant.f).set_integrator('dopri5')
            r.set_initial_value(x_0,t_0)

            t_array = []
            x_array = []
            u_array = []
            eta_array = []
            y_array = []

            t_control_last = -100000
            u_t =  0.0

            #simulate the system
            while r.successful() and r.t < t_f:
                
                #define the control input. A random value is used for data generation
                r.set_f_params(u_t).set_jac_params(u_t)
                x_t = r.integrate(r.t+self.dt_data)             

                if r.t - t_control_last > 1.0:
                    t_control_last = r.t 
                    u_t =  np.random.uniform(low = -4.0, high = 4.0)

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

            # fig, axs = plt.subplots(3, 1)
            # axs[0].plot(np.array(t_array), np.array(x_array)[:,0], 'b')
            # axs[1].plot(np.array(t_array), np.array(x_array)[:,1], 'r')
            # axs[2].plot(np.array(t_array), np.array(u_array), 'g')

            plt.show()

            Y_minus_data.append(y_array[:-1])
            U_minus_data.append(u_array[:-1])
            X_minus_data.append(x_array[:-1])
            Eta_minus_data.append(eta_array[:-1])

            Y_plus_data.append(y_array[1:])
            U_plus_data.append(u_array[1:])
            X_plus_data.append(x_array[1:])
            Eta_plus_data.append(eta_array[1:])


            t_data.append(t_array)
            x_data.append(x_array)
            u_data.append(u_array)
            eta_data.append(eta_array)
            eta_dot_data.append(eta_dot_array2)
        
        self.t_data = np.array(t_data) 
        self.x_data = np.array(x_data)
        self.u_data = np.array(u_data)
        self.eta_data = np.array(eta_data)
        self.eta_dot_data = np.array(eta_dot_data)

        self.Y_minus = np.array(Y_minus_data)
        self.U_minus = np.array(U_minus_data)
        self.X_minus = np.array(X_minus_data)
        self.Eta_minus = np.array(Eta_minus_data)

        self.Y_plus = np.array(Y_plus_data)
        self.U_plus = np.array(U_plus_data)
        self.X_plus = np.array(X_plus_data)
        self.Eta_plus = np.array(Eta_plus_data)

    def generate_H_matrix(self):
        '''
        regress the H matrix for DFL model
        '''

        Xi = np.concatenate((self.x_data.reshape(-1, self.x_data.shape[-1]),
                             self.eta_data.reshape(-1, self.eta_data.shape[-1]),
                             self.u_data.reshape(-1, self.u_data.shape[-1])),axis=1)
        
        Eta_dot = self.eta_dot_data.reshape(-1, self.eta_dot_data.shape[-1])

        self.H = lstsq(np.array(Xi,dtype='float'),np.array(Eta_dot,dtype='float'),rcond=None)[0].T
        self.H_x   = self.H[:,:self.plant.N_x]
        self.H_eta = self.H[:,self.plant.N_x:self.plant.N_x+self.plant.N_eta]
        self.H_u   = self.H[:,self.plant.N_x+self.plant.N_eta:self.plant.N_x+self.plant.N_eta+self.plant.N_u]


    def generate_K_matrix(self):

        omega = np.concatenate((self.Y_minus.reshape(-1, self.Y_minus.shape[-1]),
                                self.U_minus.reshape(-1, self.U_minus.shape[-1])),axis=1).T
        
        Y = self.Y_plus.reshape(-1, self.Y_plus.shape[-1]).T

        G = lstsq(omega.T,Y.T,rcond=None)[0].T
        
        self.A_koop = G[: , :self.Y_plus.shape[-1]] 
        self.B_koop = G[: , self.Y_plus.shape[-1]:]

        print(self.B_koop)

    def generate_hybrid_model(self):


        U = np.concatenate((self.X_minus.reshape(-1, self.X_minus.shape[-1]),
                            self.U_minus.reshape(-1, self.U_minus.shape[-1])),axis=1).T

        Y_temp = self.Eta_minus.reshape(-1, self.Eta_minus.shape[-1]).T

        Y = Y_temp[0,:] + Y_temp[1,:]
        Y = Y.T
        Y = np.expand_dims(Y, axis=0)


        # (_,A_eta_hybrid_disc,_,_,_) = cont2discrete((self.plant.A_x,self.plant.A_eta_hybrid,np.array([0.0,0.0]),np.array([0.0])),self.dt_data)
        
        (A_x_disc,_,_,_,_) = cont2discrete((self.plant.A_x,self.plant.B_x,np.array([0.0,0.0]),np.array([0.0])),self.dt_data)
        B_x_disc = self.plant.B_x*self.dt_data
        A_eta_hybrid_disc = self.plant.A_eta_hybrid*self.dt_data
        
        print('discrete matrices')
        print(A_x_disc)
        print(B_x_disc)
        print(A_eta_hybrid_disc)
       
        NumURows = 200
        NumUCols = 1500
        # print('N4SID inputs')
        # print(U.shape)
        # print(Y.shape)

        A_til,B_til,C_til,D_til,_,S = ssid.N4SID(U,Y,NumURows,NumUCols,2)
        
        # plt.plot(S / S.sum())
        # plt.show()
        print('n4sid matrices')
        print(A_til)
        print(B_til)
        print(C_til)
        print(D_til)


        B_til_1 = B_til[:,:self.plant.N_x]
        B_til_2 = B_til[:,self.plant.N_x:]
        D_til_1 = D_til[:,:self.plant.N_x]
        D_til_2 = D_til[:,self.plant.N_x:]

        A1 = A_x_disc + A_eta_hybrid_disc.dot(D_til_1)
        A2 = A_eta_hybrid_disc.dot(C_til)
        B1 = B_x_disc + A_eta_hybrid_disc.dot(D_til_2)


        print('multiplied matrices')
        print(A_eta_hybrid_disc.dot(D_til_1))
        self.A_hybrid =  np.block([[A1     ,  A2],
                                   [B_til_1,  A_til ]])

        print(eigvals(A_til))

        self.B_hybrid = np.block([[B1],
                                  [B_til_2]])

        # print(self.A_hybrid)
        # print(self.B_hybrid.shape)

    def generate_N4SID_model(self):

        U = self.U_minus.reshape(-1, self.U_minus.shape[-1]).T
        Y = self.Y_plus.reshape(-1, self.Y_plus.shape[-1]).T
        
        # print(U.shape,Y.shape)

        NumURows = 30
        NumUCols = 1800
        
        print('u',U.shape)
        print('y',Y.shape)
        print('NumURows',NumURows)
        print('NumUCols',NumUCols)

        AID,BID,CID,DID,CovID,S = ssid.N4SID(U,Y,NumURows,NumUCols,4)
        print(AID,BID,CID,DID)
        plt.plot(S / S.sum())
        plt.show()

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

    def f_hybrid(self,t,x,u):

        u = np.array([u])
        x_dot = np.dot(self.A_hybrid,x) + np.dot(self.B_hybrid,u)

        return x_dot

    @staticmethod
    def simulate_system(x_0, u_minus, t_f, dt, u_func, f_func, g_func, continuous = True):
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
            r = ode(f_func).set_integrator('vode', method = 'bdf')
            r.set_initial_value(x_0,t)

        t_array = []
        x_array = []
        u_array = []
        y_array = []
        
        # initial state and 
        x_t = copy.copy(x_0)
        y_t = g_func(t, x_t, u_minus)
        u_t = u_func(y_t, t)

        t_array.append(t)
        x_array.append(x_t)
        u_array.append([u_t])
        y_array.append(g_func(t,x_t,u_minus))


        #Simulate the system
        while t < t_f:

            if continuous:
                r.set_f_params(u_t).set_jac_params(u_t)
                x_t = r.integrate(r.t + dt)
            else:
                x_t = f_func(t, x_t, u_t)

            t = t + dt
            y_t = g_func(t, x_t, u_t)
            u_t = u_func(g_func(t, x_t, u_t), t)

            t_array.append(t)
            x_array.append(x_t)
            u_array.append([u_t])
            y_array.append(y_t)

        return np.array(t_array), np.array(u_array), np.array(x_array), np.array(y_array)

    def simulate_system_nonlinear(self, x_0, u_func, t_f):

        u_minus = np.zeros((self.plant.N_u,1))
        t,u,x,y = self.simulate_system(x_0, u_minus, t_f, self.dt_data,
                                        u_func, self.plant.f, self.plant.g,
                                        continuous = True)
        
        return t, u, x, y
        
    def simulate_system_dfl(self, x_0, u_func, t_f):

        u_minus = np.zeros((self.plant.N_u,1))
        eta_0 = self.plant.phi(0.0, x_0, u_minus)
        xi_0 = np.concatenate((x_0,eta_0))

        t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                                   u_func, self.f_dfl, self.plant.g,
                                                   continuous = True)
        
        return t, u, xi, y 

    def simulate_system_koop(self, x_0, u_func, t_f):

        u_minus = np.zeros((self.plant.N_u,1))
        xi_0 = self.plant.g(0.0, x_0, u_minus)

        t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                        u_func, self.f_koop, self.plant.g,
                                        continuous = False)
        return t, u, xi, y 

    def simulate_system_hybrid(self, x_0, u_func, t_f):

        u_minus = np.zeros((self.plant.N_u,1))
        eta_0 = self.plant.phi(0.0, x_0, u_minus)
        xi_0 = np.concatenate((x_0,eta_0))
        # print(xi_0)
        t,u,xi,y = self.simulate_system(xi_0, u_minus, t_f, self.dt_data,
                                                   u_func, self.f_hybrid, self.plant.g,
                                                   continuous = False)

        return t,u,xi,y