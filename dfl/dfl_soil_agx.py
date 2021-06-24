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

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

np.set_printoptions(precision = 4)
np.set_printoptions(suppress = True)

class DFLSoil():
    
    def __init__(self, dynamic_plant, dt_data = 0.05, dt_control = 0.05):
        
        self.plant = dynamic_plant
        self.dt_data = dt_data 
        self.dt_control = dt_control 
        self.n_s = 2

        self.koop_poly_order = 3

        self.H_disc_x   = np.zeros((self.plant.n_eta,self.plant.n_x))
        self.H_disc_eta = np.zeros((self.plant.n_eta,self.plant.n_eta)) 
        self.H_disc_s   = np.zeros((self.plant.n_eta,self.n_s)) 
        self.H_disc_u   = np.zeros((self.plant.n_eta,self.plant.n_u)) 
        self.A_disc_x   = np.zeros((self.plant.n_x,self.plant.n_x))
        self.A_disc_eta = np.zeros((self.plant.n_x,self.plant.n_eta))
        self.B_disc_x   = np.zeros((self.plant.n_x,self.plant.n_u))

    def regress_model_no_surface(self, X, Eta, U, S):
        '''
        regress the H matrix for DFL model
        '''

        X_minus   = X[:,:-1,:]
        Eta_minus = Eta[:,:-1,:]
        U_minus   = U[:,:-1,:]
        S_minus   = S[:,:-1,:]

        X_plus   = X[:,1:,:]
        Eta_plus = Eta[:,1:,:]
        U_plus   = U[:,1:,:]
        S_plus   = S[:,1:,:]

        omega = np.concatenate((X_minus.reshape(-1, X_minus.shape[-1]),
                                Eta_minus.reshape(-1, Eta_minus.shape[-1]),
                                U_minus.reshape(-1, U_minus.shape[-1])),axis=1).T
        
        Y = Eta_plus.reshape(-1, Eta_plus.shape[-1]).T

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

        A_lin =  np.block([[self.A_disc_x, self.A_disc_eta],
                           [self.H_disc_x, self.H_disc_eta]])

        print(np.absolute(np.linalg.eig(A_lin)[0]))

    def regress_model_custom(self, X, Eta, U, S):
        '''
        regress the H matrix for DFL model
        '''
        X_minus   = X[:,:-1,:]
        Eta_minus = Eta[:,:-1,:]
        U_minus   = U[:,:-1,:]
        S_minus   = S[:,:-1,:-1]

        X_plus   = X[:,1:,:]
        Eta_plus = Eta[:,1:,:]
        U_plus   = U[:,1:,:]
        S_plus   = S[:,1:,:-1]

        U_minus_zero_torque   = copy.deepcopy(U[:,:-1,:])
        U_minus_zero_torque[:,:,2] = 0.0

        omega = np.concatenate((X_minus.reshape(-1, X_minus.shape[-1]),
                                Eta_minus.reshape(-1, Eta_minus.shape[-1]),
                                S_minus.reshape(-1, S_minus.shape[-1]),
                                U_minus.reshape(-1, U_minus.shape[-1])),axis=1).T

        omega_zero_torque = np.concatenate((X_minus.reshape(-1, X_minus.shape[-1]),
                                Eta_minus.reshape(-1, Eta_minus.shape[-1]),
                                S_minus.reshape(-1, S_minus.shape[-1]),
                                U_minus_zero_torque.reshape(-1, U_minus_zero_torque.shape[-1])),axis=1).T

        Y = Eta_plus.reshape(-1, Eta_plus.shape[-1]).T

        H_disc = lstsq(omega.T,Y.T,rcond=None)[0].T

        H_disc_zero_torque = lstsq(omega_zero_torque.T,Y.T,rcond=None)[0].T

        H_disc[:2,:] =  H_disc_zero_torque[:2,:] 

        self.H_disc_x   = H_disc[:,:self.plant.n_x]
        self.H_disc_eta = H_disc[:, self.plant.n_x                                : self.plant.n_x + self.plant.n_eta]
        self.H_disc_s   = H_disc[:, self.plant.n_x + self.plant.n_eta             : self.plant.n_x + self.plant.n_eta + self.n_s]
        self.H_disc_u   = H_disc[:, self.plant.n_x + self.plant.n_eta  + self.n_s : self.plant.n_x + self.plant.n_eta + self.n_s + self.plant.n_u]

        H_disc_u_zero_torque =  H_disc_zero_torque[:, self.plant.n_x + self.plant.n_eta  + self.n_s : self.plant.n_x + self.plant.n_eta + self.n_s + self.plant.n_u]

        (self.A_disc_x, self.B_disc_x,_,_,_) = cont2discrete((self.plant.A_cont_x, self.plant.B_cont_x, 
                                                            np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                            self.dt_data)
        
        (_,self.A_disc_eta ,_,_,_)   = cont2discrete((self.plant.A_cont_x, self.plant.A_cont_eta, 
                                                    np.zeros(self.plant.n_x), np.zeros(self.plant.n_u)),
                                                    self.dt_data)

    def regress_model_new(self, X, Eta, U, S):
        '''
        regress the H matrix for DFL model
        '''
        X_minus   = X[:,:-1,:]
        Eta_minus = Eta[:,:-1,:]
        U_minus   = U[:,:-1,:]
        S_minus   = S[:,:-1,:-1]

        X_plus   = X[:,1:,:]
        Eta_plus = Eta[:,1:,:]
        U_plus   = U[:,1:,:]
        S_plus   = S[:,1:,:-1]

        omega = np.concatenate((X_minus.reshape(-1, X_minus.shape[-1]),
                                Eta_minus.reshape(-1, Eta_minus.shape[-1]),
                                S_minus.reshape(-1, S_minus.shape[-1]),
                                U_minus.reshape(-1, U_minus.shape[-1])),axis=1).T
        
        Y = Eta_plus.reshape(-1, Eta_plus.shape[-1]).T

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

    def g_Koop_x(self,x,eta,s):

        poly = PolynomialFeatures(self.koop_poly_order,include_bias=False)
        # xi =  np.array(np.concatenate((x,eta[:3])))

        y = poly.fit_transform(np.expand_dims(x,axis=0))

        return y[0,:]

    def g_Koop_x_eta(self,x,eta,s):

        poly = PolynomialFeatures(self.koop_poly_order,include_bias=False)
        xi = np.array(np.concatenate((x,eta)))
        y = poly.fit_transform(np.expand_dims(xi,axis=0))

        return y[0,:]

    def g_Koop_x_eta_2(self, x,eta,s):

        poly = PolynomialFeatures(self.koop_poly_order,include_bias=False)
        eta_new = np.array([eta[3], eta[4], eta[5], eta[3]/(0.169+eta[5]), eta[4]/(0.169+eta[5]), eta[3]*np.sin(x[2]),eta[4]*np.cos(x[2])])
        xi = np.array(np.concatenate((x,eta_new)))

        y = poly.fit_transform(np.expand_dims(xi,axis=0))

        return y[0,:]

    def g_Koop_x_eta_3(self, x,eta,s):

        poly = PolynomialFeatures(self.koop_poly_order,include_bias=False)
        r = eta[6] - x[1]
        eta_new = np.array([eta[3], eta[4], eta[5], r ])
        xi = np.array(np.concatenate((x,eta_new)))


        y = poly.fit_transform(np.expand_dims(xi,axis=0))

        return y[0,:]

    def g_Koop_x_eta_4(self, x, eta, s):

        poly = PolynomialFeatures(self.koop_poly_order,include_bias=False)

        r_S = eta[3] - x[1]
        r_com = copy.copy(r_S) 

        m_hat = 0.169 + eta[2]
        I_hat = 0.130 + eta[2]*r_com**2

        eta_new = np.array([eta[0], eta[1], eta[2],
                            eta[0]/m_hat , eta[1]/m_hat,
                            r_S*eta[0]*np.sin(x[2])/I_hat,
                            r_S*eta[1]*np.cos(x[2])/I_hat])

        xi = np.array(np.concatenate((x,eta_new)))

        y = poly.fit_transform(np.expand_dims(xi,axis=0))

        return y[0,:]

    def g_Koop_x_eta_5(self, x, eta, s):

        poly = PolynomialFeatures(self.koop_poly_order,include_bias=False)

        r_S = eta[3] - x[1]
        r_com = copy.copy(r_S) 

        m_hat = 0.169 + eta[2]
        I_hat = 0.130 + eta[2]*r_com**2

        eta_new = np.array([eta[0], eta[1], eta[2],
                            x[3]*m_hat , x[4]*m_hat,  x[5]*I_hat,
                            r_S*eta[0]*np.sin(x[2]),
                            r_S*eta[1]*np.cos(x[2])])

        xi = np.array(np.concatenate((x,eta_new)))

        y = poly.fit_transform(np.expand_dims(xi,axis=0))

        return y[0,:]

    def h_Koop_1(self, x, eta, s, u):

        r_S = eta[3] - x[1]
        r_com = eta[4] 

        m_hat = 0.169 + eta[2]
        I_hat = 0.130 + eta[2]*r_com**2

        A = np.array([[1/m_hat ,   0.    ,  0.  ],
                      [0.      , 1/m_hat ,  0.  ],
                      [0.      ,    0.   , 1/I_hat]])

        ups = A.dot(u)

        return ups

    def h_Koop_1_inverse(self, x, eta, s, ups):

        r_S = eta[3] - x[1]
        r_com = eta[4] 

        m_hat = 0.169 + eta[2]
        I_hat = 0.130 + eta[2]*r_com**2

        A = np.array([[m_hat ,   0.  ,  0.   ],
                      [0.    , m_hat ,  0.   ],
                      [0.    ,    0. , I_hat ]])

        u = A.dot(ups)

        return u

    def h_Koop_identity(self, x, eta, s, u):

        return u

    def h_Koop_identity_inverse(self, x, eta, s, ups):

        return ups

    def regress_model_Koop(self, X, Eta, U, S):
        '''
        regress the H matrix for DFL model
        '''
        X_minus   = X[:,:-1,:]
        Eta_minus = Eta[:,:-1,:]
        U_minus   = U[:,:-1,:]
        S_minus   = S[:,:-1,:]

        X_plus    = X[:,1:,:]
        Eta_plus  = Eta[:,1:,:]
        U_plus    = U[:,1:,:]
        S_plus    = S[:,1:,:]

        # omega = np.concatenate((X_minus.reshape(-1, X_minus.shape[-1]),
        #                         Eta_minus.reshape(-1, Eta_minus.shape[-1]),
        #                         U_minus.reshape(-1, U_minus.shape[-1])),axis=1).T
        
        # Y = np.concatenate((X_plus.reshape(-1, X_plus.shape[-1]),
        #                     Eta_plus.reshape(-1, Eta_plus.shape[-1])),axis=1).T

        # print(omega.shape)
        # print(Y.shape)

        # H_disc = lstsq(omega.T,Y.T,rcond=None)[0].T

        x_shape = X_minus.shape
        
        Y_minus = []
        Y_plus = []

        for j in range(x_shape[0]):
            for i in range(x_shape[1]):
                Y_minus.append( self.g_Koop(X_minus[j,i,:], Eta_minus[j,i,:], S_minus[j,i,:]))
                Y_plus.append(  self.g_Koop(X_plus[j,i,:],  Eta_plus[j,i,:],  S_plus[j,i,:]))

        Y_minus = np.array( Y_minus )
        Y_plus = np.array( Y_plus )

        Omega = np.concatenate((Y_minus,U_minus.reshape(-1, U_minus.shape[-1])),axis=1).T
        Y = Y_plus.T

        H_disc = lstsq(Omega.T,Y.T,rcond=None)[0].T

        # self.K_x   = H_disc[:,:self.plant.n_x + self.plant.n_eta ]
        # self.K_u   = H_disc[:, self.plant.n_x + self.plant.n_eta             : self.plant.n_x + self.plant.n_eta + self.plant.n_u]

        self.K_x   = H_disc[:,:-3 ]
        self.K_u   = H_disc[:,-3: ]
        
        self.n_koop = self.K_x.shape[0]

        return self.K_x, self.K_u

    def regress_model_Koop_with_surf(self, X, Eta, U, S, N = None):
        '''
        regress the H matrix for DFL model
        '''
        X_minus     = X[:,:-1,:]
        Eta_minus   = Eta[:,:-1,:]
        U_minus     = U[:,:-1,:]
        S_minus     = S[:,:-1,:-1]

        # print(S_minus.shape)

        X_plus      = X[:,1:,:]
        Eta_plus    = Eta[:,1:,:]
        U_plus      = U[:,1:,:]
        S_plus      = S[:,1:,:-1]

        x_shape = X_minus.shape
        
        Y_minus = []
        Y_plus = []
        Ups_minus = []
        Ups_plus = []

        for j in range(x_shape[0]):
            for i in range(x_shape[1]):
                Y_minus.append(self.g_Koop(X_minus[j,i,:], Eta_minus[j,i,:], S_minus[j,i,:]))
                Y_plus.append( self.g_Koop(X_plus[j,i,:],  Eta_plus[j,i,:],  S_plus[j,i,:]))
                Ups_minus.append(self.h_Koop(X_minus[j,i,:], Eta_minus[j,i,:], S_minus[j,i,:], U_minus[j,i,:]))
                Ups_plus.append( self.h_Koop(X_plus[j,i,:] , Eta_plus[j,i,:] , S_plus[j,i,:] , U_plus[j,i,:] ))
        
        Y_minus = np.array( Y_minus )
        Y_plus = np.array( Y_plus )
        Ups_minus = np.array( Ups_minus )
        Ups_plus = np.array( Ups_plus )


        if N is not None:
            
            train_indices = np.random.choice(range(Y_minus.shape[0]), size = N, replace=False)
            Y_minus     = Y_minus[ train_indices,:] 
            Y_plus      = Y_plus[ train_indices,:] 
            U_minus     = U_minus.reshape(-1, U_minus.shape[-1])[ train_indices,:]
            Ups_minus   = Ups_minus.reshape(-1, Ups_minus.shape[-1])[ train_indices,:]
            S_minus     = S_minus.reshape(-1, S_minus.shape[-1])[ train_indices,:] 

            Omega = np.concatenate((Y_minus, Ups_minus, S_minus),axis=1).T
            Y = Y_plus.T

        else:
        
            Omega = np.concatenate((Y_minus, Ups_minus.reshape(-1, Ups_minus.shape[-1]), S_minus.reshape(-1, S_minus.shape[-1])),axis=1).T
            Y = Y_plus.T

        
        result = lstsq(Omega.T,Y.T,rcond=None)
        
        H_disc = result[0].T
        residuals = result[1]

        self.K_x   = H_disc[:,:-(self.plant.n_u+self.n_s)]
        self.K_u   = H_disc[:, -(self.plant.n_u+self.n_s):-self.n_s ]
        self.K_s   = H_disc[:, -self.n_s: ]

        # clf = linear_model.ElasticNet(alpha = 0.0001, l1_ratio = 0.5)
        # clf.fit(Omega.T,Y.T)
        
        # self.K_x= clf.coef_[:,:-(self.plant.n_u+self.n_s)]
        # self.K_u = clf.coef_[:, -(self.plant.n_u+self.n_s):-self.n_s ]
        # self.K_s = clf.coef_[:, -self.n_s: ]

        # print(K_x_Lasso)

        self.n_koop = self.K_x.shape[0]

        return self.K_x, self.K_u, self.K_s
    
    def linearize_soil_dynamics_koop(self, x_nom):
        
        # print('---------------------------')
        s_nom, s_dash_nom, s_dash_dash_nom, s_dash_dash_dash_nom = self.soilShapeEvaluator.soil_surf_eval(x_nom[0])

        sigma_zero   =  np.array([s_nom, s_dash_nom]) 
        sigma_zero_d =  np.array([s_dash_nom, s_dash_dash_nom])

        T = np.zeros((self.n_s, self.n_koop ))
        T[:,0] = 1.0

        K_x_surf = self.K_x + self.K_s.dot(np.diag(sigma_zero_d)).dot(T)

        K_lin_eta = self.K_s.dot(sigma_zero) - self.K_s.dot(sigma_zero_d)*x_nom[0]

        return K_x_surf, self.K_u, K_lin_eta #  self.K_x, self.K_u, np.zeros(self.K_x.shape[0]) #       
    
    def linearize_soil_dynamics_no_surface(self, x_nom):
      
        A_lin =  np.block([[self.A_disc_x, self.A_disc_eta],
                           [self.H_disc_x, self.H_disc_eta]])

        B_lin = np.block([[self.B_disc_x],
                          [self.H_disc_u]])

        # constant bias term
        K_lin = np.concatenate((np.zeros(self.plant.n_x),np.zeros(self.plant.n_eta)))

        return   A_lin , B_lin , K_lin


    def linearize_soil_dynamics(self, x_nom):
        
        # print('---------------------------')
        s_nom, s_dash_nom, s_dash_dash_nom, s_dash_dash_dash_nom = self.soilShapeEvaluator.soil_surf_eval(x_nom[0])
        # print('x: ',x_nom[0])
        # print('s: ', s_nom)

        # sigma_zero   =  np.array([s_nom, s_dash_nom, s_dash_dash_nom]) 
        # sigma_zero_d =  np.array([s_dash_nom, s_dash_dash_nom, s_dash_dash_dash_nom])

        sigma_zero   =  np.array([s_nom, s_dash_nom]) 
        sigma_zero_d =  np.array([s_dash_nom, s_dash_dash_nom])

        T = np.zeros((self.n_s,self.plant.n_x))
        T[:,0] = 1.0

        H_disc_x_surf = self.H_disc_x + self.H_disc_s.dot(np.diag(sigma_zero_d)).dot(T)
        
        A_lin =  np.block([[self.A_disc_x , self.A_disc_eta],
                           [H_disc_x_surf , self.H_disc_eta]])

        B_lin = np.block([[self.B_disc_x],
                          [self.H_disc_u]])

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

        y_plus = np.dot(self.A_disc_dfl,x) + np.dot(self.B_disc_dfl, u)

        return y_plus

    def f_disc_dfl_tv(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])
        
        A_lin, B_lin, K_lin = self.linearize_soil_dynamics(x)
        # A_lin, B_lin, K_lin = self.linearize_soil_dynamics_no_surface(x)
        # y_plus = np.dot(self.K_x,x) + np.dot(self.K_u, u)
        y_plus = np.dot(A_lin,x) + np.dot(B_lin, u) + K_lin

        return y_plus

    def f_disc_koop(self,t,x,u):

        if not isinstance(u,np.ndarray):
            u = np.array([u])
        
        A_lin, B_lin, K_lin = self.linearize_soil_dynamics_koop(x)

        y_plus = np.dot(A_lin,x) + np.dot(B_lin, u) + K_lin
        
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

        return np.array(t_array), np.array(u_array), np.array(x_array), np.array(y_array)