#!/usr/bin/env python

import dfl.dynamic_system
import dfl.dynamic_model as dm

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

m = 1.0
k11 = 0.2
k13 = 2.0
b1  = 3.0

class Plant1(dfl.dynamic_system.DFLDynamicPlant):
    
    def __init__(self):
        self.n_x = 6
        self.n_eta = 6
        self.n_u = 3

        self.assign_random_system_model()

    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c(q):
        return np.sign(q)*q**2

    @staticmethod
    def phi_r(f):
        return 2*(1/(1+np.exp(-4*f))-0.5)
    
    # nonlinear state equations
    def f(self,t,x,u):
        x_dot = np.zeros(x.shape)
        eta = self.phi(t,x,u)
        x_dot[0] = eta[1]

        return x_dot

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        if not isinstance(u,np.ndarray):
            u = np.array([u])
            
        q = x[0]
        ec = Plant1.phi_c(q)
        er = u[0]-ec
        f = Plant1.phi_r(er)

        return np.copy(x)

    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        if not isinstance(u,np.ndarray):
            u = np.array([u])
            
        q = x[0]
        ec = Plant1.phi_c(q)
        er = u[0]-ec
        f = Plant1.phi_r(er)
        
        eta = np.zeros(self.n_eta)
        eta[0] = ec
        eta[1] = f

        return eta
    
    @staticmethod
    def generate_data_from_file(file_name: str, test_ndx: int=4):
        '''
        x = [x , y, z, v_x, v_y, omega], e = [a_x,a_y,alpha, F_x, F_y, m_soil]
        u = [u_x,u_y,tau]
        '''

        # Extract data from file
        data = np.load(file_name)
        t = data['t']
        x = data['x']
        e = data['e']
        u = data['u']

        # Assemble data into paradigm
        t_data = t
        x_data = x
        u_data = u
        eta_data = e
        eta_dot_data = e
        y_data = np.copy(x)

        # Set aside test data
        t_data_test       = np.copy(t_data[test_ndx])
        x_data_test       = np.copy(x_data[test_ndx])
        u_data_test       = np.copy(u_data[test_ndx])
        eta_data_test     = np.copy(eta_data[test_ndx])
        eta_dot_data_test = np.copy(eta_dot_data[test_ndx])
        y_data_test       = np.copy(y_data[test_ndx])

        # Remove test data from training data
        t_data       = np.delete(      t_data,test_ndx,0)
        x_data       = np.delete(      x_data,test_ndx,0)
        u_data       = np.delete(      u_data,test_ndx,0)
        eta_data     = np.delete(    eta_data,test_ndx,0)
        eta_dot_data = np.delete(eta_dot_data,test_ndx,0)
        y_data       = np.delete(      y_data,test_ndx,0)

        # Inputs
        y_minus   = np.copy(  y_data[:, :-1,:])
        u_minus   =           u_data[:, :-1,:]
        x_minus   =           x_data[:, :-1,:]
        eta_minus =         eta_data[:, :-1,:]

        # Outputs
        y_plus   = np.copy(  y_data[:,1:  ,:])
        u_plus   =           u_data[:,1:  ,:]
        x_plus   =           x_data[:,1:  ,:]
        eta_plus =         eta_data[:,1:  ,:]

        # Return
        data = {
            't': t_data,
            'u': {
                'data':  u_data,
                'minus': u_minus,
                'plus':  u_plus
            },
            'x': {
                'data':  x_data,
                'minus': x_minus,
                'plus':  x_plus
            },
            'y': {
                'data':  y_data,
                'minus': y_minus,
                'plus':  y_plus
            },
            'eta': {
                'data':  eta_data,
                'minus': eta_minus,
                'plus':  eta_plus
            },
            'eta_dot': {
                'data':  eta_dot_data
            }
        }
        test_data = {
            't': t_data_test,
            'u': u_data_test,
            'x': x_data_test,
            'y': y_data_test,
            'eta': eta_data_test,
            'eta_dot': eta_dot_data_test
        }
        return data, test_data

if __name__== "__main__":
    plant1 = Plant1()
    fig, axs = plt.subplots(3, 1)

    data, test_data = Plant1.generate_data_from_file('data_nick_flat.npz', test_ndx=4)
    driving_fun = test_data['u']
    t = test_data['t']
    dt_data = t[1]-t[0]
    dt_control = t[1]-t[0]
    x_0 = np.copy(test_data['x'][0,:])
    axs[0].plot(t, test_data['x'][:,0], 'k-', label='Ground Truth')
    axs[1].plot(t, test_data['x'][:,1], 'k-')
    axs[2].plot(test_data['x'][:,0], test_data['x'][:,1], 'k-')

    koo = dm.Koopman(plant1, dt_data=dt_data, dt_control=dt_control, observable='polynomial', n_koop=32)
    koo.learn(data)
    t, u, x_koo, y_koo = koo.simulate_system(x_0, driving_fun, t[-1])
    axs[0].plot(t, x_koo[:,0], 'g-.', label='Koopman')
    axs[1].plot(t, x_koo[:,1], 'g-.')
    axs[2].plot(x_koo[:,0], x_koo[:,1], 'g-.')

    lrn = dm.L3(plant1, 7, dt_data=dt_data, dt_control=dt_control, ac_filter='linear')
    lrn.learn(data)
    _, _, x_lrn, y_lrn = lrn.simulate_system(x_0, driving_fun, t[-1])
    axs[0].plot(t, x_lrn[:,0], 'b-.', label='L3')
    axs[1].plot(t, x_lrn[:,1], 'b-.')
    axs[2].plot(x_lrn[:,0], x_lrn[:,1], 'b-.')

    axs[0].legend()

    axs[1].set_xlabel('time')
    axs[2].set_xlabel('x')
    
    axs[0].set_ylabel('x')
    axs[1].set_ylabel('y')
    axs[2].set_ylabel('y')

    plt.show()