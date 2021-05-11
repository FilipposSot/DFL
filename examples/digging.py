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
        self.n_eta = 3
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
        pass

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        return np.copy(x)

    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        pass
    
    @staticmethod
    def generate_data_from_file(file_name: str, test_ndx: int=4):
        '''
        x = [x, y, phi, v_x, v_y, omega], e = [a_x,a_y,alpha, F_x, F_y, m_soil]
        u = [u_x,u_y,tau]
        '''

        # Extract data from file
        data = np.load(file_name)
        t = data['t']
        x = data['x']
        e = data['e']
        e = e[:,:,3:] # Filippos Curating: rm accelerations
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

def main(test_ndx):
    plant1 = Plant1()
    fig, axs = plt.subplots(3,2)
    # fig.suptitle(test_ndx)

    data, test_data = Plant1.generate_data_from_file('data_nick_not_flat.npz', test_ndx=test_ndx)
    driving_fun = test_data['u']
    t = test_data['t']
    dt_data = t[1]-t[0]
    dt_control = t[1]-t[0]
    x_0 = np.copy(test_data['x'  ][0,:])
    z_0 = np.copy(test_data['eta'][0,:])
    xs_0 = np.concatenate((x_0,z_0))
    axs[0,0].plot(t, test_data['x'  ][:,0], 'k-', label='Ground Truth') # x
    axs[1,0].plot(t, test_data['x'  ][:,1], 'k-')   # y
    axs[2,0].plot(t, test_data['x'  ][:,2], 'k-')   # phi
    axs[0,1].plot(t, test_data['eta'][:,0], 'k-')   # Fx
    axs[1,1].plot(t, test_data['eta'][:,1], 'k-')   # Fx
    axs[2,1].plot(t, test_data['eta'][:,2], 'k-')   # m

    koo = dm.Koopman(plant1, dt_data=dt_data, dt_control=dt_control, observable='polynomial', n_koop=64)
    koo.learn(data)
    t, u, x_koo, y_koo = koo.simulate_system(x_0, driving_fun, t[-1])
    axs[0,0].plot(t, x_koo[:,0], 'g-.', label='Koopman')
    axs[1,0].plot(t, x_koo[:,1], 'g-.')
    axs[2,0].plot(t, x_koo[:,2], 'g-.')
    # axs[0,1].plot(t, x_koo[:,6], 'g-.')
    # axs[1,1].plot(t, x_koo[:,7], 'g-.')
    # axs[2,1].plot(t, x_koo[:,8], 'g-.')

    dmd = dm.Koopman(plant1, dt_data=dt_data, dt_control=dt_control, observable='polynomial', n_koop=len(xs_0))
    dmd.learn(data, dmd=True)
    _, _, x_dmd, y_dmd = dmd.simulate_system(xs_0, driving_fun, t[-1])
    axs[0,0].plot(t, x_dmd[:,0], 'r-.', label='DMDc')
    axs[1,0].plot(t, x_dmd[:,1], 'r-.')
    axs[2,0].plot(t, x_dmd[:,2], 'r-.')
    axs[0,1].plot(t, x_dmd[:,6], 'r-.')
    axs[1,1].plot(t, x_dmd[:,7], 'r-.')
    axs[2,1].plot(t, x_dmd[:,8], 'r-.')

    lrn = dm.L3(plant1, 4, dt_data=dt_data, dt_control=dt_control, ac_filter='linear', retrain=False, model_fn='model_dig', hidden_units_per_layer=64)
    lrn.learn(data)
    _, _, x_lrn, y_lrn = lrn.simulate_system(xs_0, driving_fun, t[-1])
    axs[0,0].plot(t, x_lrn[:,0], 'b-.', label='L3')
    axs[1,0].plot(t, x_lrn[:,1], 'b-.')
    axs[2,0].plot(t, x_lrn[:,2], 'b-.')
    axs[0,1].plot(t, x_lrn[:,6], 'b-.')
    axs[1,1].plot(t, x_lrn[:,7], 'b-.')
    axs[2,1].plot(t, x_lrn[:,8], 'b-.')

    lrn = dm.L3(plant1, 4, dt_data=dt_data, dt_control=dt_control, ac_filter='none', retrain=True, model_fn='model', hidden_units_per_layer=64)
    lrn.learn(data)
    _, _, x_lrn, y_lrn = lrn.simulate_system(xs_0, driving_fun, t[-1])
    axs[0,0].plot(t, x_lrn[:,0], 'm-.', label='L3 (NoF)')
    axs[1,0].plot(t, x_lrn[:,1], 'm-.')
    axs[2,0].plot(t, x_lrn[:,2], 'm-.')
    axs[0,1].plot(t, x_lrn[:,6], 'm-.')
    axs[1,1].plot(t, x_lrn[:,7], 'm-.')
    axs[2,1].plot(t, x_lrn[:,8], 'm-.')

    axs[0,0].legend()

    axs[2,0].set_xlabel('time (s)')
    axs[2,1].set_xlabel('time (s)')

    axs[0,0].set_ylabel('x (m)')
    axs[1,0].set_ylabel('y (m)')
    axs[2,0].set_ylabel('phi (rad)')
    axs[0,1].set_ylabel('F_x (kN)')
    axs[1,1].set_ylabel('F_y (kN)')
    axs[2,1].set_ylabel('m_soil (MT)')

    for r in range(2):
        for c in range(2):
            axs[r,c].set_xticks([])

    for r in range(3):
        axs[r,0].set_xlim(-0.1,2)
    axs[0,0].set_ylim(-3.4,-2.2)
    axs[1,0].set_ylim(0.1,0.7)
    axs[2,0].set_ylim(1.1,2.3)
    
    # axs[0,0].set_ylabel('Depth')
    # axs[1,0].set_ylabel('F_x')
    # axs[1,1].set_ylabel('F_y')
    # axs[0,1].set_ylabel('m_soil')

    # axs[0,0].set_xlim(-0.2,5)
    # axs[0,1].set_xlim(-0.2,5)
    # axs[1,0].set_xlim(-0.2,5)
    # axs[1,1].set_xlim(-0.2,5)

    # axs[0,0].set_ylim(-1,1.5)
    # axs[0,1].set_ylim(-6,6)
    # axs[1,0].set_ylim(-20,20)
    # axs[1,1].set_ylim(-15,15)

    plt.show()

if __name__== "__main__":
    # for test_ndx in range(3,6):
    #     main(test_ndx)
    main(5)