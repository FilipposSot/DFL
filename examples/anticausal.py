#!/usr/bin/env python

import dfl.dynamic_system
import dfl.dynamic_model as dm

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

plt.rcParams["font.family"] = "Times New Roman"

class Plant1(dfl.dynamic_system.DFLDynamicPlant):
    
    def __init__(self):
        self.n_x = 1
        self.n_eta = 2
        self.n_u = 1

        # User defined matrices for DFL
        self.A_cont_x   = np.array([[0.0]])
        self.A_cont_eta = np.array([[0.0, 1.0]])
        self.B_cont_x   = np.array([[0.0]])

        # Limits for inputs and states
        self.x_min = np.array([-2.0])
        self.x_max = np.array([ 2.0])
        self.u_min = np.array([-2.5])
        self.u_max = np.array([ 2.5])

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

if __name__== "__main__":
    driving_fun = dfl.dynamic_system.DFLDynamicPlant.sin_u_func
    plant1 = Plant1()
    x_0 = np.zeros(plant1.n_x)
    fig, axs = plt.subplots(1, 1)

    tru = dm.GroundTruth(plant1)
    data = tru.generate_data_from_random_trajectories()
    t, u, x_tru, y_tru = tru.simulate_system(x_0, driving_fun, 10.0)
    axs.plot(t, u, 'gainsboro')
    axs.text(9.7, -0.43, 'u', fontsize='xx-large', color='tab:gray', fontstyle='italic')
    axs.plot(t, x_tru[:,0], 'k-', label='Ground Truth')

    koo = dm.Koopman(plant1, observable='polynomial')
    koo.learn(data)
    _, _, x_koo, y_koo = koo.simulate_system(x_0, driving_fun, 10.0)
    axs.plot(t, x_koo[:,0], 'g-.', label='Koopman')

    dmd = dm.DFL(plant1, ac_filter=False)
    dmd.learn(data)
    _, _, x_dmd, y_dmd = dmd.simulate_system(x_0, driving_fun, 10.0)
    axs.plot(t, x_dmd[:,0], 'c-.', label='DMDc')

    dfl = dm.DFL(plant1, ac_filter=True)
    dfl.learn(data)
    _, _, x_dfl, y_dfl = dfl.simulate_system(x_0, driving_fun, 10.0)
    axs.plot(t, x_dfl[:,0], 'r-.', label='DFL')

    lrn = dm.L3(plant1, 2, ac_filter='linear', model_fn='model_toy_acf', retrain=False, hidden_units_per_layer=256)
    lrn.learn(data)
    _, _, x_lrn, y_lrn = lrn.simulate_system(x_0, driving_fun, 10.0)
    axs.plot(t, x_lrn[:,0], 'b-.', label='L3')

    lnf = dm.L3(plant1, 2, ac_filter='none', model_fn='model_toy_nof', retrain=False, hidden_units_per_layer=256)
    lnf.learn(data)
    _, _, x_lnf, y_lnf = lnf.simulate_system(x_0, driving_fun, 10.0)
    axs.plot(t, x_lnf[:,0], 'm-.', label='L3 (NoF)')

    bb = (fig.subplotpars.left, fig.subplotpars.top+0.02, fig.subplotpars.right-fig.subplotpars.left, .1)
    axs.legend(bbox_to_anchor=bb, loc='lower left', ncol=6, mode="expand", borderaxespad=0., bbox_transform=fig.transFigure)

    axs.set_xlabel('time (s)')
    axs.set_ylabel('q (m)')
    fig.subplots_adjust(bottom=0.2)

    plt.show()