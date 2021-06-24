#!/usr/bin/env python

from dfl.dfl.dfl import *
from dfl.dfl.dynamic_system import *
from dfl.dfl.mpc import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['mathtext.default'] = 'rm'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams["legend.loc"] = 'upper left'
from mpl_toolkits.mplot3d import axes3d
from scipy import signal

alpha  = 0.2
beta   = 2.
gamma  = 2.1
delta  = 3.0
omega = 0.5

class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        
        self.n_x = 2
        self.n_eta = 1
        self.n_u = 1

        self.n = self.n_x + self.n_eta

        # User defined matrices for DFL
        self.A_cont_x  = np.array([[0.0, 1.0],
                                   [0.0, -delta]])

        self.A_cont_eta = np.array([[0.0], [-1.]])

        self.B_cont_x = np.array([[0.0],[1.0]])

        # Limits for inputs and states
        self.x_min = np.array([-2.0,-1.0])
        self.x_max = np.array([0.5 ,3.0])

        self.u_min = np.array([-1.5])
        self.u_max = np.array([ 1.5])

        # Hybrid model
        self.P =  np.array([[1]])

        self.A_cont_eta_hybrid =   self.A_cont_eta.dot(np.linalg.pinv(self.P))


    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c1(q):
        e = alpha*q + beta*q**3
        return e
    
    # nonlinear state equations
    def f(self,t,x,u):

        x_dot = np.zeros(x.shape)
        q,v = x[0],x[1]
        x_dot[0] = v
        x_dot[1] = -delta*v -self.phi_c1(q) + u 
        # x_dot[1] = -self.phi_rc(q,v) + u 

        return x_dot

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        q,v = x[0], x[1]
        y = np.array([q,v])
        return y 
    
    @staticmethod
    def gkoop1(t,x,u):
        q,v = x[0], x[1]
        y = np.array([q,v,Plant1.phi_c1(q)])
        return y  
    
    @staticmethod
    def gkoop3(t,x,u):
        q,v = x[0], x[1]
        y = np.array([q,v])
        return y  
    
    @staticmethod
    def gkoop2(t,x,u):
        q,v = x[0],x[1]

        y = np.array([q,v,q**2,q**3,q**4,q**5,q**6,q**7,
                      v**2,v**3,v**4,v**5,v**6,v**7,v**9,v**11,v**13,v**15,v**17,v**19,
                      v*q,v*q**2,v*q**3,v*q**4,v*q**5,
                      v**2*q,v**2*q**2,v**2*q**3,v**2*q**4,v**2
                      *q**5,
                      v**3*q,v**3*q**2,v**3*q**3,v**3*q**4,v**3*q**5])
        return y 


    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        q,v = x[0],x[1]
        
        eta = np.zeros(self.n_eta)
        eta[0] = self.phi_c1(q)

        # eta[0] = self.phi_rc(q,v)
        # eta[1] = 0.0 
        return eta

    def phi_hybrid(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        q,v = x[0],x[1]
        eta = np.zeros(self.n_eta)
        eta[0] = self.phi_c1(q) + self.phi_r1(v)

        return eta

###########################################################################################

#Dummy forcing laws
def zero_u_func(y,t):
    return 1 

def rand_u_func(y,t):
    # return np.random.normal(0.0,0.3)
    return 1.0*signal.square(3 * t)

def sin_u_func(y,t):
    # return 1.5*signal.square(3 * t)
    return gamma*np.cos(3*omega*t)

if __name__== "__main__":

    ################# DFL MODEL TEST ##############################################
    plant1 = Plant1()
    dfl1 = DFL(plant1, dt_data = 0.05, dt_control = 0.3)
    setattr(plant1, "g", Plant1.gkoop3)
    dfl1.mu_x, dfl1.sigma_x = 0.0, 0.0
    
    # x_0 = np.array([0.0,0.0])
    # t, u_nonlin, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0,  sin_u_func, 15.0)

    # fig, axs = plt.subplots(4, 1)
    # axs[0].plot(t, y_nonlin[:,0])
    # axs[1].plot(t, y_nonlin[:,1])
    # axs[2].plot( y_nonlin[:,0], y_nonlin[:,1])
    # axs[3].plot( y_nonlin[:,2], y_nonlin[:,1])

    # axs[0].set_ylabel('x')
    # plt.show()

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(y_nonlin[:,0], y_nonlin[:,1], y_nonlin[:,2], label='parametric curve')
    # ax.legend()
    # plt.show()

    dfl1.generate_data_from_random_trajectories( t_range_data = 20.0, n_traj_data = 1 )
    dfl1.generate_DFL_disc_model()
    dfl1.regress_K_matrix()

    # x_0 = np.random.uniform(plant1.x_min,plant1.x_max)
    x_0 = np.array([-1.81,-0.805])
    # x_0 = np.array([0.,0])
    seed = np.random.randint(5)
    T = 5
    np.random.seed(seed = seed)
    t, u_nonlin, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0, rand_u_func, T)
    
    np.random.seed(seed = seed)
    t, u_dfl, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, rand_u_func, T, continuous = False)

    np.random.seed(seed = seed)
    t, u_koop, x_koop, y_koop = dfl1.simulate_system_koop(x_0, rand_u_func, T)
    
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(t, y_nonlin[:,0], 'k')
    axs[0].plot(t, y_dfl[:,0] )
    axs[0].plot(t, y_koop[:,0] )

    axs[1].plot(t, y_nonlin[:,1],'k')
    axs[1].plot(t, y_dfl[:,1])
    axs[1].plot(t, y_koop[:,1] )
  
    axs[2].plot(t, u_nonlin,'k')
    # axs[2].plot(t, u_dfl,'r-.')
    # axs[2].plot(t, u_koop,'b--')

    axs[2].set_xlabel('time')
    
    axs[0].set_ylabel('x')
    axs[1].set_ylabel('v')
    axs[2].set_ylabel('u')

    plt.show()

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(y_nonlin[:,0], y_nonlin[:,1], y_nonlin[:,2], label='parametric curve')
    # ax.plot(y_koop[:,0], y_koop[:,1], y_koop[:,2], label='parametric curve')
    # ax.legend()
    # plt.show()