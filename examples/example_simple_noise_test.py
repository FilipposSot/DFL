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
import matplotlib.colors as colors
import matplotlib.pyplot as plt




from scipy import signal
from scipy.linalg import logm
m = 1.0
k11 = 0.2
k13 = 2.0
b1  = 3.0

class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        
        self.n_x = 2
        self.n_eta = 2
        self.n_u = 1

        self.n = self.n_x + self.n_eta

        # User defined matrices for DFL
        self.A_cont_x  = np.array([[0.0, 1.0],
                              [0.0, 0.0]])

        self.A_cont_eta = np.array([[0.0, 0.0],
                               [-1/m,-1/m]])

        self.B_cont_x = np.array([[0.0],[1.0]])

        # Limits for inputs and states
        self.x_min = np.array([-2.0,-2.0])
        self.x_max = np.array([2.0 ,2.0])

        self.u_min = np.array([-2.5])
        self.u_max = np.array([ 2.5])

        # Hybrid model
        self.P =  np.array([[1, 1]])

        self.A_cont_eta_hybrid =   self.A_cont_eta.dot(np.linalg.pinv(self.P))


    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c1(q):
        e = k11*q + k13*q**3
        return e

    @staticmethod
    def phi_r1(f):
        # e = b1*np.sign(f)*np.abs(f)*np.abs(f)
        e = b1*np.sign(f)*f**2
        return e

    @staticmethod
    def phi_rc(q,v):
        return 5*v*np.abs(q)
    
    # nonlinear state equations
    def f(self,t,x,u):

        x_dot = np.zeros(x.shape)
        q,v = x[0],x[1]
        x_dot[0] = v
        x_dot[1] = -self.phi_r1(v) -self.phi_c1(q) + u 
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
        y = np.array([q,v,Plant1.phi_c1(q), Plant1.phi_r1(v)])
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
        eta[1] = self.phi_r1(v)

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
    return np.random.normal(0.0,0.3)

def sin_u_func(y,t):
    return 0.5*signal.square(3 * t)
    # return np.sin(3*t) 

if __name__== "__main__":

    ################# DFL MODEL TEST ##############################################
    plant1 = Plant1()
    dfl1 = DFL(plant1, dt_data = 0.05, dt_control = 0.2)
    setattr(plant1, "g", Plant1.gkoop1)
    
    dfl1.mu_x, dfl1.sigma_x = 0.0, 0.0
    dfl1.generate_data_from_random_trajectories( t_range_data = 5.0, n_traj_data = 10)
    dfl1.generate_DFL_disc_model()
    dfl1.regress_K_matrix()

    A_noiseless = dfl1.A_disc_dfl
    K_noiseless = dfl1.A_disc_koop
    # # A_eig_noiseless, _ = np.linalg.eig( logm( A_noiseless ) )
    
    noise_sigma_array = np.linspace(0.0,0.5,20) # np.array([0.01,0.02,0.05,0.1,0.2,0.5,1.0])
    A_matrix_norm = np.zeros(noise_sigma_array.shape)
    K_matrix_norm = np.zeros(noise_sigma_array.shape)
    A_matrix_var = np.zeros(noise_sigma_array.shape)
    K_matrix_var = np.zeros(noise_sigma_array.shape)

    A_error = np.zeros(noise_sigma_array.shape)
    K_error = np.zeros(noise_sigma_array.shape)
    
    N_regressions = 50

    for i in range(len(noise_sigma_array)):

        A_matrix_array = np.zeros((4,4,N_regressions))
        K_matrix_array = np.zeros((4,4,N_regressions))

        for j in range(N_regressions):

            dfl1.mu_x, dfl1.sigma_x = 0.0, noise_sigma_array[i]
            dfl1.generate_data_from_random_trajectories( t_range_data = 5.0, n_traj_data = 10 )
            dfl1.generate_DFL_disc_model() 
            dfl1.regress_K_matrix()

            x_0 = np.array([0,0])
            seed = np.random.randint(5)

            np.random.seed(seed = seed)
            t, u_nonlin, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0, rand_u_func, 10.0)
            np.random.seed(seed = seed)
            t, u_dfl, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, rand_u_func, 10.0, continuous = False)
            np.random.seed(seed = seed)
            t, u_koop, x_koop, y_koop = dfl1.simulate_system_koop(x_0, rand_u_func, 10.0)

            if i == 0 and j == 0:
                fig, axs = plt.subplots(3, 1)

                axs[0].plot(t, y_nonlin[:,0], 'k')
                axs[0].plot(t, y_dfl[:,0] ,color = 'tab:orange')
                axs[0].plot(t, y_koop[:,0] ,color = 'tab:blue')

                axs[1].plot(t, y_nonlin[:,1],'k')
                axs[1].plot(t, y_dfl[:,1],color = 'tab:orange')
                axs[1].plot(t, y_koop[:,1] ,color = 'tab:blue')
              
                axs[2].plot(t, u_nonlin,'k')
                axs[2].plot(t, u_dfl,'k')
                axs[2].plot(t, u_koop,'k')

                axs[2].set_xlabel('Time')
                
                axs[0].set_ylabel(r'$\mathit{x}$')
                axs[1].set_ylabel(r'$\mathit{v}$')
                axs[2].set_ylabel(r'$\mathit{u}$')
                
                plt.tight_layout()

                plt.show()
                
            A_error[i] += np.mean((y_nonlin[:,:2] - y_dfl[:,:2])**2)/N_regressions
            K_error[i] += np.mean((y_nonlin[:,:2] - y_koop[:,:2])**2)/N_regressions

            A_matrix_array[:,:,j] = A_noiseless - dfl1.A_disc_dfl
            K_matrix_array[:,:,j] = K_noiseless - dfl1.A_disc_koop


        # A_eig_dfl, _       = np.linalg.eig(  logm(dfl1.A_disc_dfl ) )
        # A_eig_koop, _      = np.linalg.eig(  logm(dfl1.A_disc_koop ) )
        # print(A_noiseless)
        # print(A_eig_dfl)
        # print(A_eig_koop)
        
        # fig, axs = plt.subplots(2, 1)
        # axs[0].plot( np.real(A_eig_noiseless),  np.imag(A_eig_noiseless), 'k.', marker = 'o')
        # axs[0].plot( np.real(A_eig_dfl),  np.imag(A_eig_dfl), 'r.', marker = 'x')
        # axs[0].plot( np.real(A_eig_koop),  np.imag(A_eig_koop), 'b.', marker = '^')
        # plt.show()

        # A_matrix_norm[i] = np.linalg.norm(A_noiseless - dfl1.A_disc_dfl, 'fro')
        # K_matrix_norm[i] = np.linalg.norm(K_noiseless - dfl1.A_disc_koop,'fro')

        A_matrix_norm[i] = np.linalg.norm(np.mean(A_matrix_array,axis=2), 'fro')
        K_matrix_norm[i] = np.linalg.norm(np.mean(K_matrix_array,axis=2), 'fro')

        A_matrix_var[i] = np.linalg.norm(np.var(A_matrix_array, axis=2), 'fro')
        K_matrix_var[i] = np.linalg.norm(np.var(K_matrix_array, axis=2), 'fro')
    
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(noise_sigma_array,  A_matrix_norm, 'k', marker = '.')
    axs[0].plot(noise_sigma_array,  K_matrix_norm, color ='tab:blue', marker = '.')
    axs[0].set_xlabel(r'Noise $\sigma$')
    axs[0].set_ylabel(r'$\mathit{||\mathrm{E}[ A^0-\hat{A}]||_F}$')


    axs[1].plot(noise_sigma_array,  A_matrix_var, 'k', marker = '.')
    axs[1].plot(noise_sigma_array,  K_matrix_var,  color ='tab:blue', marker = '.')
    axs[1].set_xlabel(r'Noise $\sigma$')
    axs[1].set_ylabel(r'$\mathit{||\mathrm{Var}[ A^0-\hat{A}]||_F}$')

    axs[2].plot(noise_sigma_array,  A_error, 'k', marker = '.')
    axs[2].plot(noise_sigma_array,  K_error,  color ='tab:blue', marker = '.')
    axs[2].set_xlabel(r'Noise $\sigma$')
    axs[2].set_ylabel('Mean Squared Error')

    plt.tight_layout()

    plt.show()


    # dfl1.regress_K_matrix()
    # print()

    # x_0 = np.random.uniform(plant1.x_init_min,plant1.x_init_max)
    x_0 = np.array([0,0])
    seed = np.random.randint(5)

    np.random.seed(seed = seed)
    t, u_nonlin, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0, sin_u_func, 10.0)
    
    np.random.seed(seed = seed)
    t, u_dfl, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, sin_u_func, 10.0, continuous = False)
    np.random.seed(seed = seed)
    t, u_koop, x_koop, y_koop = dfl1.simulate_system_koop(x_0, sin_u_func, 10.0)
    
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(t, y_nonlin[:,0], 'k')
    axs[0].plot(t, y_dfl[:,0] ,color = 'tab:orange')
    axs[0].plot(t, y_koop[:,0] ,color = 'tab:blue')

    axs[1].plot(t, y_nonlin[:,1],'k')
    axs[1].plot(t, y_dfl[:,1],color = 'tab:orange')
    axs[1].plot(t, y_koop[:,1] ,color = 'tab:blue')
  
    axs[2].plot(t, u_nonlin,'k')
    axs[2].plot(t, u_dfl,'k')
    axs[2].plot(t, u_koop,'k')

    axs[2].set_xlabel('Time')
    
    axs[0].set_ylabel(r'$\mathit{x}$')
    axs[1].set_ylabel(r'$\mathit{v}$')
    axs[2].set_ylabel(r'$\mathit{u}$')
    plt.tight_layout()

    plt.show()