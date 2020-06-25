#!/usr/bin/env python


import numpy as np
from dfl import *
from dynamic_system import *
import matplotlib.pyplot as plt

# T_RANGE_DATA = 1.0
# DT_DATA = 0.05
# N_TRAJ_DATA = 20
# X_INIT_MIN = np.array([0.0,0.0,0.0])
# X_INIT_MAX = np.array([1.0,1.0,1.0])

m = 1.0
k11 = 0.2
k13 = 2.0

b1  = 3.0


class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        # Linear part of states matrices


        self.N_x = 2
        self.N_eta = 2
        self.N_u = 1
        self.N_y = 4

        self.N = self.N_x + self.N_eta

        self.A_x  = np.array([[0.0, 1.0],
                              [0.0, 0.0]])

        self.A_eta = np.array([[0.0, 0.0],
                               [-1/m,-1/m]])

        self.B_x = np.array([[0.0],[1.0]])

        self.x_init_min = np.array([-2.0,-2.0])
        self.x_init_max = np.array([2.0 ,2.0])

        ##### hybrid model
        self.N_eta_hybrid = 1
        self.A_eta_hybrid = np.array([[0.0],
                                     [-1/m]])


    # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_c1(q):
        e = k11*q + k13*q**3
        return e

    @staticmethod
    def phi_r1(f):
        # e = b1*np.sign(f)*np.abs(f)*np.abs(f)
        e = b1*f**3
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
        return x_dot

    # nonlinear observation equations
    @staticmethod
    def g(t,x,u):
        q,v = x[0],x[1]
        y = np.array([q,v])
        return y 
    
    @staticmethod
    def gkoop1(t,x,u):
        q,v = x[0],x[1]
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
        
        eta = np.zeros(self.N_eta)
        eta[0] = self.phi_c1(q)
        eta[1] = self.phi_r1(v)

        return eta

    def phi_hybrid(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        q,v = x[0],x[1]
        eta = np.zeros(self.N_eta)
        eta[0] = self.phi_c1(q) + self.phi_r1(v)

        return eta

###########################################################################################

#Dummy forcing laws
def zero_u_func(y,t):
    return 1 

def rand_u_func(y,t):
    return np.sign(np.random.normal(0.0,0.3))

def sin_u_func(y,t):
    return np.sin(3*t) 





if __name__== "__main__":
    plant1 = Plant1()

    dfl1 = DFL(plant1)
    setattr(plant1, "g", Plant1.gkoop2)

    dfl1.generate_data_from_random_trajectories()
    # dfl1.generate_H_matrix()
    dfl1.generate_hybrid_model()
    # dfl1.generate_N4SID_model()
    # dfl1.generate_K_matrix()

    # x_0 = np.random.uniform(plant1.x_init_min,plant1.x_init_max)
    x_0 = np.array([0,0])
    t, u, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0, sin_u_func, 10.0)
    # t, u, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, zero_u_func, 10.0)
    # t, u, x_koop1, y_koop = dfl1.simulate_system_koop(x_0, zero_u_func, 10.0)
    t, u, x_hybrid, y_hybrid = dfl1.simulate_system_hybrid(x_0, sin_u_func, 10.0)

    # dfl1.generate_N4SID_model()

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(t, x_nonlin[:,0], 'b')
    # axs[0].plot(t, x_dfl[:,0] ,'b-.')
    # axs[0].plot(t, x_koop1[:,0] ,'b.')
    axs[0].plot(t, x_hybrid[:,0] ,'b--')

    axs[1].plot(t, x_nonlin[:,1],'r')
    # axs[1].plot(t, x_dfl[:,1],'r-.')
    # axs[1].plot(t, x_koop1[:,1] ,'r.')
    axs[1].plot(t, x_hybrid[:,1] ,'r--')

    plt.show()

    exit()

    ##################################################################
    dfl1.generate_H_matrix()
    dfl1.generate_K_matrix()

    dfl2 = DFL(plant2)
    dfl2.generate_data_from_random_trajectories()
    dfl2.generate_K_matrix()
    
    t_f = 5.0
    N_tests = 100
    # x_0 = np.array([1.0,1.0])
    error_koopman = np.array([0,0])
    error_dfl = np.array([0,0])


    for i in range(N_tests):

        x_0 = np.random.uniform(plant1.x_init_min,plant1.x_init_max)
        t, u, x_nonlin = dfl1.simulate_system_nonlinear(x_0, zero_u_func, t_f)
        t, u, x_dfl = dfl1.simulate_system_dfl(x_0, zero_u_func, t_f)
        t, u, x_koop1 = dfl1.simulate_system_koop(x_0, zero_u_func, t_f)
        t, u, x_koop2 = dfl2.simulate_system_koop(x_0, zero_u_func, t_f)

        error_koopman1 =+ np.mean(np.power(x_nonlin - x_koop1[:,0:2],2),axis = 0)
        error_koopman2 =+ np.mean(np.power(x_nonlin - x_koop2[:,0:2],2),axis = 0)
        error_dfl     =+ np.mean(np.power(x_nonlin -  x_dfl[:,0:2],2),axis = 0)

    print('Koopman Error',error_koopman1)
    print('Koopman Error 2',error_koopman2)
    print('DFL Error', error_dfl)

    fig, axs = plt.subplots(2, 1)
    
    axs[0].plot(t, x_nonlin[:,0], 'b')
    # axs[0].plot(t, x_dfl[:,0] ,'b-.')
    axs[0].plot(t, x_koop1[:,0] ,'b.')
    axs[0].plot(t, x_koop2[:,0] ,'b--')

    axs[1].plot(t, x_nonlin[:,1],'r')
    # axs[0].plot(t, x_dfl[:,1],'r-.')
    axs[1].plot(t, x_koop1[:,1] ,'r.')
    axs[1].plot(t, x_koop2[:,1] ,'r--')

    axs[0].set_xlim(0, t_f)
    axs[1].set_xlabel('time')

    axs[0].set_ylabel('state 1')
    axs[1].set_ylabel('state 2')

    axs[0].grid(True)
    axs[1].grid(True)

    # axs[1].plot(t, u)
    # axs[1].set_ylabel('input')

    fig.tight_layout()
    plt.show()

    plt.matshow(dfl2.A_koop-np.eye(35))
    plt.show()