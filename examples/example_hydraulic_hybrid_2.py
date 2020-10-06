#!/usr/bin/env python

from dfl.dfl.dfl import *
from dfl.dfl.dynamic_system import *
from dfl.dfl.mpc import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

I_P = 1.0
I_L = 1.0
a1,a2,a3 = 1.0,1.0,1.0
k1,k2 = 1.0,1.0
C_1,C_2 = 1.0,1.0
Ps = 10
Pt = 0.1
class Plant1(DFLDynamicPlant):
    
    def __init__(self):
        
        # Structure of system
        self.N_x = 5
        self.N_eta = 4
        self.N_u = 1

        # Combined system order
        self.N = self.N_x + self.N_eta

        # User defined matrices for DFL
        # self.A_cont_x  = np.array([[0.0, 0.0, 0.0,   1/I_2],
        #                       [0.0, 0.0, 1/I_1, -1/I_2],
        #                       [0.0, 0.0, 0.0, 0.0],
        #                       [0.0, 0.0, 0.0, 0.0]])

        # self.A_cont_eta = np.array([[0.0 , 0.0, 0.0, 0.0, 0.0],
        #                             [0.0 , 0.0, 0.0, 0.0, 0.0],
        #                             [-1.0, 0.0, 0.0,-1.0, -1.0],
        #                             [0.0 ,-1.0, -1.0, 1.0, 1.0]])

        # self.B_cont_x = np.array([[0.0],[0.0],[1.0],[0.0]])
        
        # Limits for inputs and states
        # self.x_min = np.array([-1.0,-1.0,-1.0,-1.0])
        # self.x_max = np.array([1.0 ,1.0, 1.0 ,1.0])
        # self.u_min = np.array([-2.5])
        # self.u_max = np.array([ 2.5])
        
        # Hybrid model definition
        # self.N_eta_hybrid = 3
        
        # self.P =  np.array([[1, 0, 0, 0, 0],
        #                     [0, 1, 1, 0, 0],
        #                     [0, 0, 0, 1, 1]])

        # self.A_cont_eta_hybrid =   self.A_cont_eta.dot(np.linalg.pinv(self.P))

   # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_r1(e,u):
        Cv = 1.0
        if u >= 0:
            f = u*Cv*np.sqrt(Ps - e)
        elif u < 0:
            f = u*Cv*np.sqrt(e - Pt)

        return f

    @staticmethod
    def phi_r2(e,u):

        Cv = 1.0
        if u >= 0:
            f = u*Cv*np.sqrt(e - Pt)
        elif u < 0:
            f = u*Cv*np.sqrt(Ps - e)

        return f

    @staticmethod
    def phi_rL(f):
        # e = a3*np.sign(f)*f**2
        e = 0.25*(np.tanh(5*f)-np.tanh(f)) + 0.5*np.tanh(5*f) + 0.01*f
        return e

    @staticmethod
    def phi_cL(q):
        e = k1*q + k2*q**3
        return e
    
    @staticmethod
    def phi_c1(q):
        e = k1*q
        return e
    # nonlinear state equations
    def f(self,t,x,u):

        x_dot = np.zeros(x.shape)
        q_1,q_2,q_L,p_P,p_L = x[0],x[1],x[2],x[3],x[4]

        x_dot[0] = -p_P/I_P  +  self.phi_r1(C_1*q_1 , u)
        x_dot[1] =  p_L/I_L  -  self.phi_r2(C_2*q_2 , u)
        x_dot[2] =  p_P/I_P - p_L/I_L 
        x_dot[3] =  C_1*q_1 - C_2*q_2 -self.phi_rL(p_P/I_P - p_L/I_L) - self.phi_cL(q_L) 
        x_dot[4] = self.phi_rL(p_P/I_P - p_L/I_L)  + self.phi_cL(q_L) 

        return x_dot

    # nonlinear observation equations
    def g(self,t,x,u):
        q_1,q_2,q_L,p_P,p_L = x[0],x[1],x[2],x[3],x[4]
        eta = self.phi(t,x,u)
        x = np.array([q1,q2,p1,p2])
        y = np.concatenate((x,eta))
        return y 
    
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        q1,q2,p1,p2 = x[0],x[1],x[2],x[3]
        
        eta = np.zeros(self.N_eta)
        # eta[0] = self.phi_r1(p1/I_1)
        # eta[1] = self.phi_r2(p2/I_2)
        # eta[2] = self.phi_c1(q1)
        # eta[3] = self.phi_r3(p1/I_1 - p2/I_2)
        # eta[4] = self.phi_c2(q2)

        return eta

    def phi_hybrid(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        eta = self.phi(t,x,u)
        eta_hybrid = self.P.dot(eta)

        return eta_hybrid 

###########################################################################################

#Dummy forcing laws
def zero_u_func(y,t):
    return 1 

def rand_u_func(y,t):
    return np.random.normal(0.0,0.3)

def sin_u_func(y,t):
    
    # if t>2.7:
    #     u =  4.0
    # else:
    #     u = 0.0
    
    # return u
    # 1*signal.square(0.5 * t)
    return 0.2 #*signal.square(1*t)

rho = 1000
mu = 8.9*np.power(10.0,-4.0)
l = 0.03
eps = 0.00001

def darcy_f(v):
    Re = rho*v*l/mu
    a = 1/(1+ np.power(Re/2712,8.4))
    b = 1/(1+ np.power(Re/(150*l/eps),8.4))
    f = np.power(64/Re,a)*np.power(0.75*np.log(Re/5.37),(2*(a-1)*b))*np.power(0.88*np.log(l/eps),(2*(a-1)*(1-b)))
    return f

def e(v):
    f = darcy_f(v)
    dp = f*rho*v**2/l
    return dp

if __name__== "__main__":

    ####
    # v = np.linspace(0.025,.2,100)
    # f = np.zeros(100)
    # for i in range(100):
    #     f[i] = e(v[i])

    # plt.plot(v,f)
    # plt.show()
    # exit()

    ################# HYBRID MODEL TEST ##############################################
    plant2 = Plant1()
    dfl1 = DFL(plant2, dt_data = 0.05, dt_control = 0.2)

    x_0 = np.array([4.0, 0.5, 0.0, 0.0, 0.0])
    t, u_nonlin, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0, sin_u_func, 10.0)


    # np.random.seed(seed = seed)
    # t, u_dfl, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, rand_u_func, 10.0)
    # # t, u, x_koop1, y_koop = dfl1.simulate_system_koop(x_0, sin_u_func, 10.0)
    
    # np.random.seed(seed = seed)
    # t, u_hybrid, x_hybrid, y_hybrid = dfl1.simulate_system_hybrid(x_0, rand_u_func, 10.0)

    # dfl1.generate_N4SID_model()

    fig, axs = plt.subplots(6, 1)
    fig.suptitle('State variables', fontsize=16)
    axs[0].plot(t, y_nonlin[:,0],'k', label = 'True')
    axs[1].plot(t, y_nonlin[:,1],'k')
    axs[2].plot(t, y_nonlin[:,2],'k')
    axs[3].plot(t, y_nonlin[:,3],'k')
    axs[4].plot(t, y_nonlin[:,4],'k')

    axs[5].plot(t, u_nonlin,'k')


    # axs[0].plot(t, x_dfl[:,0],'r', label = 'DFL Discrete')
    # axs[1].plot(t, x_dfl[:,1],'r')
    # axs[2].plot(t, x_dfl[:,2],'r')
    # axs[3].plot(t, x_dfl[:,3],'r')
    # axs[4].plot(t, u_dfl,'r')

    # axs[0].plot(t, y_hybrid[:,0],'g', label = 'Hybrid order 3')
    # axs[1].plot(t, y_hybrid[:,1],'g')
    # axs[2].plot(t, y_hybrid[:,2],'g')
    # axs[3].plot(t, y_hybrid[:,3],'g')
    # axs[4].plot(t, u_hybrid,'g')

    # axs[0].plot(t, y_hybrid_2[:,0],'b', label = 'Hybrid order 5')
    # axs[1].plot(t, y_hybrid_2[:,1],'b')
    # axs[2].plot(t, y_hybrid_2[:,2],'b')
    # axs[3].plot(t, y_hybrid_2[:,3],'b')
    # axs[4].plot(t, u_hybrid_2,'b')

    # axs[0].set_ylabel('q1')
    # axs[1].set_ylabel('q2')
    # axs[2].set_ylabel('p1')
    # axs[3].set_ylabel('p2')
    # axs[4].set_ylabel('u')

    # axs[0].legend(loc='right')

    # fig, axs = plt.subplots(5, 1)
    # fig.suptitle('Auxilliary variables', fontsize=16)
    # axs[0].plot(t, y_nonlin[:,4],'k', label = 'True')
    # axs[1].plot(t, y_nonlin[:,5],'k')
    # axs[2].plot(t, y_nonlin[:,6],'k')
    # axs[3].plot(t, y_nonlin[:,7],'k')
    # axs[4].plot(t, y_nonlin[:,8],'k')

    # axs[0].plot(t, x_dfl[:,4],'r', label = 'DFL')
    # axs[1].plot(t, x_dfl[:,5],'r')
    # axs[2].plot(t, x_dfl[:,6],'r')
    # axs[3].plot(t, x_dfl[:,7],'r')
    # axs[4].plot(t, x_dfl[:,8],'r')

    # axs[0].set_ylabel('eR1')
    # axs[1].set_ylabel('eR2')
    # axs[2].set_ylabel('eC1')
    # axs[3].set_ylabel('eR3')
    # axs[4].set_ylabel('eC2')
    # axs[0].legend(loc='right')


    # axs[0].plot(t, y_hybrid[:,0],'g')
    # axs[1].plot(t, y_hybrid[:,1],'g')
    # axs[2].plot(t, y_hybrid[:,2],'g')
    # axs[3].plot(t, y_hybrid[:,3],'g')
    # axs[4].plot(t, u_hybrid,'g')
    
    plt.show()
    exit()
    ################# HYBRID MODEL TEST ##############################################
    plant1 = Plant1()
    dfl1 = DFL(plant1, dt_data = 0.05, dt_control = 0.2)
    # setattr(plant1, "g", Plant1.gkoop1)

    dfl1.generate_data_from_random_trajectories( t_range_data = 5.0, n_traj_data = 100 ,plot_sample = True )
    dfl1.generate_hybrid_model()
    dfl1.generate_H_matrix()
    # dfl1.generate_K_matrix()

    # x_0 = np.random.uniform(plant1.x_init_min,plant1.x_init_max)
    x_0 = np.array([0,0,0,0])
    seed = np.random.randint(5)

    np.random.seed(seed = seed)
    t, u_nonlin, x_nonlin, y_nonlin = dfl1.simulate_system_nonlinear(x_0, sin_u_func, 10.0)
    t, u_dfl, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, sin_u_func, 10.0)
    t, u_hybrid, x_hybrid, y_hybrid = dfl1.simulate_system_hybrid(x_0, sin_u_func, 10.0)


    t, u_hybrid, x_hybrid, y_hybrid = dfl1.simulate_system_hybrid(x_0, sin_u_func, 10.0)

    # np.random.seed(seed = seed)
    # t, u_dfl, x_dfl, y_dfl = dfl1.simulate_system_dfl(x_0, rand_u_func, 10.0)
    # # t, u, x_koop1, y_koop = dfl1.simulate_system_koop(x_0, sin_u_func, 10.0)
    
    # np.random.seed(seed = seed)
    # t, u_hybrid, x_hybrid, y_hybrid = dfl1.simulate_system_hybrid(x_0, rand_u_func, 10.0)

    # dfl1.generate_N4SID_model()

    fig, axs = plt.subplots(5, 1)

    axs[0].plot(t, y_nonlin[:,0],'k')
    axs[1].plot(t, y_nonlin[:,1],'k')
    axs[2].plot(t, y_nonlin[:,2],'k')
    axs[3].plot(t, y_nonlin[:,3],'k')
    axs[4].plot(t, u_nonlin,'k')

    axs[0].plot(t, y_dfl[:,0],'r')
    axs[1].plot(t, y_dfl[:,1],'r')
    axs[2].plot(t, y_dfl[:,2],'r')
    axs[3].plot(t, y_dfl[:,3],'r')
    axs[4].plot(t, u_dfl,'r')

    axs[0].plot(t, y_hybrid[:,0],'g')
    axs[1].plot(t, y_hybrid[:,1],'g')
    axs[2].plot(t, y_hybrid[:,2],'g')
    axs[3].plot(t, y_hybrid[:,3],'g')
    axs[4].plot(t, u_hybrid,'g')

    # axs[0].plot(t, y_dfl[:,0] ,'b-.')
    # axs[0].plot(t, y_hybrid[:,0] ,'b--')

    # axs[1].plot(t, y_nonlin[:,1],'r')
    # axs[1].plot(t, y_dfl[:,1],'r-.')
    # axs[1].plot(t, y_hybrid[:,1] ,'r--')
  
    # axs[2].plot(t, u_nonlin,'g')
    # axs[2].plot(t, u_dfl,'r-.')
    # axs[2].plot(t, u_hybrid,'b--')

    # axs[0].set_xlim(0, t_f)
    axs[2].set_xlabel('time')
    
    axs[0].set_ylabel('x1')
    axs[1].set_ylabel('x2')
    axs[2].set_ylabel('x3')
    axs[3].set_ylabel('x4')
    axs[4].set_ylabel('u')

    plt.show()

    exit()

    # ##################################################################
    # dfl1.generate_H_matrix()
    # dfl1.generate_K_matrix()

    # dfl2 = DFL(plant2)
    # dfl2.generate_data_from_random_trajectories()
    # dfl2.generate_K_matrix()
    
    # t_f = 5.0
    # N_tests = 100
    # # x_0 = np.array([1.0,1.0])
    # error_koopman = np.array([0,0])
    # error_dfl = np.array([0,0])


    # for i in range(N_tests):

    #     x_0 = np.random.uniform(plant1.x_init_min,plant1.x_init_max)
    #     t, u, x_nonlin = dfl1.simulate_system_nonlinear(x_0, zero_u_func, t_f)
    #     t, u, x_dfl = dfl1.simulate_system_dfl(x_0, zero_u_func, t_f)
    #     t, u, x_koop1 = dfl1.simulate_system_koop(x_0, zero_u_func, t_f)
    #     t, u, x_koop2 = dfl2.simulate_system_koop(x_0, zero_u_func, t_f)

    #     error_koopman1 =+ np.mean(np.power(x_nonlin - x_koop1[:,0:2],2),axis = 0)
    #     error_koopman2 =+ np.mean(np.power(x_nonlin - x_koop2[:,0:2],2),axis = 0)
    #     error_dfl     =+ np.mean(np.power(x_nonlin -  x_dfl[:,0:2],2),axis = 0)

    # print('Koopman Error',error_koopman1)
    # print('Koopman Error 2',error_koopman2)
    # print('DFL Error', error_dfl)

    # fig, axs = plt.subplots(2, 1)
    
    # axs[0].plot(t, x_nonlin[:,0], 'b')
    # # axs[0].plot(t, x_dfl[:,0] ,'b-.')
    # axs[0].plot(t, x_koop1[:,0] ,'b.')
    # axs[0].plot(t, x_koop2[:,0] ,'b--')

    # axs[1].plot(t, x_nonlin[:,1],'r')
    # # axs[0].plot(t, x_dfl[:,1],'r-.')
    # axs[1].plot(t, x_koop1[:,1] ,'r.')
    # axs[1].plot(t, x_koop2[:,1] ,'r--')

    # axs[0].set_xlim(0, t_f)
    # axs[1].set_xlabel('time')

    # axs[0].set_ylabel('state 1')
    # axs[1].set_ylabel('state 2')

    # axs[0].grid(True)
    # axs[1].grid(True)

    # # axs[1].plot(t, u)
    # # axs[1].set_ylabel('input')

    # fig.tight_layout()
    # plt.show()

    # plt.matshow(dfl2.A_koop-np.eye(35))
    # plt.show()