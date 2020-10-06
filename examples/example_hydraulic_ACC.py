#!/usr/bin/env python

from dfl.dfl.dfl import *
from dfl.dfl.dynamic_system import *
from dfl.dfl.mpc import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
matplotlib.rcParams['mathtext.default'] = 'rm'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['pdf.fonttype'] = 42

from scipy import signal

import warnings

class PlantMinimal(DFLDynamicPlant):
    # simple piston and resistive fluid load
    # Single nonlinear load
    def __init__(self):
        
        # Structure of system
        self.N_x = 3
        self.N_eta = 3
        self.N_u = 1

        # Combined system order
        self.N = self.N_x + self.N_eta

        # User defined matrices for DFL
        self.A_cont_x  = np.array([[ 0., -1/I_L, 1/I_F],
                                   [ 0., 0.   , 0. ],
                                   [ 0., 0.   , 0. ]])

        self.A_cont_eta = np.array([[0., 0., 0.],
                                   [ 0., 1., 1.],
                                   [-1.,-1.,-1.]])

        self.B_cont_x = np.array([[0.0],[0.0],[B_1]])
        
        # Limits for inputs and states
        self.x_min = np.array([-1.0,-1.0,-1.0])
        self.x_max = np.array([ 1.0 ,1.0, 1.0 ])
        self.u_min = np.array([-1.])
        self.u_max = np.array([ 1.])
        
        # Hybrid model definition
        self.N_eta_hybrid = 2
        
        self.P =  np.array([[1, 0, 0],
                            [0, 1, 1]])

        self.A_cont_eta_hybrid =   self.A_cont_eta.dot(np.linalg.pinv(self.P))

   # functions defining constituitive relations for this particular system

    @staticmethod
    def phi_r_fluid(f):
        # fluid resistance with transition (between regimes)
        # f_dar = darcy_f(f)
        # e = np.sign(f)*f_dar*rho*f**2/l
        # if np.abs(f) < 0.05:
        #     a = 0.2
        # else: 
        #     a = 1.0
        e = np.sign(f)*f**2
        return e

    @staticmethod
    def phi_r_load(f):
        # friction like nonlinearity
        # e = np.sign(f)*f**2
        e = 0.25*(np.tanh(5*f)-np.tanh(f)) + 0.5*np.tanh(5*f) + 0.01*f
        return e

    @staticmethod
    def phi_c_load(q):
        thresh = 0.1

        if np.abs(q) < thresh:
            e = k1*q
        else:
            e = k2*q -(k2 - k1)*thresh*np.sign(q)

        return e

    # nonlinear state equations
    def f(self,t,x,u):

        x_dot = np.zeros(x.shape)
        q_1, p_L, p_F = x[0],x[1],x[2]

        f_1 = (1/I_F)*p_F - (1/I_L)*p_L
        f_f = (1/I_F)*p_F 

        x_dot[0] = f_1 
        x_dot[1] = self.phi_c_load(q_1) + self.phi_r_load(f_1)
        x_dot[2] = B_1*u - self.phi_r_fluid(f_f) - self.phi_c_load(q_1) - self.phi_r_load(f_1)

        return x_dot
           
    # nonlinear observation equations
    def g(self,t,x,u):
        q_1, p_L, p_F = x[0],x[1],x[2]
        eta = self.phi(t,x,u)
        x = np.array([q_1, p_L, p_F])
        y = np.concatenate((x,eta))

        return y 
     
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        q_1, p_L, p_F = x[0],x[1],x[2]
        
        f_1 = (1/I_F)*p_F - (1/I_L)*p_L
        f_f = (1/I_F)*p_F 

        eta = np.zeros(self.N_eta)

        eta[0] = self.phi_r_fluid(f_f)
        eta[1] = self.phi_r_load(f_1)
        eta[2] = self.phi_c_load(q_1) 

        return eta

    def phi_hybrid(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        eta = self.phi(t,x,u)
        eta_hybrid = self.P.dot(eta)

        return eta_hybrid 

class PlantNonCausal3(DFLDynamicPlant):
    
    def __init__(self):
        
        # Using the more typical bond graph variables

        # Structure of system
        self.N_x = 4
        self.N_eta = 6
        self.N_u = 1

        # Combined system order
        self.N = self.N_x + self.N_eta

        # User defined matrices for DFL
        # x =  V_A, V_B, x_p, p_p
        self.A_cont_x  = np.array([ [0. , 0., 0., - A/m],
                                    [0. , 0., 0., a*A/m],
                                    [0. , 0., 0.,  1./m],
                                    [0. , 0., 0.,    0.]])
        
        # eta = P_A, P_B, Q_A, Q_B, eCL, eRL]
        self.A_cont_eta = np.array([[-K_le,  K_le,   1.,  0., 0., 0.],
                                    [ K_le, -K_le,   0., -1., 0., 0.],
                                    [   0.,    0.,   0.,  0., 0., 0.],
                                    [    A,  -A*a,   0.,  0.,-1., -1]])

        self.B_cont_x = np.array([[0.0],[0.0],[0.0],[0.0]])
     
        self.x_min = np.array([ 0.5*V_A_0, 0.5*V_B_0, 0.0, 0.0])
        self.x_max = np.array([ 0.5*V_A_0, 0.5*V_B_0, 0.0, 0.0])
        self.u_min = np.array([-1.])
        self.u_max = np.array([ 1.])

        # Hybrid model definition
        self.N_eta_hybrid = 4
        
        self.P =  np.array([[1, -1, 0, 0, 0, 0],
                            [0, 0,  1, 0, 0, 0],
                            [0, 0,  0, 1, 0, 0],
                            [0, 0,  0, 0, 1, 1]])

        # self.P =  np.array([[1, 0, 0, 0, 0, 0],
        #                     [0, 1, 0, 0, 0, 0],
        #                     [0, 0, 1, 0, 0, 0],
        #                     [0, 0, 0, 1, 0, 0],
        #                     [0, 0, 0, 0, 1, 0],
        #                     [0, 0, 0, 0, 0, 1]])

        self.A_cont_eta_hybrid =   self.A_cont_eta.dot(np.linalg.pinv(self.P))

   # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_rA(e,u):

        if u >= 0:
            f = u*Cv*np.sqrt(np.abs(Ps - e))*np.sign(Ps - e)
        elif u < 0:
            f = u*Cv*np.sqrt(np.abs(e - Pt))*np.sign(e - Pt)

        return f

    @staticmethod
    def phi_rB(e,u):

        if u >= 0:
            f = u*Cv*np.sqrt(np.abs(e - Pt))*np.sign(e - Pt)
        elif u < 0:
            f = u*Cv*np.sqrt(np.abs(Ps - e))*np.sign(Ps - e)

        return f

    @staticmethod
    def phi_rL(f):
        # e = 0.25*(np.tanh(5*f)-np.tanh(f)) + 0.5*np.tanh(5*f) + 0.01*f
        # e = 10000*np.sign(f)*f**2
        e = 0.1*np.sign(f)*f**2

        return e

    @staticmethod
    def phi_cL(q):
        e = k1*q + k2*q**3
        # e = 100*(k1*q + k2*q**3)
        return e
    
    @staticmethod
    def phi_CA(V):
        P = (P_A_0/(1.0 - V/V_A_0)**1.4)
        # P = V + V**3

        return P
    
    @staticmethod
    def phi_CB(V):
        P = (P_B_0/(1.0 - V/V_B_0)**1.4)
        # P = V + 1.58*V**3
        return P

    # nonlinear state equations
    def f(self,t,x,u):

        x_dot = np.zeros(x.shape)

        V_A, V_B, x_p, p_p  = x[0], x[1], x[2], x[3]
        
        P_A = self.phi_CA(V_A)
        P_B = self.phi_CB(V_B)

        x_dot[0] =  self.phi_rA(P_A,u) - A*p_p/m + K_le*(P_B - P_A)
        x_dot[1] = -self.phi_rB(P_B,u) + A*p_p/m + K_le*(P_A - P_B)
        x_dot[2] = p_p/m
        x_dot[3] = (A*(P_A-P_B) - self.phi_cL(x_p) - self.phi_rL(p_p/m))

        return x_dot

    # nonlinear observation equations
    def g(self,t,x,u):
        # P_A, P_B, x_p, p_p, x_v  = x[0], x[1], x[2], x[3], x[4]
        eta = self.phi(t,x,u)
        y = np.concatenate((x,eta))
        return y 
    
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        V_A, V_B, x_p, p_p  = x[0], x[1], x[2], x[3]

        eta = np.zeros(self.N_eta)
        
        eta[0] = self.phi_CA(V_A)
        eta[1] = self.phi_CB(V_B) 
        eta[2] = self.phi_rA(self.phi_CA(V_A),u)
        eta[3] = self.phi_rB(self.phi_CB(V_B),u)
        eta[4] = self.phi_cL(x_p) 
        eta[5] = self.phi_rL(p_p/m)

        # if eta[0] > Ps:
        #     print(t, eta[0], )

        # if eta[1] > Ps:
        #     print(t,eta[1])

        return eta

    def phi_hybrid(self,t,x,u):
        '''
        outputs the values of the auxiliary variables
        '''
        eta = self.phi(t,x,u)
        eta_hybrid = self.P.dot(eta)

        return eta_hybrid 


#Forcing laws
def zero_u_func(y,t):
    return 1.2 

def rand_u_func(y,t):
    return np.random.uniform(low = -1., high = 1.) #10*np.random.normal(0.0,0.3)

def sin_u_func(y,t):
    return 0.1*np.sin(2*t) #0.2*signal.square(2*t)

# def darcy_f(v):
#     Re = rho*v*l/mu
#     a = 1/(1+ np.power(Re/2712,8.4))
#     b = 1/(1+ np.power(Re/(150*l/eps),8.4))
#     f = np.power(64/Re,a)*np.power(0.75*np.log(Re/5.37),(2*(a-1)*b))*np.power(0.88*np.log(l/eps),(2*(a-1)*(1-b)))
#     return f

# def e(v):
#     f = darcy_f(v)
#     dp = f*rho*v**2/l
#     return dp

def nme(y_gt,y_est):
    # y_gt,y_est = y_gt+50,y_est+50
    y_error = y_est - y_gt
    y_nme   = np.divide(np.mean(np.abs(y_error),axis = 0),
                        np.max(y_gt)- np.min(y_gt))
    # print('mean abs value:', np.mean(np.abs(y_gt)   ,axis = 0))
    # print('mean abs error:', np.mean(np.abs(y_error),axis = 0))
    # print('mean error norm:', y_nme  )
    # print('--------------------')

    return y_nme

    # return np.swapaxes(np.divide(np.sum(np.square(Y_test-mu_star),axis=0),
#                        np.sum(np.square(Y_test-np.expand_dims(np.tile(y_bar,(mu_star.shape[0],1)),axis=1)),axis=0)),0,1)

if __name__== "__main__":

    # ###
    # v = np.linspace(-1,1,100)
    # f = np.zeros(100)
    # for i in range(100):
    #     f[i] = PlantMinimal.phi_r_load(v[i])

    # plt.plot(v,f)
    # plt.show()
    # exit()

    run_example_1 = False
    run_example_2 = True
    plotting = False

    ############## NUMERICAL EXAMPLE 1 ########################
    if run_example_1:
        print('----------------------- running example 1 ----------------------- ')
        colours =['k','tab:blue','tab:orange','tab:green','tab:purple','tab:brown']
        # linestyles = ['-', '--', '-.', ':','solid']
        linestyles = ['solid','solid','solid','solid','solid','solid']


        I_F= 1.0
        I_L = 1.0
        a1,a2,a3 = 1.0,1.0,1.0
        k1,k2 = 1.0,5.0
        B_1 = 2.5

        N_test_i = 10
        N_test_j = 1

        xi_order_array = np.array([1,3,8])

        nme_dfl_sid_reduced = np.zeros((len(xi_order_array),3))
        nme_dfl_sid_full = 0
        nme_dfl_sid_bad = 0
        nme_sid = 0
        
        pl = PlantMinimal()
        dfl = DFL(pl, dt_data = 0.05, dt_control = 0.2)

        seeds = np.random.randint(10000, size =  N_test_i*N_test_j )

        T_range = 5.0

        for j in range(N_test_j):

            dfl.generate_data_from_random_trajectories(x_0 = np.array([0,0,0]), t_range_data = 5.0, n_traj_data = 50 ,plot_sample = False )
            # dfl.generate_hybrid_model(xi_order = pl.N_eta)
            dfl.generate_sid_model(xi_order = (pl.N_eta + pl.N_x))
          
            for i in range(N_test_i):


                x_0 = np.random.uniform(pl.x_min, pl.x_max)*0
                seed = seeds[i*j] 
                
                ########### nonlinear simulation
                np.random.seed(seed = seed)
                t, u_nonlin, x_nonlin, y_nonlin = dfl.simulate_system_nonlinear(x_0, rand_u_func, T_range)

                ########## N4SID simulation
                np.random.seed(seed = seed)
                t, u_sid, x_sid, y_sid = dfl.simulate_system_sid(x_0, rand_u_func, T_range)

                ###############################################################
                #To compare with the reduced set of measurements -> equivalence
                dfl.plant.P =  np.array([ [1., 0,  0,],
                                          [0,  1., 1.]])
                dfl.plant.A_cont_eta_hybrid =   pl.A_cont_eta.dot(np.linalg.pinv(dfl.plant.P))
                
                # fig, axs = plt.subplots(4, 1,figsize=(7, 5))
                # axs[0].plot(t, y_nonlin[:,0],colours[0],linestyle=linestyles[0])[0]# , label = 'True'
                # axs[1].plot(t, y_nonlin[:,1],colours[0],linestyle=linestyles[0])
                # axs[2].plot(t, y_nonlin[:,2],colours[0],linestyle=linestyles[0])
                # axs[3].plot(t, u_nonlin,colours[0],linestyle=linestyles[0])

                for k in range(len(xi_order_array)):
                    dfl.generate_hybrid_model(xi_order = xi_order_array[k])
                    np.random.seed(seed = seed)
                    t, u_reduced, x_reduced, y_reduced = dfl.simulate_system_hybrid(x_0, rand_u_func, T_range)
                    nme_dfl_sid_reduced[k,:] += nme(y_nonlin[:,:3],y_reduced[:,:3])

                    # axs[0].plot(t, y_reduced[:,0],colours[k+1],linestyle=linestyles[k+1])
                    # axs[1].plot(t, y_reduced[:,1],colours[k+1],linestyle=linestyles[k+1])
                    # axs[2].plot(t, y_reduced[:,2],colours[k+1],linestyle=linestyles[k+1])

                ###############################################################
                #To compare with the full set of measurements -> equivalence
                dfl.plant.P =  np.array([ [1., 0,  0,],
                                          [0,  1., 0.],
                                          [0,  0., 1.]])
                dfl.plant.A_cont_eta_hybrid =   pl.A_cont_eta.dot(np.linalg.pinv(dfl.plant.P))
                dfl.generate_hybrid_model(xi_order = pl.N_eta)
                np.random.seed(seed = seed)
                t, u_full, x_full, y_full = dfl.simulate_system_hybrid(x_0, rand_u_func, T_range)
             
                #To compare with a bad set
                dfl.plant.P =  np.array([ [1., 0,  0,],
                                          [0,  5., 1.]])
                dfl.plant.A_cont_eta_hybrid =   pl.A_cont_eta.dot(np.linalg.pinv(dfl.plant.P))
                dfl.generate_hybrid_model(xi_order = pl.N_eta)
                np.random.seed(seed = seed)
                t, u_bad, x_bad, y_bad = dfl.simulate_system_hybrid(x_0, rand_u_func, T_range)
             
                # add error to total errot
                # nme_dfl_sid_reduced += nme(y_nonlin[:,:3],y_reduced[:,:3])
                nme_dfl_sid_full    += nme(y_nonlin[:,:3],y_full[:,:3])
                nme_dfl_sid_bad     += nme(y_nonlin[:,:3],y_bad[:,:3])
                nme_sid             += nme(y_nonlin[:,:3],y_sid[:,:3])

                # plt.show()

                if plotting:
                    
                    plot_state, plot_auxiliary, plot_auxiliary_algebraic = True, False, True  

                    if i == 0 and j == 0:
                        if plot_state == True:
                            fig, axs = plt.subplots(4, 1,figsize=(7, 5))
                            # fig.suptitle('State variables', fontsize=16)
                            
                            # line_labels = ['Full Nonlinear System',
                            #                r'DFL-SID w/ sufficient measurments, $\mathit{M_{1}}$',  
                            #                r'DFL-SID w/ insufficient measurements, $\mathit{M_{1,bad}}$']

                            # line_labels = ['Full Nonlinear System', 'DFL-SID w/ reduced measurments',  'DFL-SID w/ full measurements']

                            line_labels = ['Full Nonlinear System',
                                           'DFL-SID',  
                                           'N4SID']


                            l1 = axs[0].plot(t, y_nonlin[:,0],colours[0],linestyle=linestyles[0])[0]# , label = 'True'
                            axs[1].plot(t, y_nonlin[:,1],colours[0],linestyle=linestyles[0])
                            axs[2].plot(t, y_nonlin[:,2],colours[0],linestyle=linestyles[0])
                            axs[3].plot(t, u_nonlin,colours[0],linestyle=linestyles[0])


                            l2 = axs[0].plot(t, y_reduced[:,0],colours[1],linestyle=linestyles[1])[0] #, label = 'DFL - N4SID modified measurments')
                            axs[1].plot(t, y_reduced[:,1],colours[1],linestyle=linestyles[1])
                            axs[2].plot(t, y_reduced[:,2],colours[1],linestyle=linestyles[1])
                            # axs[3].plot(t, u_hybrid,colours[1],linestyle=linestyles[1])

                            # l3 = axs[0].plot(t, y_hybrid_2[:,0],colours[2],linestyle=linestyles[2])[0] #, label = 'DFL - N4SID full measurements')
                            # axs[1].plot(t, y_hybrid_2[:,1],colours[2],linestyle=linestyles[2])
                            # axs[2].plot(t, y_hybrid_2[:,2],colours[2],linestyle=linestyles[2])
                            # axs[3].plot(t, u_hybrid_2,colours[2],linestyle=linestyles[2])
                            
                            # l3 = axs[0].plot(t, y_full[:,0],colours[2],linestyle=linestyles[2])[0] # , label = 'DFL - Discrete regression')
                            # axs[1].plot(t,      y_full[:,1],colours[2],linestyle=linestyles[2])
                            # axs[2].plot(t,      y_full[:,2],colours[2],linestyle=linestyles[2])

                            l3 = axs[0].plot(t, y_sid[:,0],colours[2],linestyle=linestyles[2])[0] # , label = 'DFL - Discrete regression')
                            axs[1].plot(t,      y_sid[:,1],colours[2],linestyle=linestyles[2])
                            axs[2].plot(t,      y_sid[:,2],colours[2],linestyle=linestyles[2])



                            # Set the y labels usin latex notation eg r'$\mathit{x_{rb}}$ (m)'
                            axs[0].set_ylabel(r'$\mathit{q_1}$')
                            axs[1].set_ylabel(r'$\mathit{p_L}$')
                            axs[2].set_ylabel(r'$\mathit{p_p}$')
                            axs[3].set_ylabel(r'$\mathit{u}$')
                            
                            axs[3].set_xlabel(r'$\mathit{t}$')

                            # add legend
                            leg = fig.legend(handles = [l1, l2, l3],     # The line objects
                               labels=line_labels,   # The labels for each line
                               loc="upper center",   # Position of legend
                               borderaxespad=0.1,
                               fontsize=10,
                               frameon=False)  

                            fig.align_ylabels()

                            # remove time from all but bottom subplot
                            axs[0].set_xticklabels([])
                            axs[1].set_xticklabels([])
                            axs[2].set_xticklabels([])

                            # add grid to each sublot
                            axs[0].grid()
                            axs[1].grid() 
                            axs[2].grid()
                            axs[3].grid()

                            plt.subplots_adjust(top=0.85,hspace = 0.1)

                            # axs[0].legend(loc='right')

                            # Put a legend to the right of the current axis
                            # axs[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))

                        if plot_auxiliary == True:
                            fig, axs = plt.subplots(3, 1)
                            fig.suptitle('Auxilliary variables', fontsize=16)
                            axs[0].plot(t, y_nonlin[:,3],'k', label = 'True')
                            axs[1].plot(t, y_nonlin[:,4],'k')
                            axs[2].plot(t, y_nonlin[:,5],'k')

                            # axs[0].plot(t, x_dfl[:,3],'r', label = 'DFL')
                            # axs[1].plot(t, x_dfl[:,4],'r')
                            # axs[2].plot(t, x_dfl[:,5],'r')

                            axs[0].set_ylabel('eRF')
                            axs[1].set_ylabel('eRL')
                            axs[2].set_ylabel('eCL')

                            axs[0].legend(loc='right')


                            # q_1, p_L, p_F = x[0],x[1],x[2]
                            
                            # f_1 = (1/I_F)*p_F - (1/I_L)*p_L
                            # f_f = (1/I_F)*p_F 

                            # eta = np.zeros(self.N_eta)

                            # eta[0] = self.phi_r_fluid(f_f)
                            # eta[1] = self.phi_r_load(f_1)
                            # eta[2] = self.phi_c_load(q_1)

                        if plot_auxiliary_algebraic == True:
                            fig, axs = plt.subplots(3, 1)
                            fig.suptitle('Auxilliary variables', fontsize=16)

                            axs[0].plot((1/I_F)*y_nonlin[:,2], y_nonlin[:,3],'k.', label = 'True')
                            axs[1].plot((1/I_F)*y_nonlin[:,2]- (1/I_L)*y_nonlin[:,1], y_nonlin[:,4],'k.')
                            axs[2].plot(y_nonlin[:,0], y_nonlin[:,5],'k.')

                            # axs[0].plot((1/I_F)*x_dfl[:,2], x_dfl[:,3],'r.', label = 'DFL')
                            # axs[1].plot((1/I_F)*x_dfl[:,2]- (1/I_L)*x_dfl[:,1], x_dfl[:,4],'r.')
                            # axs[2].plot(x_dfl[:,0], x_dfl[:,5],'r.')

                            axs[0].set_ylabel('eRF')
                            axs[1].set_ylabel('eRL')
                            axs[2].set_ylabel('eCL')

                            axs[0].legend(loc='right')
                        
                        if plot_state or plot_auxiliary or plot_auxiliary_algebraic:
                            plt.show()

        nme_dfl_sid_reduced *= 1/(N_test_i*N_test_j)
        nme_dfl_sid_full *= 1/(N_test_i*N_test_j)
        nme_dfl_sid_bad *= 1/(N_test_i*N_test_j)
        nme_sid *= 1/(N_test_i*N_test_j)

        # print(nme_dfl_sid_reduced)

        print('nme_dfl_sid_reduced:', np.mean(nme_dfl_sid_reduced,axis=1))
        print('nme_dfl_sid_full:'   , np.mean(nme_dfl_sid_full))
        print('nme_dfl_sid_bad:'    , np.mean(nme_dfl_sid_bad))
        print('nme_only_SID:', np.mean(nme_sid))

    ############## NUMERICAL EXAMPLE 2 ########################

    if run_example_2:
        print('----------------------- running example 2 ----------------------- ')
        Ps = 1
        Pt = 0.2
        A = 1.0 #0.05**2
        a = 1.0
        
        V_A_0 = 0.3
        V_B_0 = 0.3
        P_A_0 = 0.15
        P_B_0 = 0.1
        
        K_le = 0.001 #0.000001
        m = 0.1 #5
        Cv = 1.0 # 0.005
        # lam = 40
        a1,a2,a3 = 1.0,1.0,1.0
        k1,k2 = 1.0,5.0

        N_test_i = 5
        N_test_j = 5

        xi_order_array = np.array([1,4,10])

        nme_dfl_sid_reduced = 0
        nme_dfl_sid_full = 0
        nme_dfl_sid_bad = 0
        nme_sid = 0

        nme_dfl_sid_reduced = np.zeros((len(xi_order_array),4))
        
        pl = PlantNonCausal3()
        dfl = DFL(pl, dt_data = 0.02, dt_control = 0.4)
        
        seeds = np.random.randint(10000, size =  N_test_i*N_test_j )

        for j in range(N_test_j):

            x_0 = np.array([0.2225, 0.1175, 0.425, 0.0])
            dfl.generate_data_from_random_trajectories(x_0 = x_0, t_range_data = 5.0, n_traj_data = 100 , plot_sample = False )
            dfl.generate_hybrid_model(xi_order = pl.N_eta)
            dfl.generate_sid_model(xi_order = (pl.N_eta + pl.N_x))
          
            for i in range(N_test_i):

                # x_0 = np.random.uniform(pl.x_min, pl.x_max)
                x_0 = np.array([0.2225, 0.1175, 0.425, 0.0])

                seed = seeds[i*j]

                np.random.seed(seed = seed)
                t, u_nonlin, x_nonlin, y_nonlin = dfl.simulate_system_nonlinear(x_0, rand_u_func, 5.0)
                
                np.random.seed(seed = seed)
                t, u_sid, x_sid, y_sid = dfl.simulate_system_sid(x_0, rand_u_func, 5.0)

                #To compare with the reduced set of measurements -> equivalence
                dfl.plant.P = np.array([[1, -1, 0, 0, 0, 0],
                                        [0, 0,  1, 0, 0, 0],
                                        [0, 0,  0, 1, 0, 0],
                                        [0, 0,  0, 0, 1, 1]])
                dfl.plant.A_cont_eta_hybrid =   pl.A_cont_eta.dot(np.linalg.pinv(dfl.plant.P))

                for k in range(len(xi_order_array)):
                    dfl.generate_hybrid_model(xi_order = xi_order_array[k])
                    np.random.seed(seed = seed)
                    t, u_reduced, x_reduced, y_reduced = dfl.simulate_system_hybrid(x_0, rand_u_func, 5.0)
                    nme_dfl_sid_reduced[k,:] += nme(y_nonlin[:,:4],y_reduced[:,:4])

                #To compare with the full set of measurements -> equivalence
                dfl.plant.P =  np.array([[1, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1]])
                dfl.plant.A_cont_eta_hybrid =   pl.A_cont_eta.dot(np.linalg.pinv(dfl.plant.P))
                dfl.generate_hybrid_model(xi_order = pl.N_eta)
                np.random.seed(seed = seed)
                t, u_full, x_full, y_full = dfl.simulate_system_hybrid(x_0, rand_u_func, 5.0)
             
                #To compare with a bad set
                dfl.plant.P = np.array([[1, 0, 0, 0, 0, 0],
                                        [0, 0,  1, 0, 0, 0],
                                        [0, 0,  0, 1, 0, 0],
                                        [0, 0,  0, 0, 1, 1]])
                dfl.plant.A_cont_eta_hybrid =   pl.A_cont_eta.dot(np.linalg.pinv(dfl.plant.P))
                dfl.generate_hybrid_model(xi_order = pl.N_eta)
                np.random.seed(seed = seed)
                t, u_bad, x_bad, y_bad = dfl.simulate_system_hybrid(x_0, rand_u_func, 5.0)
                # # add error to total error
                # nme_dfl_sid_reduced += nme(y_nonlin[:,:4],y_reduced[:,:4])
                nme_dfl_sid_full    += nme(y_nonlin[:,:4],y_full[:,:4])
                nme_dfl_sid_bad     += nme(y_nonlin[:,:4],y_bad[:,:4])
                nme_sid             += nme(y_nonlin[:,:4],y_sid[:,:4])

                if plotting:
                    plot_state, plot_auxiliary = True, True
                    plot_auxiliary_algebraic = True

                    # line_labels = ['Full Nonlinear System', 'DFL-SID w/ reduced measurments',  'DFL-SID w/ full measurements']
                    
                    line_labels = ['Full Nonlinear System',
                                   r'DFL-SID w/ sufficient measurments, $\mathit{M_{2}}$',  
                                   r'DFL-SID w/ insufficient measurements, $\mathit{M_{2,bad}}$']

                    colours = ['k','tab:blue','tab:orange','tab:green','tab:purple']
                    linestyles = ['-', '--', '-.', ':']

                    if plot_state == True:
                        fig, axs = plt.subplots(5, 1, figsize=(7, 6))

                        l1 = axs[0].plot(t, y_nonlin[:,0],colours[0],linestyle=linestyles[0])[0]
                        axs[1].plot(t,      y_nonlin[:,1],colours[0],linestyle=linestyles[0])
                        axs[2].plot(t,      y_nonlin[:,2],colours[0],linestyle=linestyles[0])
                        axs[3].plot(t,      y_nonlin[:,3],colours[0],linestyle=linestyles[0])
                        axs[4].plot(t,      u_nonlin,     colours[0],linestyle=linestyles[0])
                      
                        l2 = axs[0].plot(t, y_reduced[:,0],colours[1],linestyle=linestyles[1])[0] 
                        axs[1].plot(t,      y_reduced[:,1],colours[1],linestyle=linestyles[1])
                        axs[2].plot(t,      y_reduced[:,2],colours[1],linestyle=linestyles[1])
                        axs[3].plot(t,      y_reduced[:,3],colours[1],linestyle=linestyles[1])

                        # l3 = axs[0].plot(t, y_full[:,0],colours[2],linestyle=linestyles[2])[0]
                        # axs[1].plot(t,      y_full[:,1],colours[2],linestyle=linestyles[2])
                        # axs[2].plot(t,      y_full[:,2],colours[2],linestyle=linestyles[2])
                        # axs[3].plot(t,      y_full[:,3],colours[2],linestyle=linestyles[2])

                        l3 = axs[0].plot(t, y_bad[:,0],colours[2],linestyle=linestyles[2])[0]
                        axs[1].plot(t,      y_bad[:,1],colours[2],linestyle=linestyles[2])
                        axs[2].plot(t,      y_bad[:,2],colours[2],linestyle=linestyles[2])
                        axs[3].plot(t,      y_bad[:,3],colours[2],linestyle=linestyles[2])



                        axs[0].set_ylabel(r'$\mathit{q_A}$')
                        axs[1].set_ylabel(r'$\mathit{q_B}$')
                        axs[2].set_ylabel(r'$\mathit{q_p}$')
                        axs[3].set_ylabel(r'$\mathit{p_p}$')
                        axs[4].set_ylabel(r'$\mathit{u}$')

                        axs[4].set_xlabel(r'$t$')

                        # add grid to each sublot
                        for j in range(len(axs)):
                            axs[j].grid()
                            if j != len(axs)-1:
                                axs[j].set_xticklabels([])

                        fig.align_ylabels()

                        leg = fig.legend(handles = [l1, l2, l3],     # The line objects
                                       labels=line_labels,   # The labels for each line
                                       loc="upper center",   # Position of legend
                                       borderaxespad=0.1,
                                       fontsize=10,
                                       frameon=False)  

                        plt.subplots_adjust(left=0.2, top=0.89, hspace = 0.15)

                    if plot_auxiliary == True:
                        fig, axs = plt.subplots(7, 1)
                        fig.suptitle('Auxilliary variables', fontsize=16)
                        axs[0].plot(t, y_nonlin[:,4],'k', label = 'True')
                        axs[1].plot(t, y_nonlin[:,5],'k')
                        axs[2].plot(t, y_nonlin[:,6],'k')
                        axs[3].plot(t, y_nonlin[:,7],'k')
                        axs[4].plot(t, y_nonlin[:,8],'k')
                        axs[5].plot(t, y_nonlin[:,9],'k')
                        axs[6].plot(t, y_nonlin[:,4]-y_nonlin[:,5],'k')

                        axs[0].plot(t, y_full[:,4],colours[1],linestyle=linestyles[1], label = 'True')
                        axs[1].plot(t, y_full[:,5],colours[1],linestyle=linestyles[1])
                        axs[2].plot(t, y_full[:,6],colours[1],linestyle=linestyles[1])
                        axs[3].plot(t, y_full[:,7],colours[1],linestyle=linestyles[1])
                        axs[4].plot(t, y_full[:,8],colours[1],linestyle=linestyles[1])
                        axs[5].plot(t, y_full[:,9],colours[1],linestyle=linestyles[1])
                        axs[6].plot(t, y_full[:,4]-y_full[:,5],colours[1],linestyle=linestyles[1])

                        #  eta = P_A, P_B, Q_A, Q_B, eCL, eRL]
                        axs[0].set_ylabel('P_A')
                        axs[1].set_ylabel('P_B')
                        axs[2].set_ylabel('Q_A')
                        axs[3].set_ylabel('Q_B')
                        axs[4].set_ylabel('eCL')
                        axs[5].set_ylabel('eRL')

                    if plot_auxiliary_algebraic == True:

                        fig, axs = plt.subplots(6, 1)
                        fig.suptitle('Auxilliary variables', fontsize=16)
                        nx = 4
                        axs[0].plot(y_nonlin[:,0], y_nonlin[:,4],'k.', label = 'True')
                        axs[1].plot(y_nonlin[:,1], y_nonlin[:,5],'k.', label = 'True')
                        axs[2].plot(y_nonlin[:,4], y_nonlin[:,6],'k.', label = 'True')
                        axs[3].plot(y_nonlin[:,5], y_nonlin[:,7],'k.', label = 'True')
                        axs[4].plot(y_nonlin[:,2], y_nonlin[:,8],'k.', label = 'True')
                        axs[5].plot(y_nonlin[:,3], y_nonlin[:,9],'k.', label = 'True')

                        for j in range(len(axs)):
                            axs[j].grid()
                            axs[j].set_xticklabels([])


                    axs[0].legend(loc='right')
                            


                    plt.show()

        nme_dfl_sid_reduced *= 1/(N_test_i*N_test_j)
        nme_dfl_sid_full *= 1/(N_test_i*N_test_j)
        nme_dfl_sid_bad *= 1/(N_test_i*N_test_j)
        nme_sid *= 1/(N_test_i*N_test_j)

        print('nme_dfl_sid_reduced:', np.mean(nme_dfl_sid_reduced, axis = 1))
        print('nme_dfl_sid_full:'   , np.mean(nme_dfl_sid_full))
        print('nme_dfl_sid_bad:'    , np.mean(nme_dfl_sid_bad))
        print('nme_only_SID:', np.mean(nme_sid))