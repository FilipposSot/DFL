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

class Plant1(DFLDynamicPlant):
    
    def __init__(self):

        # Linear part of states matrices
        self.n_x = 5
        self.n_eta = 3
        self.n_u = 2
        self.n = self.n_x + self.n_eta

        # User defined matrices for DFL
        self.A_cont_x  = np.array([[ 0., 0., 1., 0., 0.],
                                   [ 0., 0., 0., 1., 0.],
                                   [ 0., 0., 0., 0., 0.],
                                   [ 0., 0., 0., 0., 0.],
                                   [ 0., 0., 0., 0., 0.]])

        self.A_cont_eta = np.array([[ 0., 0., 0.],
                                    [ 0., 0., 0.],
                                    [ 1., 0., 0.],
                                    [ 0., 1., 0.],
                                    [ 0., 0., 1.]])

        self.B_cont_x = np.array([[0.0, 0.0],
                                  [0.0, 0.0],
                                  [1.0, 0.0],
                                  [0.0, 1.0],
                                  [0.0, 0.0],])

        self.x_min = np.array([-1.0,-1.0, 0.0,-3.0, 0.0])
        self.x_max = np.array([ 2.0, 0.0, 5.0, 3.0, 5.0])
        self.u_min = np.array([ 0.05,-0.75])
        self.u_max = np.array([ 0.5,-0.1])

    def generate_random_surface(self, x_min = 0.0, x_max = 1.0):

        x = np.linspace(x_min,x_max,100)
        y = np.random.uniform(-0.5, 0.5, 1)

        for i in range(7):
            heap_height = np.random.uniform(0.02, 0.07, 1)
            heap_sigma  = np.random.uniform(0.05, 0.1, 1)
            
            x_c = np.random.uniform(low = np.amin(x), high = np.amax(x), size = 1)
            surf_heap_i = heap_height*np.exp(-np.square(x-x_c)/heap_sigma**2)
            y = y + np.sign(np.random.uniform(-1,2,1))*surf_heap_i

        y = y - y[0]

        return x,y 

    def set_soil_surf(self, x, y):

        self.tck_sigma = splrep(x, y, s = 0)

    def soil_surf_eval(self, x):
        # Evaluate the spline soil surface and its derivatives
        
        surf     = splev(x, self.tck_sigma, der = 0, ext=3)
        surf_d   = splev(x, self.tck_sigma, der = 1, ext=3)
        surf_dd  = splev(x, self.tck_sigma, der = 2, ext=3)
        surf_ddd = splev(x, self.tck_sigma, der = 3, ext=3)

        return surf, surf_d, surf_dd, surf_ddd

    def draw_soil(self, ax, x_min, x_max):
        # for now lets keep this a very simple path: a circle
        x = np.linspace(x_min,x_max, 200)
        y = np.zeros(x.shape)
        
        for i in range(len(x)):
            y[i],_,_,_ = self.soil_surf_eval(x[i])
        
        ax.plot(x, y, 'k--')

        return ax

    def get_I(self, gamma, D):
        '''
        Calculates variable system inertia
        '''
        return np.diag([1.,1.])


    @staticmethod
    def Phi_soil(D, x, z, v_x, v_z, gamma):
        '''
        place hold soil force
        will be replaced by FEE
        '''
        F =     (-D**2 - D*gamma - 10*D*np.sign(v_x)*v_x**2)*np.array([1.0,0.0])
        F = F + ( D + D**3 + -10*D*v_z)*np.array([0.0,1.0])
        
        # F =     (-D**2 - 10*D*v_x)*np.array([1.0,0.0])
        # F = F + ( D + D**3 + -10*D*v_z)*np.array([0.0,1.0])

        # F = (-D**2 -10*D*v_x)*np.array([1.0,0.0])
        # F = F + ( D + 5*D**3 + -10*D*v_z)*np.array([0.0,1.0])
        # F = np.array([0,D+D**3]) + -5*D*np.array([0.0,1.0])*v_z # max(v_x,0)*(np.abs(np.array([v_x,v_z])*D)) +
        # F = F + 5*max(v_x,0)*-D*np.array([1.0,0.0]) - 0.5*D*np.array([1.0,0.0])

        return F



    # nonlinear state equations
    def f(self,t,xi,u):

        eps_vx = 0.001
        F_scale = 0.001

        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        s, s_dash, s_dash_dash, s_dash_dash_dash = self.soil_surf_eval(x)
        D = s-z
        I = self.get_I(gamma,D)

        x_dot = v_x
        z_dot = v_z
        F_soil = self.Phi_soil(D, x, z, v_x, v_z, gamma, u)
        # F_noise =  np.random.normal(np.array([0.0,0.0]),np.array([0.01,0.01]))
        F_net = F_scale*F_soil + u
        
        # if F_net[0]<=0:
        #     F_net[0] = 0

        # if v_z<=0:
        #     F_net[1] = 0
        
        v_dot = np.linalg.inv(I).dot(F_net) 

        gamma_dot = v_x*D

        xi_dot = np.array([x_dot, z_dot, v_dot[0], v_dot[1], gamma_dot]) 

        return xi_dot

    # nonlinear observation equations
    def g(self,x,u,t):   
        return x

    # observation including soil surface shape
    def g_state_and_surface(self,t,xi,u):
        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        s, s_dash, s_dash_dash, s_dash_dash_dash = self.soil_surf_eval(x)
        D = s-z
        # F = self.Phi_soil(D,x, z,v_x,v_z)
        y = np.array([x, z, v_x, v_z, gamma])

        return y  
    
    # observation including soil surface shape
    def g_state_and_auxiliary(self,t,xi,u):
        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        eta = self.phi(t,xi,u)
        y = np.concatenate((xi,eta))

        return y  

    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,xi,u):
        '''
        outputs the values of the auxiliary variables
        '''
        eta = np.zeros(self.n_eta)
        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        s, s_dash, s_dash_dash, s_dash_dash_dash = self.soil_surf_eval(x)
        D = s - z
        F_soil = self.Phi_soil(D, x, z, v_x, v_z, gamma, u)
        
        eta[0] = F_soil[0]
        eta[1] = F_soil[1]
        eta[2] = v_x*D

        return eta

class PlantMinimal(DFLDynamicPlant):
    # simple piston and resistive fluid load
    # Single nonlinear load
    def __init__(self):
        
        # Structure of system
        self.N_x = 4
        self.N_eta = 4
        self.N_u = 2

        # Combined system order
        self.N = self.N_x + self.N_eta

        # User defined matrices for DFL
        # User defined matrices for DFL
        self.A_cont_x  = np.array([[ 0., 0., 1., 0.],
                                   [ 0., 0., 0., 1.],
                                   [ 0., 0., 0., 0.],
                                   [ 0., 0., 0., 0.]])

        self.A_cont_eta = np.array([[ 0.,  0.,   0.,  0.],
                                    [ 0.,  0.,   0.,  0.],
                                    [ -1., -1.,  0.,  0.],
                                    [ 0. ,  0., -1., -1.]])

        self.B_cont_x = np.array([[0.0, 0.0],
                                  [0.0, 0.0],
                                  [1.0, 0.0],
                                  [0.0, 1.0]])

       
        # Limits for inputs and states
        self.x_min = np.array([-1.0,-1.0, 0.0,-3.0])
        self.x_max = np.array([ 2.0, 0.0, 5.0, 3.0])
        self.u_min = np.array([ 0.05,-0.75])
        self.u_max = np.array([ 0.5, -0.1])

        
        # Hybrid model definition
        self.N_eta_hybrid = 2
        
        self.P =  np.array([[1, 1, 0, 0],
                            [0, 0, 1, 1]])

        self.A_cont_eta_hybrid =   self.A_cont_eta.dot(np.linalg.pinv(self.P))

   # functions defining constituitive relations for this particular system
    @staticmethod
    def phi_r_load_x(f):
        # friction like nonlinearity
        # e = np.sign(f)*f**2
        e = 0.25*(np.tanh(5*f)-np.tanh(f)) + 0.5*np.tanh(5*f) + 0.01*f
        return e

    @staticmethod
    def phi_c_load_x(q):
        thresh = 0.1

        if np.abs(q) < thresh:
            e = k1*q
        else:
            e = k2*q -(k2 - k1)*thresh*np.sign(q)

        return e

    @staticmethod
    def phi_r_load_z(f):
        # friction like nonlinearity
        # e = np.sign(f)*f**2
        e = 0.25*(np.tanh(5*f)-np.tanh(f)) + 0.5*np.tanh(5*f) + 0.01*f
        return e

    @staticmethod
    def phi_c_load_z(q):
        thresh = 0.1

        e = k1*q + k2*q**3
        # e = -0.01*np.sign(q)*q**2
        # if np.abs(q) < thresh:
        #     e = k1*q
        # else:
        #     e = k2*q -(k2 - k1)*thresh*np.sign(q)

        return e

    def get_I(self):
        '''
        Calculates variable system inertia
        '''
        return np.diag([1.,1.])

    # nonlinear state equations
    def f(self,t,xi,u):

        x, z, v_x, v_z  = xi[0], xi[1], xi[2], xi[3]
        I = self.get_I()

        x_dot = v_x
        z_dot = v_z
        
        F_r_x = self.phi_r_load_x(v_x)
        F_c_x = self.phi_c_load_x(x)
        F_r_z = self.phi_r_load_z(v_z)
        F_c_z = self.phi_c_load_z(z)

        F_soil = np.array([F_r_x+F_c_x , F_r_z+ F_c_z])

        F_net = -F_soil + u
        
        v_dot = np.linalg.inv(I).dot(F_net) 

        xi_dot = np.array([x_dot, z_dot, v_dot[0], v_dot[1]]) 

        return xi_dot



    # nonlinear observation equations
    def g(self,t,xi,u):
        x_t, z_t, v_x, v_z  = xi[0], xi[1], xi[2], xi[3]
        eta = self.phi(t,xi,u)
        x = np.array([ x_t, z_t, v_x, v_z])
        y = np.concatenate((x,eta))

        return y 
     
    # auxiliary variables (outputs from nonlinear elements)
    def phi(self,t,xi,u):
        '''
        outputs the values of the auxiliary variables
        '''
        x, z, v_x, v_z  = xi[0], xi[1], xi[2], xi[3]
        
        eta = np.zeros(self.N_eta)

        eta[0] = self.phi_r_load_x(v_x)
        eta[1] = self.phi_c_load_x(x)
        eta[2] = self.phi_r_load_z(v_z)
        eta[3] = self.phi_c_load_z(z)
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

def rand_u_func(y,t):
    
    u_min = np.array([ 0.05,-0.75])
    u_max = np.array([ 0.5, -0.1])

    if y[0]> -0.05:
        u_t =  np.random.uniform(low = u_min , high = u_max ) #+ np.array([0.0,0.7])
    else:
        u_t =  np.random.uniform(low = u_min , high = u_max)

    return u_t


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

    run_example_1 = True
    plotting = True

    ############## NUMERICAL EXAMPLE 1 ########################
    if run_example_1:
        print('----------------------- running example 1 ----------------------- ')
        colours =['k','tab:blue','tab:orange','tab:green','tab:purple','tab:brown']
        linestyles = ['-', '--', '-.', ':','solid']
        # linestyles = ['solid','solid','solid','solid','solid','solid']


        I_F= 1.0
        I_L = 1.0
        a1,a2,a3 = 1.0,1.0,1.0
        k1,k2 = 1.0,5.0
        B_1 = 2.5

        N_test_i = 1
        N_test_j = 1

        xi_order_array = np.array([4])

        nme_dfl_sid_reduced = np.zeros((len(xi_order_array),3))
        nme_dfl_sid_full = 0
        nme_dfl_sid_bad = 0
        nme_sid = 0
        
        pl = PlantMinimal()
        dfl = DFL(pl, dt_data = 0.05, dt_control = 0.2)

        seeds = np.random.randint(10000, size =  N_test_i*N_test_j )

        T_range = 5.0

        for j in range(N_test_j):

            dfl.generate_data_from_random_trajectories(x_0 = np.array([0,0,0,0]), t_range_data = 5.0, n_traj_data = 50 ,plot_sample = True )
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
                dfl.plant.P = np.array([[1, 1, 0, 0],
                                        [0, 0, 1, 1]])
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
                dfl.plant.P =  np.array([ [1., 0,  0,  0],
                                          [0,  1., 0., 0],
                                          [0,  0., 1., 0],
                                          [0,  0., 0,  1.]])
                dfl.plant.A_cont_eta_hybrid =   pl.A_cont_eta.dot(np.linalg.pinv(dfl.plant.P))
                dfl.generate_hybrid_model(xi_order = pl.N_eta)
                np.random.seed(seed = seed)
                t, u_full, x_full, y_full = dfl.simulate_system_hybrid(x_0, rand_u_func, T_range)
             
                #To compare with a bad set
                dfl.plant.P =  np.array([[1, 2, 0, 0],
                                         [0, 0, 1, 2]])
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
                            fig, axs = plt.subplots(5, 1,figsize=(7, 5))
                            # fig.suptitle('State variables', fontsize=16)
                            
                            # line_labels = ['Full Nonlinear System',
                            #                r'DFL-SID w/ sufficient measurments, $\mathit{M_{1}}$',  
                            #                r'DFL-SID w/ insufficient measurements, $\mathit{M_{1,bad}}$']

                            # line_labels = ['Full Nonlinear System',
                            #                'DFL-SID w/ reduced measurments',
                            #                'DFL-SID w/ full measurements']

                            line_labels = ['Full Nonlinear System',
                                           'DFL-SID',  
                                           'N4SID']

                            l1 = axs[0].plot(t, y_nonlin[:,0],colours[0],linestyle=linestyles[0])[0]# , label = 'True'
                            axs[1].plot(t, y_nonlin[:,1],colours[0],linestyle=linestyles[0])
                            axs[2].plot(t, y_nonlin[:,2],colours[0],linestyle=linestyles[0])
                            axs[3].plot(t, y_nonlin[:,3],colours[0],linestyle=linestyles[0])
                            axs[4].plot(t, u_nonlin[:,0,0],colours[0],linestyle=linestyles[0])
                            axs[4].plot(t, u_nonlin[:,0,1],colours[0],linestyle=linestyles[0])


                            l2 = axs[0].plot(t, y_reduced[:,0],colours[1],linestyle=linestyles[1])[0] #, label = 'DFL - N4SID modified measurments')
                            axs[1].plot(t, y_reduced[:,1],colours[1],linestyle=linestyles[1])
                            axs[2].plot(t, y_reduced[:,2],colours[1],linestyle=linestyles[1])
                            axs[3].plot(t, y_reduced[:,3],colours[1],linestyle=linestyles[1])

                            # axs[3].plot(t, u_hybrid,colours[1],linestyle=linestyles[1])

                            # l3 = axs[0].plot(t, y_hybrid_2[:,0],colours[2],linestyle=linestyles[2])[0] #, label = 'DFL - N4SID full measurements')
                            # axs[1].plot(t, y_hybrid_2[:,1],colours[2],linestyle=linestyles[2])
                            # axs[2].plot(t, y_hybrid_2[:,2],colours[2],linestyle=linestyles[2])
                            # axs[3].plot(t, u_hybrid_2,colours[2],linestyle=linestyles[2])
                            
                            # l3 = axs[0].plot(t,  y_full[:,0],colours[2],linestyle=linestyles[2])[0] # , label = 'DFL - Discrete regression')
                            # axs[1].plot(t,  y_full[:,1],colours[2],linestyle=linestyles[2])
                            # axs[2].plot(t,  y_full[:,2],colours[2],linestyle=linestyles[2])
                            # axs[3].plot(t,  y_full[:,3],colours[2],linestyle=linestyles[2])

                            l3 = axs[0].plot(t, y_sid[:,0],colours[2],linestyle=linestyles[2])[0] # , label = 'DFL - Discrete regression')
                            axs[1].plot(t,      y_sid[:,1],colours[2],linestyle=linestyles[2])
                            axs[2].plot(t,      y_sid[:,2],colours[2],linestyle=linestyles[2])
                            axs[3].plot(t,      y_sid[:,3],colours[2],linestyle=linestyles[2])

                            # Set the y labels usin latex notation eg r'$\mathit{x_{rb}}$ (m)'
                            axs[0].set_ylabel(r'$\mathit{x}$')
                            axs[1].set_ylabel(r'$\mathit{z}$')
                            axs[2].set_ylabel(r'$\mathit{v_x}$')
                            axs[3].set_ylabel(r'$\mathit{v_z}$')
                            axs[4].set_ylabel(r'$\mathit{u}$')

                            axs[4].set_xlabel(r'$\mathit{t}$')

                            # # add legend
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