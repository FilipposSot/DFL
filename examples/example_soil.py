#!/usr/bin/env python

import numpy as np
from dfl.dfl.dfl_soil import *
from dfl.dfl.mpcc import *

from dfl.dfl.dynamic_system import *
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splrep, splev, splint

# T_RANGE_DATA = 1.0
# DT_DATA = 0.05
# N_TRAJ_DATA = 20
# X_INIT_MIN = np.array([0.0,0.0,0.0])
# X_INIT_MAX = np.array([1.0,1.0,1.0])
def cot(x):
    return 1.0/np.tan(x)

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

    # @staticmethod
    # def Phi_soil(D, x, z, v_x, v_z, q, u):
    #     '''
    #     place hold soil force
    #     will be replaced by FEE
    #     '''
    #     # This is the rake angle
    #     rho = np.pi/3

    #     theta_v = np.arctan2(v_z,v_x)
    #     v_mag = np.sqrt(v_x**2 + v_z**2)

    #     # added propeties:
    #     K_stiff = 100
    #     B_pen = 100

    #     # These are soil-tool properties
    #     gamma = 1.0
    #     g = 1.0
    #     c = 1.0
    #     c_a = 1.0

        
        
    #     # F =     (-D**2 - 10*D*v_x)*np.array([1.0,0.0])
    #     # F = F + ( D + D**3 + -10*D*v_z)*np.array([0.0,1.0])

    #     # F = (-D**2 -10*D*v_x)*np.array([1.0,0.0])
    #     # F = F + ( D + 5*D**3 + -10*D*v_z)*np.array([0.0,1.0])
    #     # F = np.array([0,D+D**3]) + -5*D*np.array([0.0,1.0])*v_z # max(v_x,0)*(np.abs(np.array([v_x,v_z])*D)) +
    #     # F = F + 5*max(v_x,0)*-D*np.array([1.0,0.0]) - 0.5*D*np.array([1.0,0.0])

    #     return F

    @staticmethod
    def Phi_soil(D, x, z, v_x, v_z, gamma):
        '''
        place hold soil force
        will be replaced by FEE
        '''
        # F =     (-D**2 - D*gamma - 10*D*np.sign(v_x)*v_x**2)*np.array([1.0,0.0])
        F = (-D**2 -10*D*v_x)*np.array([1.0,0.0])
        F = F + ( D + D**3 + -10*D*v_z)*np.array([0.0,1.0])
        
        # F =     (-D**2 - 10*D*v_x)*np.array([1.0,0.0])
        # F = F + ( D + D**3 + -10*D*v_z)*np.array([0.0,1.0])

        # F = (-D**2 -10*D*v_x)*np.array([1.0,0.0])
        # F = F + ( D + 5*D**3 + -10*D*v_z)*np.array([0.0,1.0])
        # F = np.array([0,D+D**3]) + -5*D*np.array([0.0,1.0])*v_z # max(v_x,0)*(np.abs(np.array([v_x,v_z])*D)) +
        # F = F + 5*max(v_x,0)*-D*np.array([1.0,0.0]) - 0.5*D*np.array([1.0,0.0])

        return F


    # @staticmethod
    # def Phi_soil(d, x, z, v_x, v_z, q, u):
    #     '''
    #     place hold soil force
    #     will be replaced by FEE
    #     '''
    #     # this is the 
    #     q = 0.01*q

    #     # this is the direction of motion
    #     theta = np.arctan2(v_z, v_x)


    #     beta  = np.pi/6
    #     delta = np.pi/6
    #     phi   = np.pi/4
        
    #     gamma = 3.
    #     g = 1.0
    #     w = 1.0

    #     c = 1.0
    #     C_a = 1.0

    #     a    = w*d/(np.cos(beta + delta) + np.sin(beta + delta)*cot(rho + phi))
    #     N_g  = 0.5*(cot(beta)+cot(rho))
    #     N_q  = cot(beta) + cot(rho)
    #     N_c  = 1 + cot(rho)*np.cos(rho + phi)
    #     N_ca = 1 - cot(beta)*cot(rho + phi)
    #     N_a  = (np.tan(rho) + cot(rho + phi))/(1 + np.tan(rho)*cot(beta))
        
    #     # print(a, N_g, N_q, N_c, N_ca, N_a)

    #     T = a*(N_g*gamma*g*d + N_q*q + N_c*c + N_ca*C_a + N_a*gamma*v_x**2) 

    #     # print(np.array([a*(N_g*gamma*g*d), a*(N_q*q), a*(N_c*c), a*(N_ca*C_a),a*(N_a*gamma*v_x**2)]))

    #     F = T*np.array([-np.sin(beta + delta), np.cos(beta + delta)])

    #     # F[0] = -d
    #     # F[1] = 0.1*d - 0.01*v_z
    #     # print(F)

        # return F

    # @staticmethod
    # def Phi_soil(d, x, z, v_x, v_z, q, u):
    #     '''
    #     place hold soil force
    #     will be replaced by FEE
    #     '''
    #     # This is the rake angle
    #     rho = np.pi/3

    #     theta_v = np.arctan2(v_z,v_x)
    #     v_mag = np.sqrt(v_x**2 + v_z**2)

    #     # added propeties:
    #     K_stiff = 100
    #     B_pen = 100

    #     theta_v_min =  -rho
    #     theta_v_compression = -(80./180.)*np.pi
    #     theta_v_max = np.pi/6

    #     # Soil angle
    #     alpha = 0.0



    #     # These are soil-tool properties
    #     beta = np.pi/4  # failure angle
    #     phi = np.pi/6 # soil internal friction angle
    #     delta = np.pi/6 # soil tool friction angle
    #     gamma = 1.0
    #     g = 1.0
    #     c = 1.0
    #     c_a = 1.0
    #     # calculate modified FEE coefficients
    #     denom_common = np.sin(delta+rho+phi+beta)
        
    #     N_gamma = (cot(rho) + cot(beta))*np.sin(alpha+phi+beta)/(2*denom_common)
    #     N_q = np.sin(alpha+phi+beta)/(denom_common)
    #     N_c = np.cos(phi)/(np.sin(beta)*denom_common)
    #     N_a = -np.cos(rho+phi+beta)
    #     # N_i = (np.tan(beta) + cot(beta + phi))/(1 + np.tan(beta)*cot(rho))

    #     F =  N_gamma*gamma*g*d**2 + N_c*c*d + N_q*g*q + N_a*c_a*d

    #     F_x = -F*(np.cos(delta)*np.sin(rho-alpha) - np.sin(delta)*np.cos(rho-alpha))
    #     F_z = -F*(np.cos(delta)*np.cos(rho-alpha) + np.sin(delta)*np.sin(rho-alpha))



    #     #depending on direction of motion
    #     if theta_v > theta_v_min and theta_v < theta_v_max: # within FEE range
    #         print("in FEE region")
    #         pass

    #     elif theta_v > theta_v_max: # moving upwards 
    #         print("moving upwards")
    #         F_z = -q*g*gamma

    #     elif theta_v < theta_v_min:
    #         F_z = -B_pen*v_z + d

    #     elif theta_v < theta_v_min: # compressing soil
    #         print("moving downwards")
    #         F_z = -B_pen*v_z - u[1] #cancel force and slow down

    #     if F_z<

    #     F_r = np.array([F_x,F_z])
    #     print(F_r)

    #     return F_r


    # nonlinear state equations
    def f(self,t,xi,u):

        eps_vx = 0.001
        F_scale = 1.0

        x, z, v_x, v_z, gamma = xi[0], xi[1], xi[2], xi[3], xi[4]
        s, s_dash, s_dash_dash, s_dash_dash_dash = self.soil_surf_eval(x)
        D = s-z
        I = self.get_I(gamma,D)

        x_dot = v_x
        z_dot = v_z
        F_soil = self.Phi_soil(D, x, z, v_x, v_z, gamma)
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
        F_soil = self.Phi_soil(D, x, z, v_x, v_z, gamma)
        
        eta[0] = F_soil[0]
        eta[1] = F_soil[1]
        eta[2] = v_x*D

        return eta

def zero_u_func(y,t):
    u = np.array([0.0,0.0])
    if t < 2:
        u[0] = 1.0
        u[1] = -1.0
    if t >= 2:
        u[0] = 2.0
        u[1] = -1.0
    if t >= 2:
        u[0] = 1.0
        u[1] = -0.2
    return u

def control_u_func(y,t):
    u = np.array([0.0,0.0])
    # print(u)
    u[0] =  1.0
    u[1] = -0.5
    return u

# def rand_u_func 
#                     if D_t > 0.15:
#                         u_t =  np.random.uniform(low = self.plant.u_min , high = self.plant.u_max + np.array([0.0,0.7]))
#                     else:
#                         u_t =  np.random.uniform(low = self.plant.u_min , high = self.plant.u_max)

                # these are the inherent

if __name__== "__main__":

    # initialize the excavation system plant
    plant = Plant1()


    # generate a soil surface to test on. add it to the plant
    x_soil_points = np.array([0,0.25, 0.5,0.75 ,1. ,1.25, 1.5])
    y_soil_points = np.array([0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0])
    plant.set_soil_surf(x_soil_points, y_soil_points)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    # for i in range(5):

    #     x, y = plant.generate_random_surface()
    #     y = y - y[0]
    #     plt.plot(x, y, color='tab:blue')

    # ax.axis('equal')
    # plt.show()
    # exit()



    # create the dfl object with the digging plant
    dfl = DFLSoil(plant)
    
    # # set the observation to include soil shape parameters
    # setattr(plant, "g", plant.g_state_and_surface)
    
    # set a fixed initial position (zero)
    xi_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    t_f = 15.0
    t, u, x_nonlin, y_nonlin= dfl.simulate_system_nonlinear(xi_0, control_u_func, t_f)

    print(x_nonlin)

    # # generare training data
    dfl.generate_data_from_random_trajectories(t_range_data = 5.0,
                                               n_traj_data = 20,
                                               x_0 = xi_0,
                                               plot_sample = True)
    
    # regress the dfl (hybrid) soil model
    dfl.regress_model_no_surface()

    # generate a soil surface to test on. add it to the plant
    x_soil_points = np.array([0,0.25, 0.5,0.75,1.,1.25,1.5])
    y_soil_points = np.array([0, 0.1, 0.2,0.3 ,0.4,0.5,0.6])

    x_soil_points, y_soil_points = plant.generate_random_surface()
    plant.set_soil_surf(x_soil_points, y_soil_points)
    
    # define the path to be followed by MPCC
    x_path = np.array([0., 0.25, 0.5, 0.75, 1.])
    y_path = np.array([0.,-0.1 ,-0.1, -0.1 , 0.])
    spl_path = spline_path(x_path,y_path)

    # define the MPCC constraints (states and auxiliary variables)
    # need to find a more elegant way to define constraints on auxiliary variables
    x_min = np.concatenate((plant.x_min, np.array([-10.,-10.,-10.])))
    x_max = np.concatenate((plant.x_max, np.array([ 10., 10., 10.])))

    # define input constraints on the system
    u_min = np.array([0.0, -0.4])
    u_max = np.array([3.0,  1.0])
    
    # instantiate the MPCC object
    mpcc = MPCC(np.zeros((plant.n, plant.n)), np.zeros((plant.n, plant.n_u)),
                x_min, x_max,
                u_min, u_max,
                dt = dfl.dt_data, N = 30)

    # set the observation function, path object and linearization function
    setattr(plant, "g", plant.g_state_and_auxiliary)
    setattr(mpcc , "path_eval", spl_path.path_eval)
    setattr(mpcc , "get_linearized_model", dfl.linearize_soil_dynamics_no_surface)
    setattr(mpcc , "get_soil_surface", plant.soil_surf_eval)

    # define the MPCC cost matrices and coef.
    Q = sparse.diags([100., 100.])
    R = sparse.diags([.001, .001, .001])
    q_theta = .1

    # Set the MPCC initial state (includes states, aux variables and path variable)
    x_0 = np.concatenate((xi_0, plant.phi(0.0, xi_0, np.array([0.,0.])), np.array([-10.0])))
    
    # set initial input (since input cost is differential)
    u_minus = np.array([0.0, 0.0, 0.0])

    # sets up the new mpcc problem
    mpcc.setup_new_problem(Q, R, q_theta, x_0, u_minus)
    
    # simulate the nonlinear system with linearized mpcc
    t_f = 3.2
    t, u, x_nonlin, y = dfl.simulate_system_nonlinear(xi_0, mpcc.control_function, t_f)
    
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x_nonlin[:,0], x_nonlin[:,1], color='tab:blue')
    ax = mpcc.draw_path(ax, -10.0, -8.0)
    ax = plant.draw_soil(ax, 0.0, 1.0)
    ax.axis('equal')
    plt.show()

    solved, result  = mpcc.solve()
    x_optimal, u_optimal  = mpcc.extract_result(result)
    mpcc.plot_result(result)

    # exit()
    ##############################################################
    # plant = Plant1()
    # dfl = DFL(plant)

    # setattr(plant, "g", plant.g_state_and_surface)

    # x_0 = np.array([0.0,0.0,0.0,0.0,0.0])
    # dfl.generate_data_from_random_trajectories(t_range_data = 5.0, n_traj_data = 2, x_0 = x_0, plot_sample = True)
    # # dfl.generate_K_matrix()
    # print(dfl.Y_minus.shape)
   
    # x_0 = np.array([0.0,0.0,0.0,0.0,0.0])
    # t_f = 5.0

    # t, u, x_nonlin, y_nonlin= dfl.simulate_system_nonlinear(x_0, zero_u_func, t_f)
    # # t, u, x_koop1, y_koop = dfl.simulate_system_koop(x_0,zero_u_func, t_f)

    fig, axs = plt.subplots(4, 1)   
    
    axs[0].plot(t, x_nonlin[:,0],'b', t, x_nonlin[:,1],'r')
    # axs[0].plot(t, x_koop1[:,0],'b--',  t, x_koop1[:,1],'r--')
    axs[0].set_xlim(0, t_f)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('position states')
    axs[0].grid(True)

    axs[1].plot(t, x_nonlin[:,2],'b', t, x_nonlin[:,3],'r')
    # axs[1].plot(t, x_koop1[:,2],'b--',  t, x_koop1[:,3],'r--')
    axs[1].set_xlim(0, t_f)
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('velocity states')
    axs[1].grid(True)

    axs[2].plot(t, x_nonlin[:,4],'g')
    # axs[2].plot(t, x_koop1[:,4],'g--')
    axs[2].set_xlim(0, t_f)
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('bucket filling')
    axs[2].grid(True)

    axs[3].plot(t, u[:,0,0],'k')
    axs[3].plot(t, u[:,0,1],'k--')
    axs[3].set_xlim(0, t_f)
    axs[3].set_ylim(-3, 3)
    axs[3].set_xlabel('time')
    axs[3].set_ylabel('input')
    axs[3].grid(True)

    fig.tight_layout()
    plt.show()

    # #########################################


    # #########################################

