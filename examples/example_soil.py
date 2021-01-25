#!/usr/bin/env python

import numpy as np
from dfl.dfl.dfl_soil import *
from dfl.dfl.mpcc import *

from dfl.dfl.dynamic_system import *
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splrep, splev, splint

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

   
    @staticmethod
    def Phi_soil(D, x, z, v_x, v_z, gamma):
        '''
        place hold soil force
        will be replaced by FEE
        '''
        F = (-D**2 -10*D*v_x)*np.array([1.0,0.0])
        F = F + ( D + D**3 + -10*D*v_z)*np.array([0.0,1.0])
        
        return F

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

if __name__== "__main__":

    # initialize the excavation system plant
    plant = Plant1()

    # generate a soil surface to test on. add it to the plant
    x_soil_points = np.array([0,0.25, 0.5,0.75 ,1. ,1.25, 1.5])
    y_soil_points = np.array([0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0])
    plant.set_soil_surf(x_soil_points, y_soil_points)

    # create the dfl object with the digging plant
    dfl = DFLSoil(plant)
    
    # # set the observation to include soil shape parameters
    # setattr(plant, "g", plant.g_state_and_surface)
    
    # set a fixed initial position (zero)
    xi_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    t_f = 15.0
    t, u, x_nonlin, y_nonlin= dfl.simulate_system_nonlinear(xi_0, control_u_func, t_f)

    

    # # generare training data
    dfl.generate_data_from_random_trajectories(t_range_data = 5.0,
                                               n_traj_data = 20,
                                               x_0 = xi_0,
                                               plot_sample = True)
    
    # regress the dfl soil model
    # Here we don't include the shape of the soil surface
    dfl.regress_model_no_surface()
    
    t, u, x_dfl, y_dfl = dfl.simulate_system_dfl(xi_0, control_u_func, t_f, continuous = False)

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


    #########################  PLOTTING  ####################################

    fig, axs = plt.subplots(4, 1)   
    
    # axs[0].plot(t, x_nonlin[:,0],'b', t, x_nonlin[:,1],'r')
    axs[0].plot(t, x_dfl[:,0],'b--',  t, x_dfl[:,1],'r--')
    axs[0].set_xlim(0, t_f)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('position states')
    axs[0].grid(True)

    # axs[1].plot(t, x_nonlin[:,2],'b', t, x_nonlin[:,3],'r')
    axs[1].plot(t, x_dfl[:,2],'b--',  t, x_dfl[:,3],'r--')
    axs[1].set_xlim(0, t_f)
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('velocity states')
    axs[1].grid(True)

    # axs[2].plot(t, x_nonlin[:,4],'g')
    axs[2].plot(t, x_dfl[:,4],'g--')
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