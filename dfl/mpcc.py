import osqp
import numpy as np
import scipy as sp
from scipy.linalg import expm
from scipy import sparse
import matplotlib.pyplot as plt
import copy
from scipy.signal import cont2discrete
from scipy.linalg import toeplitz
from scipy.interpolate import splprep, splrep, splev, splint
import scipy.integrate as integrate

import time

np.set_printoptions(precision=4)

class spline_path():
    def __init__(self, x, y, N=500):
        
        self.tck_u_to_path, U_splrep = splprep([x,y],s=0)
        self.N = N
        self.U = np.linspace(U_splrep[0],U_splrep[-1],N)
        self.calculate_arclength()
        self.tck_s_to_u = splrep(self.S, self.U, s=0)

    def path_eval(self, theta, d=0):
        # for now lets keep this a very simple path: a circle
        s = theta + 10.0
        if d == 0:
            u = splev(s, self.tck_s_to_u, der=0, ext=3)
            x,y = splev(u, self.tck_u_to_path, der=0, ext=3)
            return x,y

        elif d == 1:
            u = splev(s, self.tck_s_to_u, der=0, ext=3)
            du_ds = splev(s, self.tck_s_to_u, der=1, ext=3)
            dx_du, dy_du = splev(u, self.tck_u_to_path, der=1, ext=3)
            
            dx_ds = dx_du*du_ds
            dy_ds = dy_du*du_ds
             
            return  dx_ds, dy_ds

        elif d == 2:
            u                   = splev(s, self.tck_s_to_u, der = 0, ext=3)
            du_ds               = splev(s, self.tck_s_to_u, der = 1, ext=3)
            d2u_ds2             = splev(s, self.tck_s_to_u, der = 2, ext=3)
            dx_du, dy_du        = splev(u, self.tck_u_to_path, der = 1, ext=3)
            d2x_du2, d2y_du2    = splev(u, self.tck_u_to_path, der = 2, ext=3)

            d2x_ds2 = d2x_du2*(du_ds)**2 + dx_du*d2u_ds2
            d2y_ds2 = d2y_du2*(du_ds)**2 + dy_du*d2u_ds2
            
            return d2x_ds2, d2y_ds2

    def calculate_arclength(self):      
        self.S = np.zeros(self.U.shape)
        for i in range(self.N):
            self.S[i] = integrate.quad(self.arclength_integrand,self.U[0],self.U[i],args=(self.tck_u_to_path))[0]

    @staticmethod
    def arclength_integrand(u,tck):
        x,y = splev([u], tck, der=1)
        return np.sqrt(x**2 + y**2)    

class MPCC():
    
    def __init__(self, Ad, Bd, x_min, x_max, u_min, u_max, dt, N = 50, animate = True):
        
        # system dimensions
        [self.nxi, self.nu] = Bd.shape
        self.nx = self.nxi + 1
        self.nv = self.nu + 1

        self.temp_val = 0
        self.max_init_its = 100
        self.t_last_plot = -1000000

        # optimization horizon
        self.N = N

        self.dt = dt

        self.Ad = np.zeros(( self.nx, self.nx))
        self.Ad[:self.nxi,:self.nxi] = Ad
        self.Ad[-1,-1] = 1.0

        self.Bd = np.zeros((self.nx, self.nv))
        self.Bd[:self.nxi,:self.nu] = Bd
        self.Bd[-1,-1] = dt

        self.x_min = np.concatenate((x_min,np.array([-10.5]))) 
        self.x_max = np.concatenate((x_max,np.array([0.0]))) 
       
        self.u_min = np.concatenate((u_min,np.array([0.00001]))) 
        self.u_max = np.concatenate((u_max,np.array([50.]))) 

        self.prob = osqp.OSQP()



    def error_eval(self, x_phys, y_phys, theta):

        x_virt, y_virt                      = self.path_eval(theta, d = 0)
        dxvirt_dtheta , dyvirt_dtheta       = self.path_eval(theta, d = 1)
        d2xvirt_dtheta2 , d2yvirt_dtheta2   = self.path_eval(theta, d = 2)

        phi_virt = np.arctan2(dyvirt_dtheta, dxvirt_dtheta) # orientation of virtual position

        #  difference in position between virtual and physical
        Dx = (x_phys-x_virt)
        Dy = (y_phys-y_virt)

        eC =  np.sin(phi_virt)*Dx - np.cos(phi_virt)*Dy
        eL = -np.cos(phi_virt)*Dx - np.sin(phi_virt)*Dy

        #  computes {d phi_virt / d theta} evaluated at theta_k
        numer = dxvirt_dtheta*d2yvirt_dtheta2 -dyvirt_dtheta*d2xvirt_dtheta2
        denom = dxvirt_dtheta**2 + dyvirt_dtheta**2
        dphivirt_dtheta = numer/denom

        cos_phi_virt = np.cos(phi_virt)
        sin_phi_virt = np.sin(phi_virt)

        tmp1 = np.array([dphivirt_dtheta, 1])
        tmp2 = np.array([[cos_phi_virt],[sin_phi_virt]])

        MC = np.array([[ Dx, Dy],[dyvirt_dtheta, -dxvirt_dtheta]]) # CHECK
        ML = np.array([[-Dy, Dx],[dxvirt_dtheta,  dyvirt_dtheta]])

        deC_dtheta = tmp1.dot(MC).dot(tmp2)
        deL_dtheta = tmp1.dot(ML).dot(tmp2)

        grad_eC = np.concatenate((np.array([sin_phi_virt]),
                                  np.array([-cos_phi_virt]),
                                  np.zeros(self.nx-3),
                                  deC_dtheta))

        grad_eL = np.concatenate((np.array([-cos_phi_virt]),
                                  np.array([-sin_phi_virt]),
                                  np.zeros(self.nx-3),
                                  deL_dtheta))

        return eC, eL, grad_eC, grad_eL, 

    # TODO: Probably syntatically full of errors, will figure out later
    def generate_contouring_objective(self, Q, R, q_theta, x_array):

        # P = sparse.csc_matrix(np.zeros((self.nx, self.nx)))
        # q = np.zeros(self.nx)

        x_traj, y_traj, theta_traj = x_array[:,0], x_array[:,1], x_array[:,-1]

        x_path = []
        y_path = []

        # quadratic objective
        for i in range(len(theta_traj)):
            
            eC, eL, grad_eC, grad_eL = self.error_eval(x_traj[i], y_traj[i], theta_traj[i])
            
            x_virt, y_virt  = self.path_eval(theta_traj[i], d = 0)
            x_path.append(x_virt)
            y_path.append(y_virt)

            errorgrad = np.array([grad_eC,grad_eL])
            error = np.array([eC,eL])

            # quadratic error
            Qtilde = 2*np.transpose(errorgrad).dot(Q.toarray()).dot(errorgrad) #Check dimensions
            Qtilde = 0.5*(Qtilde + np.transpose(Qtilde))

            error_zero = error - errorgrad.dot(x_array[i,:])
            
            # linear error
            qtilde = 2*error_zero.dot(Q.toarray()).dot(errorgrad)  - q_theta*np.concatenate((np.zeros(self.nx-1),np.array([1])))

            if i == 0:
                P = sparse.csc_matrix(Qtilde)
                q = copy.copy(qtilde)
            else:
                P = sparse.block_diag([P, Qtilde])
                q = np.concatenate((q,qtilde))

        toep_col = np.concatenate((np.array([-1]), np.zeros(self.N*self.nv - 1)))
        toep_row = np.concatenate((np.array([-1]), np.zeros(self.nv - 1), np.array([1]), np.zeros((self.N+1)*self.nv - self.nv - 1)))
        toep = sparse.csc_matrix(toeplitz(toep_col,toep_row))

        Rtilde =  np.transpose(toep).dot(sparse.kron(sparse.eye(self.N), R)).dot(toep)

        P = sparse.block_diag([P,Rtilde])
        q = np.concatenate((q,np.zeros((self.N+1)*self.nv)))
        
        ########################## Adding Power to Cost function ###############
        # PP = np.zeros(((self.N+1)*(self.nx+self.nv),(self.N+1)*(self.nx+self.nv)))
        # # print(P.shape)
        # # print(PP.shape)
        
        # a = np.zeros((self.nx,self.nv))
        # a[3,0] = 10.0
        # a[4,1] = 0.0
        # a[5,2] = 0.0
        # A = np.kron(np.eye(self.N+1), a)
        # PP[:(self.N+1)*self.nx,(self.N+1)*self.nx:] = A
        # PP[(self.N+1)*self.nx:,:(self.N+1)*self.nx] = np.transpose(A)

        # P = P + sparse.csc_matrix(PP)
        
        # # import pickle
        # # fig, axs = plt.subplots(1,1)
        # # plt.imshow(np.sign(P),interpolation='none',cmap='bwr')
        # # pickle.dump(fig, open('mpcc_cost_matrix.fig.pickle', 'wb')) 
        # # plt.show()
        # # exit()

        ######################## Added cost for stabilizing the angular ########
        QQ = np.zeros((self.nx,self.nx))
        QQ[2,2]= 300.0
        QQN = QQ
        
        RR = 0.*np.eye(self.nv)
        RR[2,2] = 0.0

        xr = np.zeros(self.nx)
        xr[2] = 1.55

        PP = sparse.block_diag([sparse.kron(sparse.eye(self.N), QQ), QQN,
                                sparse.kron(sparse.eye(self.N+1), RR)], format='csc')
        
        qq = np.hstack([np.kron(np.ones(self.N), -QQ.dot(xr)), -QQN.dot(xr),
                       np.zeros((self.N+1)*self.nv)])

        P = P+PP
        q = q+qq
        #########################################################################

        return P, q
    
    # TODO
    def generate_contouring_constraints(self, x0, u_minus, x_array):
        
        # generates the constraints for the system including state, inputs and dynamics
        # l <= Ax <= u

        # - linear dynamics constraint
        Ax_1 = sparse.kron(sparse.eye(self.N+1),-sparse.eye(self.nx)) # + sparse.kron(sparse.eye(self.N+1, k=-1), self.Ad)
        Ax_2 = np.zeros((self.nx*(self.N+1),self.nx*(self.N+1)))
        Bu = np.zeros(((self.N+1)*self.nx, (self.N+1)*self.nv))

        l_dyn = -x0

        for i in range(self.N):
            i_row_start = self.nx*i + self.nx 
            i_row_end = i_row_start + self.nx
            
            i_col_start = self.nx*i
            i_col_end = i_col_start  + self.nx

            Ad_xi, Bd_xi, Kd_xi = self.get_linearized_model(x_array[i,:])

            Ad = np.zeros(( self.nx, self.nx))
            Ad[:self.nxi,:self.nxi] = Ad_xi
            Ad[-1,-1] = 1.0

            Bd = np.zeros((self.nx, self.nv))
            Bd[:self.nxi,:self.nu] = Bd_xi
            Bd[-1,-1] = self.dt

            # check!
            Kd = np.concatenate((-Kd_xi,np.array([0])))

            Ax_2[i_row_start:i_row_end, i_col_start:i_col_end] = Ad
            Bu[i_row_start:i_row_end, self.nv*i+ self.nv :self.nv*i + self.nv +self.nv ] = Bd           
            l_dyn = np.concatenate((l_dyn , Kd))
        
        Ax = Ax_1 + Ax_2


        A_dyn = sparse.hstack([Ax, sparse.csc_matrix(Bu)])
        u_dyn = l_dyn
        
        # - input and state constraints
        Aineq = sparse.eye((self.N+1)*self.nx + (self.N+1)*self.nv)
        lineq = np.hstack([np.kron(np.ones(self.N+1), self.x_min), u_minus, np.kron(np.ones(self.N), self.u_min)])
        uineq = np.hstack([np.kron(np.ones(self.N+1), self.x_max), u_minus, np.kron(np.ones(self.N), self.u_max)])
        
        for i in range(self.N+1):
            uineq[1+self.nx*i],_,_,_ = self.get_soil_surface(x_array[i,0])
        
        # - OSQP constraints
        A = sparse.vstack([A_dyn, Aineq], format = 'csc')
        l = np.hstack([l_dyn, lineq])
        u = np.hstack([u_dyn, uineq])
        
        return A, l, u

    def initialize_trajectory(self, x0, u_minus):

        u_array = np.zeros((self.N,self.nv))

        u_array[:,0] = 0.*np.ones(self.N) #0.0*np.linspace(0,1,self.N)
        u_array[:,1] = 0.*np.ones(self.N) #0.0*np.linspace(0,1,self.N)
        u_array[:,2] = 0.*np.ones(self.N) #0.1*np.ones(self.N)

        x_array = np.zeros((self.N+1,self.nx))
        x_array[0,:] = x0

        for i in range(self.N):
            
            Ad_xi, Bd_xi, Kd_xi = self.get_linearized_model(x_array[i,:])

            Ad = np.zeros(( self.nx, self.nx))
            Ad[:self.nxi,:self.nxi] = Ad_xi
            Ad[-1,-1] = 1.0

            Bd = np.zeros((self.nx, self.nv))
            Bd[:self.nxi,:self.nu] = Bd_xi
            Bd[-1,-1] = self.dt

            Kd_x = np.concatenate((Kd_xi,np.array([0.])))

            x_array[i+1,:] = Ad.dot(x_array[i,:]) + Bd.dot(u_array[i,:]) + Kd_x

        for j in range(self.max_init_its):

            self.temp_val = j

            x_array_old = x_array[:]
            u_array_old = u_array[:]

            P, q = self.generate_contouring_objective(self.Q, self.R, self.q_theta, x_array)
            A, l, u = self.generate_contouring_constraints(x0, u_minus, x_array) 
     
            prob = osqp.OSQP()
            prob.setup(P, q, A, l, u, warm_start = True, verbose = False)
            
            result  = prob.solve()
            
            if result.info.status == 'solved':
                x_opt , u_opt = self.extract_result(result)
                u_array = u_opt
                x_array = x_opt
            else:
                print('failed to find mpc solution during initialization')
                exit()

            delta_u_norm = np.linalg.norm(x_array-x_array_old)
            
            if delta_u_norm < 0.001:
                break
        
        # self.plot_opt(x_array, u_array)

        return x_array, u_array

    def plot_opt(self, x_array,u_array):

        fig, axs = plt.subplots(7,1, figsize=(8,10))

        axs[0] = self.draw_path(axs[0], x_array[0,-1], x_array[-1,-1])
        axs[0].plot(x_array[:,0], x_array[:,1], marker=".")
        # axs[0].plot(x_array[:,1],marker=".")
        
        axs[1].plot(x_array[:,0],marker=".")
        axs[1].plot(x_array[:,1],marker=".")
        axs[1].plot(x_array[:,2],marker=".")

        axs[2].plot(x_array[:,3],marker=".")
        axs[2].plot(x_array[:,4],marker=".")
        axs[2].plot(x_array[:,5],marker=".")

        axs[3].plot(x_array[:,6],marker=".")
        axs[3].plot(x_array[:,7],marker=".")
        axs[3].plot(x_array[:,8],marker=".")

        axs[4].plot(x_array[:,6],marker=".")
        axs[4].plot(x_array[:,7],marker=".")
        
        
        axs[5].plot(u_array[:,0],marker=".")
        axs[5].plot(u_array[:,1],marker=".")
        # axs[5].plot(u_array[:,2],marker=".")

        # axs[6].plot(x_array[:,8],marker=".")
        # axs[6].plot(x_array[:,11],marker=".")

        plt.show()

    def setup_new_problem(self, Q, R, q_theta, x0, u_minus, t = 0):
        # sets up a new osqp mpc problem by determining all the 
        # matrices and vectors and adding them to the problem
        
        self.Q = Q
        self.R = R
        self.q_theta = q_theta
        
        self.u_minus = u_minus 
        
        x_array, u_array = self.initialize_trajectory(x0, u_minus)

        print("############ Trajectory Initialized ################")

        P, q = self.generate_contouring_objective(self.Q, self.R, self.q_theta, x_array)
        A, l, u = self.generate_contouring_constraints(x0, u_minus, x_array)

        self.u_array = u_array[:]
        self.x_array = x_array[:]

        self.P = copy.copy(P)
        self.q = copy.copy(q)
        self.A = copy.copy(A)
        self.l = copy.copy(l)
        self.u = copy.copy(u)

        self.prob.setup(P, q, A, l, u, warm_start = False, verbose = False, polish = 1)
        solved, result  = self.solve()

        print("############ OSQP Setup ################")

    def update_problem(self, xi0, u_minus):

        x0 = np.concatenate((xi0, np.array([self.x_array[0,-1] + self.dt*u_minus[-1]])))

        x_array = self.x_array[:]
        
        P, q = self.generate_contouring_objective(self.Q, self.R, self.q_theta, x_array)
        A, l, u = self.generate_contouring_constraints(x0, u_minus, x_array)

        self.prob = osqp.OSQP()
        self.prob.setup(P, q, A, l, u,  warm_start = False, verbose = False, polish = 1)

    def solve(self):
        #solve the mpc problem
        result = self.prob.solve()
        return result.info.status == 'solved', result 

    def extract_result(self, result):
        #extract the optimal trajectory and control input
        X = result.x

        x_optimal = np.reshape(X[:(self.N+1)*self.nx ], (self.N+1, self.nx))
        u_optimal = np.reshape(X[ self.nv+(self.N+1)*self.nx:], (self.N, self.nv))

        return x_optimal, u_optimal 

    def plot_result(self,result):
        x_optimal, u_optimal  = self.extract_result(result)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_optimal[:,0], x_optimal[:,1],'.', color='tab:blue')
        ax = self.draw_path(ax, x_optimal[0,-1], x_optimal[-1,-1])
        ax = self.draw_soil(ax,x_optimal[0,0], x_optimal[-1,0])
        ax.axis('equal')
        plt.show()
        
    def draw_path(self,ax,theta_min,theta_max):
        # for now lets keep this a very simple path: a circle
        theta = np.linspace(theta_min,theta_max, self.N)
        x = np.zeros(theta.shape)
        y = np.zeros(theta.shape)
        
        for i in range(len(theta)):
            x[i],y[i] = self.path_eval(theta[i])
        
        ax.plot(x, y,'--', color='tab:orange' , marker='.')

        return ax

    def draw_soil(self,ax,x_min,x_max):
        # for now lets keep this a very simple path: a circle
        x = np.linspace(x_min,x_max, 100)
        y = np.zeros(x.shape)
        
        for i in range(len(x)):
           y[i],_,_,_ = self.get_soil_surface(x[i])
        
        ax.plot(x, y, color = 'black')

        return ax


    def control_function(self, xi, t):
        # callable control function
        self.update_problem(xi, self.u_minus)
        solved, result  = self.solve()

        if solved:
            x_optimal, u_optimal  = self.extract_result(result)
            u = u_optimal[0,:self.nu]
            
            self.u_minus = u_optimal[0,:self.nv]
            self.x_opt_last = x_optimal
            self.u_opt_last = u_optimal
              
            self.x_array = copy.copy(x_optimal)

            return u , x_optimal[1,:], x_optimal, u_optimal

        else:
            print('Failed to find MPC solution')
            
            return self.u_minus[:self.nu] , self.x_opt_last[1,:], self.x_opt_last, self.u_opt_last 

if __name__== "__main__":

    exit()
