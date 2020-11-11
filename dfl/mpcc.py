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

np.set_printoptions(precision=4)

class spline_path():
    def __init__(self, x,y,N=500):
        
        self.tck_u_to_path, U_splrep = splprep([x,y],s=0)
        self.N = N
        self.U = np.linspace(U_splrep[0],U_splrep[-1],N)
        self.calculate_arclength()
        self.tck_s_to_u = splrep(self.S, self.U, s=0)


    def path_eval(self, theta, d=0):
        # for now lets keep this a very simple path: a circle
        s = theta + 10.0
        if d == 0:
            u = splev(s, self.tck_s_to_u, der=0 )
            x,y = splev(u, self.tck_u_to_path, der=0)
            return x,y

        elif d == 1:
            u = splev(s, self.tck_s_to_u, der=0)
            du_ds = splev(s, self.tck_s_to_u, der=1)
            dx_du, dy_du = splev(u, self.tck_u_to_path, der=1)
            
            dx_ds = dx_du*du_ds
            dy_ds = dy_du*du_ds
             
            return  dx_ds, dy_ds

        elif d == 2:
            u                   = splev(s, self.tck_s_to_u, der = 0)
            du_ds               = splev(s, self.tck_s_to_u, der = 1)
            d2u_ds2             = splev(s, self.tck_s_to_u, der = 2)
            dx_du, dy_du        = splev(u, self.tck_u_to_path, der = 1)
            d2x_du2, d2y_du2    = splev(u, self.tck_u_to_path, der = 2)

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
    
    def __init__(self, Ad, Bd, x_min, x_max, u_min, u_max, N = 50):
        
        # system dimensions
        [self.nxi, self.nu] = Bd.shape
        self.nx = self.nxi + 1
        self.nv = self.nu + 1

        self.temp_val = 0
        self.max_init_its = 50
        # optimization horizon
        self.N = N

        self.Ad = np.zeros(( self.nx, self.nx))
        self.Ad[:self.nxi,:self.nxi] = Ad
        self.Ad[-1,-1] = 1.0

        self.Bd = np.zeros((self.nx, self.nv))
        self.Bd[:self.nxi,:self.nu] = Bd
        self.Bd[-1,-1] = dt

        self.x_min = np.concatenate((x_min,np.array([-10.0]))) 
        self.x_max = np.concatenate((x_max,np.array([0.0]))) 
       
        self.u_min = np.concatenate((u_min,np.array([0.001]))) 
        self.u_max = np.concatenate((u_max,np.array([1.0]))) 

        self.prob = osqp.OSQP()

    # TODO: function that returns path variables and gradients

    def path_eval(self, theta, d=0):

        # for now lets keep this a very simple path: a circle
        if d == 0:
            return  np.cos(theta), np.sin(theta)
        elif d == 1:
            return  -np.sin(theta), np.cos(theta)
        elif d == 2:
            return -np.cos(theta), -np.sin(theta)

    # dummy
    def get_linearized_model(self,x):
        return self.Ad

    def draw_path(self,ax,theta_min,theta_max):
        # for now lets keep this a very simple path: a circle
        theta = np.linspace(theta_min,theta_max, self.N)
        x = np.zeros(theta.shape)
        y = np.zeros(theta.shape)
        
        for i in range(len(theta)):
            x[i],y[i] = self.path_eval(theta[i])
        
        ax.plot(x, y, 'r--')

        return ax

    def error_eval(self, x_phys, y_phys, theta):

        x_virt, y_virt                      = self.path_eval(theta, d = 0)
        dxvirt_dtheta , dyvirt_dtheta       = self.path_eval(theta, d = 1)
        d2xvirt_dtheta2 , d2yvirt_dtheta2   = self.path_eval(theta, d = 2)

        phi_virt = np.arctan2(dyvirt_dtheta, dxvirt_dtheta) # orientation of virtual position

        # % difference in position between virtual and physical
        Dx = (x_phys-x_virt)
        Dy = (y_phys-y_virt)

        eC =  np.sin(phi_virt)*Dx - np.cos(phi_virt)*Dy
        eL = -np.cos(phi_virt)*Dx - np.sin(phi_virt)*Dy

        # % computes {d phi_virt / d theta} evaluated at theta_k
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

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(x_traj, y_traj,'.', color='tab:blue')
        # ax.plot(np.array(x_path), np.array(y_path),'.', color='tab:red')
        # ax.axis('equal')
        # plt.show()

        return P, q
    
    # TODO
    def generate_contouring_constraints(self, x0, u_minus, x_array):
        
        # generates the constraints for the system including state, inputs and dynamics
        # l <= Ax <= u

        # - linear dynamics constraint
        Ax_1 = sparse.kron(sparse.eye(self.N+1),-sparse.eye(self.nx)) # + sparse.kron(sparse.eye(self.N+1, k=-1), self.Ad)
        Ax_2 = np.zeros((self.nx*(self.N+1),self.nx*(self.N+1)))
        
        for i in range(self.N):
            i_row_start = self.nx*i + self.nx 
            i_row_end = i_row_start + self.nx
            
            i_col_start = self.nx*i
            i_col_end = i_col_start  + self.nx

            Ad = self.get_linearized_model(x_array[i,:])
            
            Ax_2[i_row_start:i_row_end,i_col_start:i_col_end] = Ad

        Ax = Ax_1 + Ax_2 

        Bu = sparse.kron(sparse.hstack([sparse.csc_matrix((self.N+1,1)), sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)])]), self.Bd)
        # Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), self.Bd)

        A_dyn = sparse.hstack([Ax, Bu])
        l_dyn = np.hstack([-x0, np.zeros(self.N*self.nx)])
        u_dyn = l_dyn

        # - input and state constraints
        Aineq = sparse.eye((self.N+1)*self.nx + (self.N+1)*self.nv)
        lineq = np.hstack([np.kron(np.ones(self.N+1), self.x_min), u_minus, np.kron(np.ones(self.N), self.u_min)])
        uineq = np.hstack([np.kron(np.ones(self.N+1), self.x_max), u_minus, np.kron(np.ones(self.N), self.u_max)])
        # lineq = np.hstack([np.kron(np.ones(self.N+1), self.x_min), u_minus, np.kron(np.ones(self.N), self.u_min)])
        # uineq = np.hstack([np.kron(np.ones(self.N+1), self.x_max), u_minus, np.kron(np.ones(self.N), self.u_max)])

        # - OSQP constraints
        A = sparse.vstack([A_dyn, Aineq], format = 'csc')
        l = np.hstack([l_dyn, lineq])
        u = np.hstack([u_dyn, uineq])
        
        return A, l, u


    def initialize_trajectory(self, x0, u_minus):

        u_array = np.zeros((self.N,self.nv))

        u_array[:,0] =  0.0*np.linspace(0,1,self.N)
        u_array[:,1] =  0.0*np.linspace(0,1,self.N)
        u_array[:,2] =  0.0*np.ones(self.N)

        x_array = np.zeros((self.N+1,self.nx))
        x_array[0,:] = x0

        for i in range(self.N):
            x_array[i+1,:] = self.Ad.dot(x_array[i,:]) + self.Bd.dot(u_array[i,:])

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

        return x_array, u_array


    def setup_new_problem(self, Q, R, q_theta, x0, u_minus, t = 0):
        # sets up a new osqp mpc problem by determining all the 
        # matrices and vectors and adding them to the problem
        
        self.Q = Q
        self.R = R
        self.q_theta = q_theta

        x_array, u_array = self.initialize_trajectory(x0, u_minus)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x_array[:,0], x_array[:,1],'.', color='tab:blue')
        ax = self.draw_path(ax,x_array[0,-1], x_array[-1,-1])
        ax.axis('equal')
        plt.show()

        P, q = self.generate_contouring_objective(self.Q, self.R, self.q_theta, x_traj)
        A, l, u = self.generate_contouring_constraints(x0)

        self.P = P
        self.q = q
        self.A = A
        self.l = l
        self.u = u

        self.prob.setup(P, q, A, l, u, warm_start = True, verbose = False)

        solved, result  = self.solve()
        
        if solved:
            x_traj = self.extract_result(result)
        else:
            print('failed to find mpc solution')
            exit()

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


    # ctrl = result.x[-self.N*self.nu:-(self.N-1)*self.nu]
    # return ctrl

    def update_initial_state(self,x0):
        # Update initial state for optimization
        self.l[:self.nx] = -x0
        self.u[:self.nx] = -x0
        self.prob.update(l = self.l, u = self.u)

    def update_objective(self, P, q):
        # Update initial state for optimization
        self.P = P
        self.q = q
        self.prob.update(Px = P.data, q = q)

    def get_reference(self,t):
        # produces the reference for the mpc controller
        if t < self.t_traj[-1]:
            xr = self.x_traj[self.t_traj>t,:]
        else:
            xr = self.x_traj[-1,:]
        return xr

    def control_function(self, x, t):
        # callable control function
        self.update_initial_state(x)

        xr = self.get_reference(t)
        P, q = self.generate_objective(self.Q, self.QN, self.R, xr)
        self.update_objective(P, q)

        solved, result  = self.solve()

        if solved:
            u = self.get_control(result)
        else:
            print('failed to find mpc solution')
            exit()

        return u



if __name__== "__main__":

    dt = 0.2

    A = np.array([[0.0 , 0.0, 1.0, 0.0],
                  [0.0 , 0.0, 0.0, 1.0],
                  [0.0 , 0.0, -.01, 0.0],
                  [0.0 , 0.0, 0.0, -0.01]]) 

    B = np.array([[0.0,0.0],
                  [0.0,0.0],
                  [1.0,0.0],
                  [0.0,1.0]])

    [nx, nu] = B.shape

    (Ad, Bd,_,_,_) = cont2discrete((A, B, np.zeros(nx), np.zeros(nu)), dt)
    # Objective function
    Q = sparse.diags([1., 1.])
    R = sparse.diags([.001, .001, 1.])
    q_theta = 0.0001

    # state limits
    x_min = np.array([-3., -3., -3., -3.])
    x_max = np.array([ 3.,  3.,  3.,  3.])
    u_min = np.array([-2.,-2.])
    u_max = np.array([ 2., 2.])


    x = np.array([-0.7,-1.0,-0.6,0.0 ,0.5,0.9,0.3])
    y = np.array([0.5 ,0.3 ,-0.1 ,-0.4,0.3,0.9,0.9])

    path = spline_path(x,y)

    mpc = MPCC(Ad,
               Bd,
               x_min, x_max,
               u_min, u_max,
               N = 20)
    
    setattr(mpc, "path_eval", path.path_eval)

    x0 = np.array([-0.7, 0.5, 0.0, 0.0, -10.0])
    u_minus = np.array([0.0, 0.0, 0.0])

    mpc.setup_new_problem(Q, R, q_theta, x0, u_minus)
    

    # x0 = np.array([1.0, 0.0, 0.0, 0.0])
    # x_array = []
    # x = copy.copy(x0)
    # for i in range(1000):
    #     x_array.append(x)
    #     x = Ad.dot(x) + Bd.dot(np.sign(np.sin(i/100))*(np.array([1.0,0.0])))


    # x_array = np.array(x_array)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(x_array[:,2] , color='tab:blue')
    # plt.show()


    # ####### TESTING splprep
    # N = 100

    # x = np.array([-0.7,-1.0,-0.6,0.0 ,0.5,0.9,0.3])
    # y = np.array([0.5 ,0.3 ,-0.1 ,-0.4,0.3,0.9,0.9])

    # tck, u = splprep([x,y],s=0)
    # u_new = np.linspace(0,1,N)
    # new_points = splev(u_new, tck)
    # # new_points_der = splev(u_new, tck,der=1)
    
    # def integrand(u,tck):
    #     x,y = splev([u],tck,der=1)
    #     return np.sqrt(x**2 + y**2)
    
    # I_array = np.zeros(u_new.shape)
    # I_array_2 = np.zeros(u_new.shape)

    # for i in range(N):
    #     I_array[i] = integrate.quad(integrand,0,u_new[i],args=(tck))[0]

    # fig, ax = plt.subplots()
    # ax.plot(u_new, I_array, 'r')
    # plt.show()