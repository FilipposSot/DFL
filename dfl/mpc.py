import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt

class MPC():
    
    def __init__(self, Ad, Bd, x_min, x_max, u_min, u_max, N = 50):
        
        # system dimensions
        [self.nx, self.nu] = Bd.shape
        
        # optimization horizon
        self.N = N

        self.Ad = Ad
        self.Bd = Bd
        self.x_min = x_min 
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max

        self.prob = osqp.OSQP()


    def generate_objective(self, Q, QN, R, xr):
        #generates the QP objective function
        # 0.5x'Px + q'x

        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), Q), QN,
                               sparse.kron(sparse.eye(self.N), R)], format='csc')
        
        # - linear objective
        # fixed reference state
        if len(xr.shape) == 1:
            q = np.hstack([np.kron(np.ones(self.N), -Q.dot(xr)), -QN.dot(xr),
                           np.zeros(self.N*self.nu)])

        # reference state trajectory
        elif len(xr.shape) == 2:

            # if length of reference < horizon then add final reference
            if xr.shape[0] < self.N:
                xr = np.vstack((xr,np.tile(xr[-1,:],(self.N-xr.shape[0],1))))
            # if length of reference > horizon then we trim the reference
            elif xr.shape[0] > self.N:
                xr = xr[:self.N,:]

            q = np.hstack([-Q.dot(xr.T).T.flatten(), -QN.dot(xr[-1,:]),
                           np.zeros(self.N*self.nu)])

        return P, q

    def generate_constraints(self, x0):
        # generates the constraints for the system including state, inputs and dynamics
        # l <= Ax <= u

        # - linear dynamics constraint
        Ax = sparse.kron(sparse.eye(self.N+1),-sparse.eye(self.nx)) + sparse.kron(sparse.eye(self.N+1, k=-1), self.Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, self.N)), sparse.eye(self.N)]), self.Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(self.N*self.nx)])
        ueq = leq

        # - input and state constraints
        Aineq = sparse.eye((self.N+1)*self.nx + self.N*self.nu)
        lineq = np.hstack([np.kron(np.ones(self.N+1), self.x_min), np.kron(np.ones(self.N), self.u_min)])
        uineq = np.hstack([np.kron(np.ones(self.N+1), self.x_max), np.kron(np.ones(self.N), self.u_max)])

        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format = 'csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        return A, l, u

    def setup_new_problem(self, Q, QN, R, t_traj, x_traj, x0, t = 0):
        # sets up a new osqp mpc problem by determining all the 
        # matrices and vectors and adding them to the problem
        
        self.Q = Q
        self.QN = QN
        self.R = R

        self.t_traj = t_traj
        self.x_traj = x_traj

        xr = self.get_reference(t)

        P, q = self.generate_objective(Q, QN, R, xr)
        A, l, u = self.generate_constraints(x0)

        self.P = P
        self.q = q
        self.A = A
        self.l = l
        self.u = u

        self.prob.setup(P, q, A, l, u, warm_start = True, verbose = False)


    def solve(self):
        #solve the mpc problem
        result = self.prob.solve()
        return result.info.status == 'solved', result 

    def get_control(self, result):
        #extract first control action from optimal solution
        ctrl = result.x[-self.N*self.nu:-(self.N-1)*self.nu]
        return ctrl
    
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