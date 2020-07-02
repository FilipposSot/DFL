import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt

class MPC():
    
    def __init__(self, Ad, Bd, x_min, x_max, u_min, u_max, N = 20):
        
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
        q = np.hstack([np.kron(np.ones(self.N), -Q.dot(xr)), -QN.dot(xr),
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

    def setup_new_problem(self, Q, QN, R, xr, x0):
        # sets up a new osqp mpc problem by determining all the matrices and vectors
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

    def control_function(self, x, t):

        self.update_initial_state(x)
        solved, result  = self.solve()

        if solved:
            u = self.get_control(result)
        else:
            print('failed to find mpc solution')
            exit()

        return u