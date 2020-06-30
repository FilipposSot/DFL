import osqp
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt

class MPC():
    
    def __init__(self, Ad, Bd, x_min, x_max, u_min, u_max, N = 20):
        
        [self.nx, self.nu] = Bd.shape
        self.N = N

        self.Ad = Ad
        self.Bd = Bd
        self.x_min = x_min 
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max

        self.prob = osqp.OSQP()


    def generate_objective(self, Q, QN, R, xr):

        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), Q), QN,
                               sparse.kron(sparse.eye(self.N), R)], format='csc')
        # - linear objective
        q = np.hstack([np.kron(np.ones(self.N), -Q.dot(xr)), -QN.dot(xr),
                       np.zeros(self.N*self.nu)])

        return P, q

    def generate_constraints(self, x0):

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

        P, q = self.generate_objective(Q, QN, R, xr)
        A, l, u = self.generate_constraints(x0)

        self.P = P
        self.q = q
        self.A = A
        self.l = l
        self.u = u

        self.prob.setup(P, q, A, l, u, warm_start=True)


    def solve(self):
        result = self.prob.solve()
        # # Check solver status
        # if result.info.status != 'solved':
        #     print('hi')
        #     raise ValueError('OSQP did not solve the problem!')
        return result.info.status == 'solved', result 

    def get_control(self, result):
        ctrl = result.x[-self.N*self.nu:-(self.N-1)*self.nu]
        return ctrl
    
    def update_initial_state(self,x0):
        # Update initial state
        print('self.l', self.l[:self.nx])
        print('x0', x0)

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

if __name__== "__main__":



    prob,l,u = create_MPC_problem(N,Q,QN,R,Ad,Bd,xmin,xmax,umin,umax)

    # Simulate in closed loop
    X = []
    nsim = 15
    for i in range(nsim):
        # Solve
        res = prob.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        # Apply first control input to the plant
        ctrl = res.x[-N*nu:-(N-1)*nu]
        X.append(x0)
        x0 = Ad.dot(x0) + Bd.dot(ctrl)

        # Update initial state
        l[:nx] = -x0
        u[:nx] = -x0
        prob.update(l=l, u=u)

    X = np.array(X)
    plt.plot(X[:,5])
    plt.show()