# author: Filippos Sotiropoulos (fes@mit.edu)

import numpy as np 
import scipy 
from scipy.integrate import ode
import control
import matplotlib.pyplot as plt


# VERY rough code for asada grant (to be templatized)
# class DflSystem():
#     def __init__(self):

#     def f(t,x,u):
#         '''
#         Full nonlinear dynamical transition equation
#         '''

#     def f_DFL(t,x,u):
#         '''
#         Dynamic transition equation based on lifted space linearization
#         '''

#     def find_DFL_matriced():

#     def generate_training_data():
    
#     def 
'''
f1 system 1
f2 system 2
    - eigs_i
    - k_i

f_star
n4sid --> system_star
    - eigs
    -
'''
def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
     
    dx/dt = A x + B u
     
    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151
     
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
 
    return K, X, eigVals
 
def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
     
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
     
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
     
    return K, X, eigVals

A1 = np.array([[0.89, 0.], [0., 0.45]])
A2 = np.array([[0.60, 0.], [0., 0.45]])
B1 = np.array([[0.3],[2.5]])
B2 = np.array([[0.3],[2.5]])

A3 = np.array([[0.8, 0.0],[0.0, 0.45]])
B3 = np.array([[0.3],[2.5]])

Q  = np.array([[1.0, 0.], [0., 1.0]])
R  = np.array([[1.0]])

K1, X1, eigVals1 = lqr(A1, B1, Q, R)
K2, X2, eigVals2 = lqr(A2, B2, Q, R)

def f1_good(t, y):
    return np.matmul((A3-B3*K1),np.transpose(np.array(y)))

def f1_bad(t, y):
    return np.matmul((A3-B3*K2),np.transpose(np.array(y)))



if __name__ == '__main__':
    
    A_id = np.array([[0.75, -0.02],[0.03, 0.41]])
    B_id = np.array([[0.32],[2.55]])

    A_diff_1 = np.sum(np.abs(A_id - A1))
    A_diff_2 = np.sum(np.abs(A_id - A2))

    y0, t0 = [1.0, 1.0], 0
    r_good = ode(f1_good).set_integrator('dopri5', method='bdf')
    r_bad  = ode(f1_bad).set_integrator('dopri5', method='bdf')

    r_good.set_initial_value(y0, t0)
    r_bad.set_initial_value(y0, t0)

    t1 = 10
    dt = 0.02
    t = [t0]
    x_good = [y0]
    x_bad = [y0]

    while r_good.successful() and r_good.t < t1:
        t.append(r_good.t+dt)
        x_good.append(r_good.integrate(r_good.t+dt))
        x_bad.append(r_bad.integrate(r_bad.t+dt))

    x_good = np.array(x_good)
    x_bad = np.array(x_bad)
    t = np.array(t)

    import numpy as np


    fig, ax = plt.subplots()
    line1, = ax.plot(t, x_good[:,1],label='Using K from closest example')
    line2, = ax.plot(t, x_bad[:,1], label='Using K from incorrect example')
    ax.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('State variable')
    plt.show()


        # print(A_diff_1, A_diff_2)


