# author: Filippos Sotiropoulos (fes@mit.edu)

import numpy as np 
import scipy 
from scipy.integrate import ode
import control
import matplotlib.pyplot as plt

def f1_good(t, y):
    return np.matmul((A3-B3*K1),np.transpose(np.array(y)))

def f1_bad(t, y):
    return np.matmul((A3-B3*K2),np.transpose(np.array(y)))


m1 = 1.0
m2 = 1.0
k1 = 1.0
k2 = 1.0



def f_nonlinear(t,y,i):
    # x = q1 q2 p1 p2
    
    q1 = y[0]
    q2 = y[1]
    p1 = y[2]
    p2 = y[3]

    u = 0.0

    q1dot = (1/m1)*p1 -(1/m2)*p2
    q2dot = (1/m2)*p2
    p1dot = u - k1*q1
    p2dot = k1*q1 - k2*q2 - phi_R1((1/m2)*p2,i)

    return [q1dot,q2dot,p1dot,p2dot]

def phi_R1(f,i=0):
    c = 1.0
    if i == 0:
        eR1 = c*f*np.abs(f)
    else:
        eR1 = c*np.sign(f)*np.sqrt(np.abs(f))

    return eR1


if __name__ == '__main__':

    y0, t0 = [1.0, 1.0, 0.0, 0.0], 0
    r = ode(f_nonlinear).set_integrator('dopri5', method='bdf')

    r.set_initial_value(y0, t0).set_f_params(1)

    t1 = 10
    dt = 0.02
    t = [t0]
    x = [y0]

    while r.successful() and r.t < t1:
        t.append(r.t+dt)
        x.append(r.integrate(r.t+dt))

    x = np.array(x)
    t = np.array(t)

    import numpy as np


    fig, ax = plt.subplots()
    line1, = ax.plot(t, x[:,1],label='Using K from closest example')
    # ax.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('State variable')
    plt.show()


        # print(A_diff_1, A_diff_2)


