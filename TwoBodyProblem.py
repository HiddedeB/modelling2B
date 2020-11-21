import numpy as np
from scipy.integrate import solve_ivp
import scipy as sp


def matrix_element_calc(n_1,n_2,m_1,m_2,m_c,alpha12,b_1,b_2):
    #A-elements matrix
    A_11 = n_1 * (1/4) * (m_2/(m_c + m_1)) * np.abs(alpha12)**2 * b_1
    A_22 = n_2 * (1/4) * (m_1/(m_c + m_2)) * np.abs(alpha12)**2 * b_1
    A_12 = -1*n_1 * (1/4) * (m_2/(m_c + m_1)) * np.abs(alpha12)**2 * b_2
    A_21 = -1*n_2 * (1/4) * (m_1/(m_c + m_2)) * np.abs(alpha12)**2 * b_2

    #B_elements matrix
    B_11 = -1*n_1 * (1/4) * (m_2 / (m_c + m_1)) * np.abs(alpha12) ** 2 * b_1
    B_22 = -1*n_2 * (1/4) * (m_1 / (m_c + m_2)) * np.abs(alpha12) ** 2 * b_1
    B_12 =  n_1 * (1/4) * (m_2 / (m_c + m_1)) * np.abs(alpha12) ** 2 * b_1
    B_21 =  n_2 * (1/4) * (m_1 / (m_c + m_2)) * np.abs(alpha12) ** 2 * b_1

    return np.array([A_11,A_12,A_21,A_22]), np.array([B_11,B_12,B_21,B_22])

# System of equations for solving the ODE
def sys_of_eq(t,y,A11,A12,A21,A22,B11,B12,B21,B22):
    h1,h2,k1,k2,p1,p2,q1,q2 = y

    return np.array([A11*k1 + A12*k2, A21*k1+A22*k2, -A11*h1 - A12*h2, -A21*h1-A22*h2,
            B11*q1 + B12*q2, B21*q1+B22*q2, -B11*p1 - B12*p2, -B21*p1-B22*p2])

# Arguments to pass on to the ODE Solver
def args(smaxis1,smaxis2,m_1,m_2,m_c):

    # Berekenen van de angle phi van de r van Venus en Mercury aan de hand van de bekende 6 parameters
    alpha12 = smaxis1/smaxis2
    n_1 = 2*np.pi/(smaxis1**(3/2))
    n_2 = 2*np.pi/(smaxis2**(3/2))

    # Defining the integrable constant b_twothirds for the system of ODE
    integrabel_function_1 = lambda phi: np.cos(phi)/(1-2*alpha12*np.cos(phi) + alpha12**2)**(3/2) #angle phi
    integrabel_function_2 = lambda phi: np.cos(2*phi)/(1-2*alpha12*np.cos(2*phi) + alpha12**2)**(3/2) #angle 2*phi
    b_twothirds_one = (1/np.pi) * sp.integrate.quad(integrabel_function_1, 0 ,2*np.pi)[0]
    b_twothirds_two =  (1/np.pi) * sp.integrate.quad(integrabel_function_2, 0, 2*np.pi)[0]

    vec_1,vec_2 = matrix_element_calc(n_1,n_2,m_1,m_2,m_c,alpha12,b_twothirds_one,b_twothirds_two)

    return (vec_1[0],vec_1[1],vec_1[2],vec_1[3],
                   vec_2[0],vec_2[1],vec_2[2],vec_2[3])

# Initial condition calculator
def initial_conditions(e1,e2,omega1,omega2,I1,I2,Omega1,Omega2):
    return np.array([e1*np.sin(omega1),e2*np.sin(omega2),
            e1*np.cos(omega1),e2*np.cos(omega2),
            I1*np.sin(Omega1),I2*np.sin(Omega2),
            I1*np.cos(Omega1),I2* np.cos(Omega2)])

# Solving the system of differential equations
def ODE_solv(smaxis1,smaxis2,m_1,m_2,m_c,e1,e2,omega1,omega2,I1,I2,Omega1,Omega2, time, t_ev):
    IC = initial_conditions(e1,e2,omega1,omega2,I1,I2,Omega1,Omega2)
    return solve_ivp(sys_of_eq,[0,time], IC, args= args(smaxis1,smaxis2,m_1,m_2,m_c),t_eval=t_ev, dense_output=True)

# Transforming the variables back to usual coordinate system, returns e,I,omega,Omega
def variable_transfromations(h,k,p,q):
    return np.array([np.sqrt(h**2+k**2), np.sqrt(p**2+q**2),
                    np.arcsin(h/np.sqrt(h**2+k**2)), np.arcsin(p/np.sqrt(p**2+q**2))])
