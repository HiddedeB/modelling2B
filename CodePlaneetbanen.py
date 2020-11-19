import numpy as np
from scipy.integrate import solve_ivp
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
import copy
from data import PlanetaryDataHandler
import time

pdh = PlanetaryDataHandler()

# Plotting the ellips given de known six properties

# Function for finding the right distance from the origin at a given moment
def r(theta, eccentricity, smaxis,omega):
    b_squared = (1-eccentricity**2)*smaxis**2
    c = eccentricity * smaxis

    return b_squared / (smaxis + c * np.cos(theta - omega))

# The x,y,z-coordinates of planetary motion for the right lagrange parameters
def xyz(theta,eccentricity,smaxis,Omega,omega,I):
    x = r(theta, eccentricity, smaxis, omega)*(np.cos(Omega)*np.cos(omega+theta) -
                                               np.sin(Omega)*np.sin(omega+theta)*np.cos(I))
    y = r(theta, eccentricity, smaxis, omega)*(np.sin(Omega)*np.cos(omega+theta) +
                                               np.cos(Omega)*np.sin(omega+theta)*np.cos(I))
    z = r(theta, eccentricity, smaxis, omega)*(np.sin(omega+theta) * np.sin(I))

    return x,y,z, np.array([x,y,z])

# Plot function for 3d plots of the ellipses
fig = plt.figure()
ax = fig.gca(projection='3d')


def plot_elips_3d(theta,eccentricity,smaxis,Omega,omega,I, input):
    X,Y,Z,vector = xyz(theta,eccentricity,smaxis,Omega,omega, I)
    ax.plot(X,Y,Z, c=input)
    #plt.show()
    return vector

# Making plot for 0 to 2pi for Mercury
theta_values = np.linspace(0,2*np.pi,10**3)
m=pdh.mercury
vector_mercury = plot_elips_3d(theta_values,0.2056,m.smaxis,m.loanode,
                             m.argperiapsis, np.deg2rad(m.orbital_inclination), 'r')

# Making plot for 0 to 2pi for Venus
v=pdh.venus
vector_venus = plot_elips_3d(theta_values,0.006772,v.smaxis,v.loanode,
                           v.argperiapsis, np.deg2rad(v.orbital_inclination),'grey')

# Right hand side of the system of differential equations
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





# Probeersel voor het vinden van de juiste parameters


#tijd = 3*10**14*365.25*24*60*60
tijd = 3*10**16*365.25*24*60*60
results = ODE_solv(m.smaxis,v.smaxis,m.mass,v.mass,pdh.sun.mass,0.2056,0.006772,
         m.loanode,v.loanode,np.deg2rad(m.orbital_inclination),np.deg2rad(v.orbital_inclination),
         m.argperiapsis,v.argperiapsis, time = tijd, t_ev = None)


t = np.linspace(0,tijd,500)
z=results.sol(t)

# Kijken of de parameters op t=1000 sense maken in het originele coordinate system

original_system_mercury = variable_transfromations(z.T[-1][0], z.T[-1][2],
                                                   z.T[-1][4], z.T[-1][6])

fig = plt.figure()
ax = fig.gca(projection='3d')
theta_values = np.linspace(0,2*np.pi,10**3)
plt.ioff()
for i in range(np.size(z[0])):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_zticks([-10**10, 0, 10**10])
    # ax.set_xticks([-10**10, 0, 10**10])
    # ax.set_yticks([-10**10, 0, 10**10])
    plt.xlim([-2*10**11, 2*10**11])
    plt.ylim([-2*10**11, 2*10**11])
    ax.set_zlim([-2*10**10, 2*10**10])

    original_system_mercury = variable_transfromations(z.T[i][0], z.T[i][2],
                                                       z.T[i][4], z.T[i][6])
    original_system_venus = variable_transfromations(z.T[i][1], z.T[i][3],
                                                       z.T[i][5], z.T[i][7])

    plot_elips_3d(theta_values,original_system_mercury[0],m.smaxis,original_system_mercury[3],original_system_mercury[2],original_system_mercury[1],'r')
    plot_elips_3d(theta_values,original_system_venus[0],v.smaxis,original_system_venus[3],original_system_venus[2],original_system_venus[1],'grey')
    ax.plot(0,0,0,'ko')
    plt.savefig('animatie/testje{}.png'.format(i))
    plt.close(fig)

    #plt.close(fig)


