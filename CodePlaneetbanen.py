import numpy as np
from scipy.integrate import solve_ivp
import scipy as sp
import astropy.constants as const
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
import copy
from data import PlanetaryDataHandler

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

def plotelips3d(theta,eccentricity,smaxis,Omega,omega,I, input):
    X,Y,Z,vector = xyz(theta,eccentricity,smaxis,Omega,omega, I)
    ax.plot(X,Y,Z, c=input)
    plt.show()
    return vector
# Making plot for 0 to 2pi for Mercury
theta_values = np.linspace(0,2*np.pi,10**3)
m=pdh.mercury
vector_mercury = plotelips3d(theta_values,0.2056,m.smaxis,m.loanode,
                             m.argperiapsis, np.deg2rad(m.orbital_inclination), 'r')

# Making plot for 0 to 2pi for Venus
v=pdh.venus
vector_venus = plotelips3d(theta_values,0.006772,v.smaxis,v.loanode,
                           v.argperiapsis, np.deg2rad(v.orbital_inclination),'grey')

# Right hand side of the system of differential equations
def Matrixelementcalc(n_1,n_2,m_1,m_2,m_c,alpha12,b_1,b_2):
    #A-elements matrix
    A_11 = n_1 * (1/4) * (m_2/(m_c + m_1)) * np.abs(alpha12)**2 * b_1
    A_22 = n_2 * (1/4) * (m_1/(m_c + m_2)) * np.abs(alpha12)**2 * b_1
    A_12 = -1*n_1 * (1/4) * (m_2/(m_c + m_1)) * np.abs(alpha12)**2 * b_2
    A_21 = -1*n_2 * (1/4) * (m_1/(m_c + m_2)) * np.abs(alpha12)**2 * b_2

    #B_elements matrix
    B_11 = -1*n_1 * (1 / 4) * (m_2 / (m_c + m_1)) * np.abs(alpha12) ** 2 * b_1
    B_22 = -1*n_2 * (1 / 4) * (m_1 / (m_c + m_2)) * np.abs(alpha12) ** 2 * b_1
    B_12 =  n_1 * (1 / 4) * (m_2 / (m_c + m_1)) * np.abs(alpha12) ** 2 * b_1
    B_21 =  n_2 * (1 / 4) * (m_1 / (m_c + m_2)) * np.abs(alpha12) ** 2 * b_1

    return np.array([A_11,A_12,A_21,A_22]), np.array([B_11,B_12,B_21,B_22])

def rhsf(smaxis1,smaxis2,a_1,a_2,m_1,m_2,m_c,vector_1,vector_2):

    # Berekenen van de angle phi van de r van Venus en Mercury aan de hand van de bekende 6 parameters
    alpha12 = smaxis1/smaxis2
    n_1 = 2*np.pi/(a_1**(3/2))
    n_2 = 2*np.pi/(a_2**(3/2))

    vector_1_normed = vector_1/np.linalg.norm(vector_1)
    vector_2_normed = vector_2 / np.linalg.norm(vector_2)
    phi = np.cos(np.dot(vector_1_normed,vector_2_normed))

    #defining the integrable constant b_twothirds for the system of ODE
    integrabel_function_1 = lambda phi: np.cos(phi)/(1-2*alpha12*np.cos(phi) + alpha12**2)**(3/2) #angle phi
    integrabel_function_2 = lambda phi: np.cos(2*phi)/(1-2*alpha12*np.cos(2*phi) + alpha12**2)**(3/2) #angle 2*phi
    b_twothirds_one = (1/np.pi) * sp.intgrate_quad(integrabel_function_1, 0 ,2*np.pi)
    b_twothirds_two =  (1/np.pi) * sp.intgrate_quad(integrabel_function_2, 0, 2*np.pi)

    vec_1,vec_2 = Matrixelementcalc(n_1,n_2,m_1,m_2,m_c,alpha12,b_twothirds_one,b_twothirds_two)






