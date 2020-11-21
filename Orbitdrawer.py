import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# theta_values = np.linspace(0,2*np.pi,10**3)


def plot_elips_3d(theta,eccentricity,smaxis,Omega,omega,I, input,figure):
    ''' Functie die de elipsbaan van een planeet tekent in 3D. Als je veel plotjes wilt opslaan kan je beter plt.show() weghalen
    NOTE:
    :param theta: hoeken van ellips die getekend moeten worden. voor gehele ellipse vector van 0 tot 360 graden
    :param eccentricity: eccentriciteit van de ellipse
    :param smaxis: to do
    :param Omega: to do
    :param omega: to do
    :param I: to do
    :param input: kleur voor getekende ellipse
    :param figure: figure waarin je ellipse wilt tekenen
    '''
    X,Y,Z,vector = xyz(theta,eccentricity,smaxis,Omega,omega, I)

    figure.plot(X,Y,Z, c=input)

    #plt.show()
    return vector
