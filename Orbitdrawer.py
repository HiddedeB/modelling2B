import numpy as np
import matplotlib.pyplot as plt

# mercury Data
mass_m = 0.33e24 #kg
radius_m = 2.44e6 #m
smaxis_m = 57.91e9 #m
period_m = 0.241 #year
orbital_inclination_m = 7 #degrees
eccentricity_m=0.2056 #no units
loanode_m=48.331 #degrees
argperiapsis_m =29.124 #degrees

# Data venus
mass_v=4.8675e24 #kg
radius_v=6.05e6 #m
smaxis_v=108.2e9 #m
period_v=0.615198 #year
orbital_inclination_V=3.394 #degrees
eccentricity_v = 0.006772 #no units
loanode_v =76.68 #degrees
argperiapsis_v=54.884 #degrees


# Plotting the ellips given de known six properties

#function for finding the right distance from the origin at a given moment
def r(theta, eccentricity, smaxis,omega):
    b_squared = (1-eccentricity**2)*smaxis**2
    c = eccentricity * smaxis
    return b_squared / (smaxis + c * np.cos(theta - omega))

#the x,y,z-coordinates of planetary motion for the right lagrange parameters
def xyz(theta,eccentricity,smaxis,Omega,omega,I):
    x = r(theta, eccentricity, smaxis, omega)*(np.cos(Omega)*np.cos(omega+theta) - np.sin(Omega)*np.sin(omega+theta)*np.cos(I))
    y = r(theta, eccentricity, smaxis, omega)*(np.sin(Omega)*np.cos(omega+theta) + np.cos(Omega)*np.sin(omega+theta)*np.cos(I))
    z = r(theta, eccentricity, smaxis, omega)*(np.sin(omega+theta) * np.sin(I))
    return x,y,z

#Plot function for 3d plots of the ellipses
def plotelips3d(theta,eccentricity,smaxis,Omega,omega,I):
    X,Y,Z = xyz(theta,eccentricity,smaxis,Omega,omega, I)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X,Y,Z, c='red')
    plt.show()



#making plot for 0 to 2pi
theta_values = np.linspace(0,2*np.pi,1000)
plotelips3d(theta_values,eccentricity_m,smaxis_m,loanode_m,argperiapsis_m, orbital_inclination_m)