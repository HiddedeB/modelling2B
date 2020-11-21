import Orbitdrawer as Od
import TwoBodyProblem as TBP
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

m=pdh.mercury
v=pdh.venus


tijd = 3*10**16*365.25*24*60*60
results = TBP.ODE_solv(m.smaxis,v.smaxis,m.mass,v.mass,pdh.sun.mass,0.2056,0.006772,
         m.loanode,v.loanode,np.deg2rad(m.orbital_inclination),np.deg2rad(v.orbital_inclination),
         m.argperiapsis,v.argperiapsis, time = tijd, t_ev = None)


t = np.linspace(0,tijd,500)
z=results.sol(t)

# Kijken of de parameters op t=1000 sense maken in het originele coordinate system

original_system_mercury = TBP.variable_transfromations(z.T[-1][0], z.T[-1][2],
                                                   z.T[-1][4], z.T[-1][6])

fig = plt.figure()
ax = fig.gca(projection='3d')
theta_values = np.linspace(0,2*np.pi,10**3)
plt.ioff()
for i in range(np.size(z[0])):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.xlim([-2*10**11, 2*10**11])
    plt.ylim([-2*10**11, 2*10**11])
    ax.set_zlim([-2*10**10, 2*10**10])

    original_system_mercury = TBP.variable_transfromations(z.T[i][0], z.T[i][2],
                                                       z.T[i][4], z.T[i][6])
    original_system_venus = TBP.variable_transfromations(z.T[i][1], z.T[i][3],
                                                       z.T[i][5], z.T[i][7])

    Od.plot_elips_3d(theta_values,original_system_mercury[0],m.smaxis,original_system_mercury[3],original_system_mercury[2],original_system_mercury[1],'r')
    Od.plot_elips_3d(theta_values,original_system_venus[0],v.smaxis,original_system_venus[3],original_system_venus[2],original_system_venus[1],'grey')
    ax.plot(0,0,0,'ko')
    plt.savefig('animatie/testje{}.png'.format(i))
    plt.close(fig)

