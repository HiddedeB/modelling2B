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
steps = 5
results = TBP.ODE_solv(m.smaxis,v.smaxis,m.mass,v.mass,pdh.sun.mass,0.2056,0.006772,
         m.loanode,v.loanode,np.deg2rad(m.orbital_inclination),np.deg2rad(v.orbital_inclination),
         m.argperiapsis,v.argperiapsis, time = tijd, t_ev = None)
t = np.linspace(0,tijd,steps)
z=results.sol(t)


plt.ioff()      #automatisch plots showen uit zetten
theta_values = np.linspace(0, 2 * np.pi, 10 ** 3)
original_system_mercury = [None]*steps
original_system_venus = [None]*steps
for i in range(steps):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.xlim([-2*10**11, 2*10**11])
    plt.ylim([-2*10**11, 2*10**11])
    ax.set_zlim([-2*10**10, 2*10**10])

    original_system_mercury[i] = TBP.variable_transfromations(z.T[i][0], z.T[i][2],
                                                       z.T[i][4], z.T[i][6])
    original_system_venus[i] = TBP.variable_transfromations(z.T[i][1], z.T[i][3],
                                                       z.T[i][5], z.T[i][7])

    Od.plot_elips_3d(theta_values,original_system_mercury[i][0],m.smaxis,original_system_mercury[i][3],original_system_mercury[i][2],original_system_mercury[i][1],'r',ax)
    Od.plot_elips_3d(theta_values,original_system_venus[i][0],v.smaxis,original_system_venus[i][3],original_system_venus[i][2],original_system_venus[i][1],'grey',ax)
    ax.plot(0,0,0,'ko')
    plt.savefig('animatie/testje{}.png'.format(i))
    plt.close(fig)

##################### andere animatie methode

from matplotlib import animation

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.set_xlim([-2 * 10 ** 11, 2 * 10 ** 11])
ax.set_ylim([-2 * 10 ** 11, 2 * 10 ** 11])
ax.set_zlim([-2 * 10 ** 10, 2 * 10 ** 10])

def init():
    ellips.set_data([],[])
    return ellips


def animate(i):
    X, Y, Z, vector = Od.xyz(theta_values,original_system_mercury[i][0],m.smaxis,original_system_mercury[i][3],original_system_mercury[i][2],original_system_mercury[i][1])
    ellips.set_data(X,Y)
    ellips.set_3d_properties(np.array(Z))
    return ellips,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=steps, interval=20, blit=True)




