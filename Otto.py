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
import mpl_toolkits.mplot3d.axes3d as p3

pdh = PlanetaryDataHandler()

m=pdh.mercury
v=pdh.venus


tijd = 3*10**14*365.25*24*60*60
steps = 100
results = TBP.ODE_solv(m.smaxis,v.smaxis,m.mass,v.mass,pdh.sun.mass,0.2056,0.006772,
         m.loanode,v.loanode,np.deg2rad(m.orbital_inclination),np.deg2rad(v.orbital_inclination),
         m.argperiapsis,v.argperiapsis, time = tijd, t_ev = None)
t = np.linspace(0,tijd,steps)
z=results.sol(t)


theta_values = np.linspace(0, 2 * np.pi, 10 ** 3)
original_system_mercury = [None]*steps
original_system_venus = [None]*steps
for i in range(steps):
    original_system_mercury[i] = TBP.variable_transfromations(z.T[i][0], z.T[i][2],
                                                       z.T[i][4], z.T[i][6])
    original_system_venus[i] = TBP.variable_transfromations(z.T[i][1], z.T[i][3],
                                                       z.T[i][5], z.T[i][7])

from matplotlib import animation

#Animeren van twee objecten
# fig = plt.figure()
# ax = p3.Axes3D(fig)
# ax.set_xlim3d([-2 * 10 ** 11, 2 * 10 ** 11])
# ax.set_xlabel('X')
# ax.set_ylim3d([-2 * 10 ** 11, 2 * 10 ** 11])
# ax.set_ylabel('Y')
# ax.set_zlim3d([-2 * 10 ** 10, 2 * 10 ** 10])
# ax.set_zlabel('Z')
# ax.set_title('3D Test')
#
# line1 = ax.plot3D([],[],[])
# line2 = ax.plot3D([],[],[])
#
#
# def animate(i,line1,line2):
#     X, Y, Z, vector = Od.xyz(theta_values,original_system_mercury[i][0],m.smaxis,original_system_mercury[i][3],original_system_mercury[i][2],original_system_mercury[i][1])
#     line1[0].set_data([X, Y])
#     line1[0].set_3d_properties(Z)
#     X, Y, Z, vector = Od.xyz(theta_values,original_system_venus[i][0],v.smaxis,original_system_venus[i][3],original_system_venus[i][2],original_system_venus[i][1])
#     line2[0].set_data([X, Y])
#     line2[0].set_3d_properties(Z)
#
#
#
# anim = animation.FuncAnimation(fig, animate, fargs = (line1,line2),
#                                frames=steps, interval=10, blit=False)
#
# plt.show()


#animeren van N objecten + voorbeeldje mbv mercury en venus
planeetparameters = [original_system_mercury,original_system_venus]
smallaxis = [m.smaxis,v.smaxis]

fig1, animate, plotobjecten = Od.animatieN(planeetparameters,smallaxis,steps)
anim = animation.FuncAnimation(fig1, animate, fargs=(plotobjecten, planeetparameters, smallaxis),
                               frames=steps, interval=10, blit=False)

