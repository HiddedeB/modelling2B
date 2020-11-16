import scipy as sc
import numpy as np
from scipy.integrate import solve_ivp
import astropy.constants as const
import matplotlib.pyplot as plt
from numba import njit
import copy
from data import PlanetaryDataHandler
pdh = PlanetaryDataHandler



if __name__ == "__main__":
    # Example of 2D 2-body problem earth and sun

    # Parameters for the simulator
    time_frame = np.array([0, 365.25*24*3600], dtype=int)
    step = 1000
    method = 'RK23'
    absolute_tolerance = 1e5
    relative_tolerance = 1e4

    # Masses
    mass_earth = pdh.earth.mass
    mass_sun = 1.989e30
    mass = np.array([mass_earth, mass_sun], dtype=float)

    # Initial conditions
    au = 1.495e11
    G = copy.deepcopy(const.G.value)
    initial_position = [au, 0, 0, 0]
    initial_velocity = [0, 2.9765e4, 0, 0]
    initial_conditions = np.array(initial_position + initial_velocity, dtype=float)


    # @njit
    # def equation_of_speed(t, vec, mass, G):
    #     length = int(len(vec)/2)
    #     r = np.sqrt(vec[:length:2]**2+vec[1:length:2]**2)
    #     x, y = vec[:length:2], vec[1:length:2]
    #     a = np.zeros(length, dtype=float)
    #
    #     #maybe matrix voor for loop.
    #     for i in range(1, int(length/2)):
    #         # Calculation of V via gravitation force (Newtonian)
    #         theta = np.arctan2((y-np.roll(y, i)), (x-np.roll(x, i)))
    #
    #         a[::2] = a[::2] - np.roll(mass, i)*G/np.abs(r-np.roll(r, i))**2 * np.cos(theta)
    #         a[1::2] = a[1::2] - np.roll(mass, i)*G/np.abs(r-np.roll(r, i))**2 * np.sin(theta)
    #
    #     dv_total = a
    #     d_total = vec[length:]
    #
    #     d_vec = np.concatenate((d_total, dv_total))
    #     return d_vec
    #
    # # dy = equation_of_speed(time_frame[0], initial_conditions, mass, G)
    # solution = solve_ivp(equation_of_speed, t_span=time_frame, y0=initial_conditions, args=(mass, G), max_step=step,
    #                     method=method, rtol=relative_tolerance, atol=absolute_tolerance)
    # data = solution['y']
    # plt.figure()
    # plt.plot(data[0], data[1])
    # plt.plot(data[2], data[3])
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    # print(dy)

