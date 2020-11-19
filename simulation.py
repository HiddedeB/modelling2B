import scipy as sc
import numpy as np
from scipy.integrate import solve_ivp
import astropy.constants as const
import matplotlib.pyplot as plt
from numba import njit
import copy
from data import PlanetaryDataHandler
pdh = PlanetaryDataHandler()



if __name__ == "__main__":
    def static(value):
        return copy.deepcopy(value)

    def orbital_velocity(axis, mass, G):
        return np.sqrt(G/axis*mass)

    # Example of 2D 2-body problem earth and sun

    # Parameters for the simulator
    time_frame = np.array([0, 10*365.25*24*3600], dtype=int)
    step = 1000
    method = 'RK23'
    absolute_tolerance = 1e5
    relative_tolerance = 1e4

    # Masses
    mass_earth = static(pdh.earth.mass)
    mass_sun = static(pdh.sun.mass)
    mass_mars = static(pdh.mars.mass)
    mass_jupiter = static(pdh.jupiter.mass)
    mass_saturn = static(pdh.saturn.mass)

    mass = np.array([mass_earth,
                     mass_mars,
                     mass_jupiter,
                     mass_saturn,
                     mass_sun], dtype=float)

    # Initial conditions
    G = static(const.G.value)
    earth_axis = static(pdh.earth.smaxis)
    sun_axis = 0
    mars_axis = static(pdh.mars.smaxis)
    jupiter_axis = static(pdh.jupiter.smaxis)
    saturn_axis = static(pdh.saturn.smaxis)

    initial_position = [earth_axis, 0,
                        -mars_axis, 0,
                        jupiter_axis, 0,
                        -saturn_axis, 0,
                        sun_axis, 0]
    initial_velocity = [0, 2.9765e4,
                         0, -orbital_velocity(mars_axis, mass_sun, G),
                         0, orbital_velocity(jupiter_axis, mass_sun, G),
                         0, -orbital_velocity(saturn_axis, mass_sun, G),
                        0, 0]
    initial_conditions = np.array(initial_position + initial_velocity, dtype=float)

    @njit
    def equation_of_speed(t, vec, mass, G):
        length = int(len(vec)/2)
        r = np.sqrt(vec[:length:2]**2+vec[1:length:2]**2)
        x, y = vec[:length:2], vec[1:length:2]
        a = np.zeros(length)

        #maybe matrix voor for loop.
        for i in range(1, int(length/2)):
            # Calculation of V via gravitation force (Newtonian)
            theta = np.arctan2((y-np.roll(y, i)), (x-np.roll(x, i)))

            a[::2] = a[::2] - np.roll(mass, i)*G/np.abs(r-np.roll(r, i))**2 * np.cos(theta)
            a[1::2] = a[1::2] - np.roll(mass, i)*G/np.abs(r-np.roll(r, i))**2 * np.sin(theta)

        dv_total = a
        d_total = vec[length:]

        d_vec = np.concatenate((d_total, dv_total))
        return d_vec

    dy = equation_of_speed(time_frame[0], initial_conditions, mass, G)
    solution = solve_ivp(equation_of_speed, t_span=time_frame, y0=initial_conditions, args=(mass, G), max_step=step,
                        method=method, rtol=relative_tolerance, atol=absolute_tolerance)
    data = solution['y']
    plt.figure()
    for i in range(len(mass)):
        plt.plot(data[2*i], data[1+2*i])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

