import scipy as sc
import numpy as np
from scipy.integrate import solve_ivp
import astropy.constants as const
import matplotlib.pyplot as plt
from numba import njit
import copy

if __name__ == '__main__':
    method_used = 'RK23'
    mass_sun = 1.989 * 10 ** (30)  # kg, massa zon
    mass_earth = 5.972 * 10 ** (26)  # kg, massa aarde
    G = copy.deepcopy(const.G.value)
    # mass = np.append(mass_sun, np.array([mass_earth for i in range(4)]))
    mass = np.array([1, 2])
    y0 = [100, 70, 0, 0]
    step = 1
    t = np.array((1,20))

    # @njit
    def equation_of_speed(t,vec, mass):
        F = np.zeros(int(len(vec)/2))
        r = vec[0:int(len(vec)/2)]
        for i in range(1,int(len(vec)/2)):
            F = F - 1/(r-np.roll(r,i))*np.roll(mass,i)

        a = F/mass

        dx = vec[int(len(vec)/2):]
        dv = a
        dvec = np.concatenate((dx,dv))
        return dvec


    dy = equation_of_speed(t[1], y0, mass)
    solution = solve_ivp(equation_of_speed, t_span=t, y0=y0, args=(mass,), max_step=step,
                         method=method_used)