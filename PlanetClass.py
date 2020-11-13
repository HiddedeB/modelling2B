import scipy as sc
import numpy as np
from scipy.integrate import solve_ivp
import astropy.constants as const
import matplotlib.pyplot as plt
from numba import njit
import copy


class planet:
    '''Class to hold all variables related to a certain planet'''

    def __init__(self, mass, radius, initial_position=0, initial_velocity=0, loanode=0, period=0, name="", eccentricity=0,
                 smaxis=0, argperiapsis=0):
        '''NOTE:
        :param initial_position: initial position vector
        :type initial_position: ndarray TODO fix data type
        :param initial_velocity: initial velocity array
        :type initial_velocity: ndarray
        :param float mass: mass of the planet
        :type mass: float
        :param float radius: radius of the planet
        :type radius: float
        :param name: name of the object
        :type name: str
        :param eccentricity: eccentricity of the orbit of the planet
        :type eccentricity: float
        :param smaxis: ?? TODO update this
        :type smaxis: float
        :param period: period of the orbit
        :type period: float
        :param loanode: ?? TODO update this
        :type loanode: float
        :param argperiapsis: ?? TODO update this
        :type argperiapsis: float
        '''
        self.history = []
        self.pos = initial_position
        self.mass = mass
        self.radius = radius
        self.velocity = initial_velocity
        self.name = name
        self.e = eccentricity
        self.smaxis = smaxis
        self.period = period
        self.loanode = loanode
        self.argperiapsis = argperiapsis

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, updated_pos):
        self._pos = updated_pos
        self._history = self._history.append(updated_pos)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, updated_velocity):
        self._velocity = updated_velocity

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, var):
        self._history = var

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, var):
        self._name = var

    @property
    def e(self):
        return self._e

    @e.setter
    def e(self, value):
        self._e = value

    @property
    def smaxis(self):
        return self._smaxis

    @smaxis.setter
    def smaxis(self, var):
        self._smaxis = var

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, value):
        self._period = value

    @property
    def orbital_inclination(self):
        return self._orbital_inclination

    @orbital_inclination.setter
    def orbital_inclination(self, value):
        self._orbital_inclination = value

    @property
    def argperiapsis(self):
        return self._argperiapsis

    @argperiapsis.setter
    def argperiapsis(self, var):
        self._argperiapsis = var

    @property
    def loanode(self):
        return self._loanode

    @loanode.setter
    def loanode(self, var):
        self._loanode = var


if __name__ == "__main__":
    # Example of 2D 2-body problem earth and sun
    initial_vector_earth = np.array([const.au.value, 0],dtype=float)
    initial_velocity_earth = np.array([0, np.sqrt(const.M_sun.value*const.G.value/const.R_earth.value)], dtype=float)
    initial_vector_sun = np.array([-const.M_earth.value*const.au.value/const.M_sun.value, 0], dtype=float)
    initial_velocity_sun = np.array([0,0],dtype=float)
    earth = planet(mass=const.M_earth.value, initial_position=initial_vector_earth, radius=const.R_earth.value,
                   initial_velocity=initial_velocity_earth)
    sun = planet(mass=const.M_sun.value, initial_position=initial_vector_sun, radius=const.R_sun.value,
                initial_velocity=initial_velocity_sun)
    y_test = earth.pos
    for i in range(3):
        y_test = np.append((i+2)*y_test, earth.pos)
    y0 = np.concatenate([y_test, sun.pos])
    time_frame = (0, 4*10**9)
    step = 1000
    method_used = 'RK23'
    relative_tolerance = 100
    absolute_tolerance = 10**4
    mass = [earth.mass for i in range(4)]

    x = copy.deepcopy(const.au.value)
    position = np.array([x, 0],dtype=float)
    y_test = position
    for i in range(3):
        y_test = np.append(y_test, (i+2)*position)
    mass_sun = 1.989 * 10 ** (30)  # kg, massa zon
    mass_earth = 5.972 * 10 ** (26)  # kg, massa aarde
    G = copy.deepcopy(const.G.value)
    mass = np.append(mass_sun, np.array([mass_earth for i in range(4)]))

    @njit
    def equation_of_speed(t,y, mass, G):
        r = np.sqrt(y[::2]**2+y[1::2]**2)
        v = np.zeros(r.shape[0])
        dx, dy = np.zeros(r.shape[0]), np.zeros(r.shape[0])
        d_total = np.zeros(2*r.shape[0])
        theta = np.arctan2(y[1::2], y[::2])

        for i in range(1, r.shape[0]):
            # Calculation of V via centrifugal and gravitation force (Newtonian
            v = v + np.roll(mass,i)*G*(r-np.roll(r, i))/np.abs(r-np.roll(r,i))**3

        v = v/mass
        v[:-1] = v[:-1] + 2 * mass[-1] * G/np.abs(r[:-1]-r[-1]) # Adding the extra substracted term back in and adding
        # another.
        dx[:-1] = -v[:-1]*np.sin(theta[:-1])
        dy[:-1] = v[:-1]*np.cos(theta[:-1])

        # Sun calculations
        v[-1] = -np.sum(mass[:-1]*v[:-1])/mass[-1]

        dx[-1] = -v[-1]*np.sin(theta[-1])
        dy[-1] = v[-1]*np.cos(theta[-1])
        d_total[::2] = dx
        d_total[1::2] = dy
        return d_total

    solution = solve_ivp(equation_of_speed, t_span=time_frame, y0=y0, args=(mass, G), max_step=step,
                         method=method_used, rtol=relative_tolerance, atol=absolute_tolerance)

    plt.figure()
    data = solution['y']
    for i in range(4):
        plt.plot(data[2*i], data[2*i+1])
    plt.plot(data[-1], data[-2], label='sun_orbit')
    plt.show()