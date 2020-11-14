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
                 smaxis=0, argperiapsis=0, orbital_inclination=0):
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
        self.orbital_inclination = orbital_inclination

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

    #Parameters for the simulator
    time_frame = np.array([0, 10**7], dtype=int)
    step = 1000
    method = 'RK23'
    absolute_tolerance = 1e7
    relative_tolerance = 1e4

    #Masses
    mass_earth = 5.972e24
    mass_sun = 1.989e30
    mass = np.array([mass_earth, mass_sun], dtype=float)

    # Initial conditions
    au = 1.495e10
    G = copy.deepcopy(const.G.value)
    initial_position = [au, au, 0, 0]
    initial_velocity = [0, 120e6, 0, 0]
    initial_conditions = np.array(initial_position + initial_velocity, dtype=float)


    @njit
    def equation_of_speed(t,vec, mass, G):
        length = int(len(vec)/2)
        r = np.sqrt(vec[:length:2]**2+vec[1:length:2]**2)
        x, y = vec[:length:2], vec[1:length:2]
        F = np.zeros(length)
        dv_total = np.zeros(length)

        for i in range(1, int(length/4)+1):
            # Calculation of V via gravitation force (Newtonian)
            theta = np.arctan2((y-np.roll(y, i)), (x-np.roll(x, i)))
            F[::2] = F[::2] + np.roll(mass, i)*G/np.abs(r-np.roll(r, i))**2 * np.sin(theta)
            F[1::2] = F[1::2] + np.roll(mass, i)*G/np.abs(r-np.roll(r, i))**2 * np.cos(theta)

        dv_total[::2] = F[::2]/mass
        dv_total[1::2] = F[1::2]/mass
        d_total = vec[2*r.shape[0]:]

        d_vec = np.concatenate((d_total, dv_total))
        return d_vec             #TODO fix the fact that we now have to return an array for [y', v']

    dy = equation_of_speed(time_frame[0], initial_conditions, mass, G)
    solution = solve_ivp(equation_of_speed, t_span=time_frame, y0=initial_conditions, args=(mass, G), max_step=step,
                         method=method, rtol=relative_tolerance, atol=absolute_tolerance)
    data = solution['y']
    plt.figure()
    plt.plot(data[0], data[1])
    plt.plot(data[2], data[3])