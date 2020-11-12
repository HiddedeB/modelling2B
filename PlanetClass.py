import scipy as sc
import numpy as np
from scipy.integrate import solve_ivp
import astropy.constants as const
import matplotlib.pyplot as plt
from numba import njit
import copy

class planet:
    '''Class to hold all variables related to a certain planet'''
    def __init__(self,
                 initial_position,
                 mass,
                 radius,
                 initial_velocity):
        '''NOTE:
        :param initial_position: initial position vector
        :type initial_position: ndarray
        :param float mass: mass of the planet
        :type mass: float
        :param float radius: radius of the planet
        :type radius: float
        TODO: planeten misschien een nullable naam geven, baanargumenten zoals eccentriciteit enzo ook als variabelen
        '''
        self._history = []
        self.pos = initial_position
        self.mass = mass
        self.radius = radius
        self.velocity = initial_velocity

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
    mass = np.array([mass_earth for i in range(4)])

    @njit
    def equation_of_speed(t,y, mass, G, mass_sun):
        equation = np.zeros(int(y.shape[0])-2)
        v_sun = 0
        for i in range(int(y.shape[0]/2-1)):
            # Calculation of V via centrifugal and gravitation force (Newtonian)
            v_i = np.sqrt(mass_sun*G/np.linalg.norm(np.array([y[2*i],y[2*i+1]])))
            m_i = mass[i]

            # Determine theta and change
            theta = np.arctan2(y[2*i+1],y[2*i])
            dx_i = -v_i*np.sin(theta)
            dy_i = v_i*np.cos(theta)
            equation[2*i] = dx_i
            equation[2*i+1] = dy_i

            v_sun += -v_i*m_i/mass_sun
        # Sun calculations
        stheta = np.arctan2(y[-2],y[-1])
        dx_sun = -v_sun*np.sin(stheta)
        dy_sun = v_sun*np.cos(stheta)
        equation_sun = np.zeros(2)
        equation_sun[0] = dx_sun
        equation_sun[1] = dy_sun
        return np.hstack((equation, equation_sun))

    solution = solve_ivp(equation_of_speed, t_span=time_frame, y0=y0, args=(mass, G, mass_sun), max_step=step,
                         method=method_used, rtol=relative_tolerance, atol=absolute_tolerance)

    plt.figure()
    data = solution['y']
    for i in range(4):
        plt.plot(data[2*i], data[2*i+1])
    plt.plot(data[-1], data[-2], label='sun_orbit')
    plt.show()