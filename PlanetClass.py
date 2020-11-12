import scipy as sc
import numpy as np
from scipy.integrate import solve_ivp
import astropy.constants as const

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

    y0 = np.concatenate([earth.pos, sun.pos])
    time_frame = (0, 10**8)
    step = 500
    method_used = 'RK45'
    relative_tolerance = 100
    absolute_tolerance = 10**4

    def equation_of_speed(t,y):
        # Calculation of V via centrifugal and gravitation force (Newtonian)
        v_earth = np.sqrt(sun.mass*const.G.value/np.linalg.norm(np.array([y[0],y[1]])))

        # Determine theta and change
        theta = np.arctan2(y[1],y[0])
        dx_earth = -v_earth*np.sin(theta)
        dy_earth = v_earth*np.cos(theta)

        # Sun calculations
        stheta = np.arctan2(y[3],y[2])
        v_sun = -v_earth*earth.mass/sun.mass
        dx_sun = -v_sun*np.sin(stheta)
        dy_sun = v_sun*np.cos(stheta)
        equation = [dx_earth, dy_earth, dx_sun, dy_sun]
        return np.hstack((equation))

    solution = solve_ivp(equation_of_speed, time_frame, y0, max_step=step, method=method_used, rtol=relative_tolerance,
                         atol=absolute_tolerance)
    plt.figure()
    data = solution['y']
    plt.plot(data[0], data[1], label='earth_orbit')
    plt.plot(data[2], data[3], label='sun_orbit')
    plt.show()