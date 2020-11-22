from numba import types, typed, typeof
from numba.experimental import jitclass


class planet:
    '''Class to hold all variables related to a certain planet'''

    def __init__(self, mass, radius, initial_position=0, initial_velocity=0, loanode=0, period=0, name="", eccentricity=0,
                 smaxis=0, argperiapsis=0, orbital_inclination=0, mean_longitude=0):
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
        :param orbital_inclination: ?? minor TODO update this
        :type orbital_inclination: float
        :param mean_longitude: ?? minor TODO update this
        :type mean_longitude: float
        :param longitude_periapsis: sum of loanode and argperiapsis
        :type longitude_periapsis: float
        '''
        self.history = np.array([], dtype=np.float64)
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
        self.mean_longitude = mean_longitude
        self.longitude_periapsis = self.loanode + self.argperiapsis

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, updated_pos):
        self._pos = updated_pos
        self._history = np.concatenate(self._history, updated_pos)

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

    @property
    def mean_longitude(self):
        return self._mean_longitude

    @mean_longitude.setter
    def mean_longitude(self,var):
        self._mean_longitude = var

    @property
    def longitude_periapsis(self):
        return self._longitude_periapsis

    @longitude_periapsis.setter
    def longitude_periapsis(self, var):
        self._longitude_periapsis = var
    
    def __str__(self):
        return "Instance of planet " + self.name