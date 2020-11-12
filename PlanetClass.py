import numpy as np
import scipy as sc


class Planet:
    '''Class to hold all variables related to a certain planet'''
    def __init__(self,
                 initial_position,
                 mass,
                 radius,
                 initial_velocity):
        '''NOTE:
        :param initial_position: initial position vector
        :type initial_position: array
        :param float mass: mass of the planet
        :type mass: float
        :param float radius: radius of the planet
        :type radius: float
        '''
        self._pos = initial_position
        self.mass = mass
        self.radius = radius
        self._velocity = initial_velocity

    @property
    def pos(self):
        return self._pos()

    @pos.setter
    def pos(self, updated_pos):
        self._pos = updated_pos

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, updated_velocity):
        return self._velocity = updated_velocity