import numpy as np
import scipy.integrate as solve_ivp
from PlanetClass import planet
import astropy.constants as const

#important constants

massa_zon = 1.989 * 10**(30) #kg, massa zon
massa_aarde = 5.972 * 10**(26) #kg, massa aarde
massa_mercurius = 3.285 * 10**(23) #kg, massa mercurius
massa_venus = 4.867 * 10**(24) #kg, massa venus
massa_mars = 6.39 * 10**(23) #kg, massa mars
massa_jupiter = 1.898 * 10**(27) #kg, massa jupiter
massa_saturnus = 5.682 * 10**(26) #kg, massa saturnus
massa_uranus = 8.681 * 10**(25) #kg, massa Uranus
massa_neptunus = 1.024 * 10**(26) #kg, massa neptunus

# Initialize planet.
initial_vector = np.array([1,1,1])
initial_vel = np.array([1,0,0])
earth = planet(mass=const.M_earth.value, initial_position=initial_vector, radius=const.R_earth.value, initial_velocity=initial_vel)
