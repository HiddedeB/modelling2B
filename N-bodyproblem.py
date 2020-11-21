import numpy as np
import matplotlib.pyplot as plt
from data import PlanetaryDataHandler
from scipy.integrate import solve_ivp
from scipy.integrate import quad
# Imporing all the important planetary data
pdh = PlanetaryDataHandler()
j=pdh.jupiter
s = pdh.saturn
u = pdh.uranus
n = pdh.neptune

# Semi-major axis of the 4 giants
alpha_vector = np.array([j.smaxis, s.smaxis, u.smaxis, n.smaxis])

# Calculating the right alpha elements, alpha and alpha bar for the matrix elements

def alpha_calculator(semi_major):
    # Calculating alpha for jupiter
    alpha_jupiter = semi_major[0]/semi_major[1:]

    # Calculating alpha for saturn
    semi_major_s = np.concatenate([semi_major[:1],semi_major[2:]])
    alpha_saturnus_internal = semi_major[1]/(semi_major_s[0:1]) #internal elements
    alpha_saturnus_exeternal = semi_major[1]/(semi_major_s[1:]) #external elements

    # Calculating alpha for uranus
    semi_major_u = np.concatenate([semi_major[:2],semi_major[3:]])
    alpha_uranus_internal = semi_major[2]/(semi_major_u[0:2]) #internal elements
    alpha_uranus_exeternal = semi_major[2]/(semi_major_u[2:]) #external elements

    # Calculating alpha for neptune
    semi_major_n = semi_major[:3]
    alpha_neptunus_total = semi_major[3]/(semi_major_n[:4]) #internal elements

    # Calculating alpha_bar for jupiter
    alpha_bar_jupiter = np.ones(3)

    # Calculating alpha bar elements for saturn
    alpha_bar_saturn = np.concatenate([np.ones(1),alpha_saturnus_exeternal])

    # Calculating_alpha_bar elements for uranus
    alpha_bar_uranus = np.concatenate([np.ones(2), alpha_uranus_exeternal])

    # Calculating_alpha_bar elements for neptunus
    alpha_bar_neptunus = alpha_neptunus_total

    # Making matrix of these elements for alpha:
    zero_matrix = np.zeros((4,4))
    zero_matrix[0,1:] = alpha_jupiter
    zero_matrix[1][0] = alpha_saturnus_internal
    zero_matrix[1,2:] = alpha_saturnus_exeternal
    zero_matrix[2,0:2] = alpha_uranus_internal
    zero_matrix[2][3] = alpha_uranus_exeternal
    zero_matrix[3,0:3] = alpha_neptunus_total
    alpha_matrix = zero_matrix

    # Making matrix of these elements for alpha_times_bar:
    zero_matrix = np.zeros((4,4))
    zero_matrix[0,1:] = alpha_jupiter * alpha_bar_jupiter
    zero_matrix[1][0] = alpha_saturnus_internal * alpha_bar_saturn[:1]
    zero_matrix[1,2:] = alpha_saturnus_exeternal * alpha_bar_saturn[1:]
    zero_matrix[2,0:2] = alpha_uranus_internal * alpha_bar_uranus[:2]
    zero_matrix[2][3] = alpha_uranus_exeternal*alpha_bar_uranus[2:]
    zero_matrix[3,0:3] = alpha_neptunus_total * alpha_bar_neptunus
    alpha_times_bar_matrix = zero_matrix

    return alpha_matrix, alpha_times_bar_matrix

alpha_matrix,alpha_times_bar_matrix = alpha_calculator(alpha_vector)

# n_vector, needed for calculating A_jj, A_jk etc later on
n_vector = 2*np.pi/(alpha_vector**(3/2))

# Adjusting the alpha_matrix in a slightly better form for doing calculations later on
alpha_times_bar_matrix[0,:]=alpha_times_bar_matrix[0,:] * n_vector[0]
alpha_times_bar_matrix[1,:] = alpha_times_bar_matrix[1,:]  * n_vector[1]
alpha_times_bar_matrix[2,:] = alpha_times_bar_matrix[2,:]  * n_vector[2]
alpha_times_bar_matrix[3,:] = alpha_times_bar_matrix[3,:]  * n_vector[3]

# The b_3/2 elements
def integrand1(phi, alpha):
    return (1 / np.pi) * np.cos(phi) / (1 - 2 * alpha_matrix[0,:] *
                                        np.cos(phi) + alpha ** 2) ** (3/2) # angle phi

def integrand2(phi,alpha):
    return (1 / np.pi) * np.cos(2 * phi) / (1 - 2 * alpha_matrix[0,:] *
                                            np.cos(2 * phi) + alpha_matrix[1,:] ** 2) ** (3 / 2)  # angle 2*phi

def beta1(alpha):
    return quad(integrand1, 0, 2*np.pi, args=(alpha))[0]

def beta2(alpha):
    return quad(integrand2, 0, 2*np.pi, args=(alpha))[0]

vec_beta1 = np.vectorize(beta1)
vec_beta2 = np.vectorize(beta2)

# Making the Beta Matrix

