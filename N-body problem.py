import numpy as np
from data import PlanetaryDataHandler

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

    # Making matrix of these elements:
    zero_matrix = np.zeros((4,4))
    zero_matrix[0,1:] = alpha_jupiter
    zero_matrix[1][0] = alpha_saturnus_internal
    zero_matrix[1,2:] = alpha_saturnus_exeternal
    zero_matrix[2,0:2] = alpha_uranus_internal
    zero_matrix[2][3] = alpha_uranus_exeternal
    zero_matrix[3,0:3] = alpha_neptunus_total
    alpha_matrix = zero_matrix

    zero_matrix = np.zeros((4,4))
    zero_matrix[0,1:] = alpha_bar_jupiter
    zero_matrix[1][0] = alpha_bar_saturn[:1]
    zero_matrix[1,2:] = alpha_bar_saturn[1:]
    zero_matrix[2,0:2] = alpha_bar_uranus[:2]
    zero_matrix[2][3] = alpha_bar_uranus[2:]
    zero_matrix[3,0:3] = alpha_bar_neptunus
    alpha_bar_matrix = zero_matrix

    return alpha_matrix, alpha_bar_matrix

alpha_matrix,alpha_bar_matrix = alpha_calculator(alpha_vector)

