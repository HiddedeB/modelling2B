import numpy as np
import matplotlib.pyplot as plt
from data import PlanetaryDataHandler
from scipy.integrate import solve_ivp
from scipy.integrate import quad
# Imporing all the important planetary data
pdh = PlanetaryDataHandler()
sun = pdh.sun
j=pdh.jupiter
s = pdh.saturn
u = pdh.uranus
n = pdh.neptune

# Important vectors/consants
alpha_vector = np.array([j.smaxis, s.smaxis, u.smaxis, n.smaxis])
masses_vector = np.array([j.mass, s.mass, u.mass, n.mass])
masses_dividor = 1/(sun.mass+masses_vector)
J_2_vector = np.array([14736, 16298, 3343, 3411])
J_4_vector = np.array([-587, -915, -29, -35])


# n_vector, needed for calculating A_jj, A_jk etc later on
n_vector = 2*np.pi/(alpha_vector**(3/2))

# Calculating the right alpha elements, alpha and alpha bar for the matrix elements

def alpha_calculator(semi_major):
    alpha_matrix = np.zeros(4)
    alpha_bar_times_alpha_matrix = np.zeros(4)
    for i in range(len(semi_major)):
        current_alpha = semi_major[i] / np.delete(semi_major, i)
        current_alpha = np.insert(current_alpha, obj=i, values=0)
        alpha_matrix = np.vstack((alpha_matrix, current_alpha))

        alpha_bar_times_alpha_matrix = np.vstack((alpha_bar_times_alpha_matrix,
                                                  np.concatenate((np.repeat(1, i), current_alpha[i:])) * current_alpha))

    alpha_matrix = alpha_matrix[1:]
    alpha_bar_times_alpha_matrix = alpha_bar_times_alpha_matrix[1:]

    return alpha_matrix, alpha_bar_times_alpha_matrix

alpha_matrix,alpha_times_bar_matrix = alpha_calculator(alpha_vector)
a,b = alpha_calculator(alpha_vector)
# Adjusting the alpha_matrix in a slightly better form for doing calculations later on
alpha_times_bar_matrix[0,:]= alpha_times_bar_matrix[0,:] * n_vector[0]
alpha_times_bar_matrix[1,:] = alpha_times_bar_matrix[1,:]  * n_vector[1]
alpha_times_bar_matrix[2,:] = alpha_times_bar_matrix[2,:]  * n_vector[2]
alpha_times_bar_matrix[3,:] = alpha_times_bar_matrix[3,:]  * n_vector[3]

# The b_3/2 elements
def integrand1(phi, alpha):
    return (1 / np.pi) * np.cos(phi) / (1 - 2 * alpha *
                                        np.cos(phi) + alpha ** 2) ** (3/2) # angle phi

def integrand2(phi,alpha):
    return (1 / np.pi) * np.cos(2 * phi) / (1 - 2 * alpha *
                                            np.cos(2 * phi) + alpha ** 2) ** (3 / 2)  # angle 2*phi

def beta1(alpha):
    return quad(integrand1, 0, 2*np.pi, args=(alpha))[0]

def beta2(alpha):
    return quad(integrand2, 0, 2*np.pi, args=(alpha))[0]

vec_beta1 = np.vectorize(beta1)
vec_beta2 = np.vectorize(beta2)


A_matrix = np.zeros((4,4))

# A calculations unequal
beta_jupiter_2 = vec_beta2(alpha_matrix[0,1:])
A_matrix[0,1:] = -(1/4) * masses_dividor[0]* masses_vector[1:] *alpha_times_bar_matrix[0,1:] * beta_jupiter_2

beta_saturn_21 = vec_beta2(alpha_matrix[1,:1])
beta_saturn_22 = vec_beta2(alpha_matrix[1,2:])

A_matrix[1,:1] = -(1/4) * masses_dividor[1]*masses_vector[0]  * alpha_times_bar_matrix[1,:1] * beta_saturn_21
A_matrix[1,2:] = -(1/4) * masses_dividor[1] * masses_vector[2:]*alpha_times_bar_matrix[1,2:] * beta_saturn_22

beta_uranus_21 = vec_beta2(alpha_matrix[2,:2])
beta_uranus_22 = vec_beta2(alpha_matrix[2,3:])

A_matrix[2,:2] = -(1/4) * masses_dividor[2]*masses_vector[:2]  * alpha_times_bar_matrix[2,:2] * beta_uranus_21
A_matrix[2,3:] = -(1/4) * masses_dividor[2] * masses_vector[3:]*alpha_times_bar_matrix[2,3:] * beta_uranus_22

beta_neptune_2 = vec_beta2(alpha_matrix[3,:3])
A_matrix[3,:3] = -(1/4) * masses_dividor[3]* masses_vector[:3] *alpha_times_bar_matrix[3,:3] * beta_neptune_2

#Calculating B-elements unequal
B_matrix = np.zeros((4,4))
beta_jupiter_1 = vec_beta1(alpha_matrix[0,1:])
B_matrix[0,1:] = (1/4) * masses_dividor[0]* masses_vector[1:] *alpha_times_bar_matrix[0,1:] * beta_jupiter_1

beta_saturn_11 = vec_beta1(alpha_matrix[1,:1])
beta_saturn_12 = vec_beta1(alpha_matrix[1,2:])

B_matrix[1,:1] = (1/4) * masses_dividor[1]*masses_vector[0] * alpha_times_bar_matrix[1,:1] * beta_saturn_11
B_matrix[1,2:] = (1/4) * masses_dividor[1] * masses_vector[2:]*alpha_times_bar_matrix[1,2:] * beta_saturn_12

beta_uranus_11 = vec_beta1(alpha_matrix[2,:2])
beta_uranus_12 = vec_beta1(alpha_matrix[2,3:])

B_matrix[2,:2] = (1/4) * masses_dividor[2]*masses_vector[:2]*alpha_times_bar_matrix[2,:2]*beta_uranus_11
B_matrix[2,3:] = (1/4) * masses_dividor[2]*masses_vector[3:]*alpha_times_bar_matrix[2,3:]*beta_uranus_12

beta_neptune_1 = vec_beta1(alpha_matrix[3,:3])
B_matrix[3,:3] = (1/4) * masses_dividor[3]* masses_vector[:3] * alpha_times_bar_matrix[3,:3] * beta_neptune_1


# Calculating A-elements and B-elements equal
handige_array = np.array([sum(B_matrix[0,:]), sum(B_matrix[1,:]), sum(B_matrix[2,:]), sum(B_matrix[3,:])])
A_vector_equal = n_vector*((3/2)*J_2_vector*(sun.radius/alpha_vector)**2 -(9/8)*J_2_vector**2 * (sun.radius/alpha_vector)**4 +(15/4)*J_4_vector * (sun.radius/alpha_vector)**4)+handige_array
B_vector_equal = -n_vector*((3/2)*J_2_vector*(sun.radius/alpha_vector)**2 -(27/8)*J_2_vector**2 * (sun.radius/alpha_vector)**4 +(15/4)*J_4_vector * (sun.radius/alpha_vector)**4)+handige_array

# Total A and B matrices for solving the ODES :)
A_matrix = A_matrix + np.diag(A_vector_equal)
B_matrix = B_matrix + np.diag(B_vector_equal)

# System of equations maker
#def sys_of_eq_maker(t,y, A_matrix,B_matrix):
#    h1,h2,h3,h4,k1,k2,k3,k4,p1,p2,p3,p4,q1,q2,q3,q4 = y
#    elements1 = np.array([[k1,k2,k3,k4]
#                         [-h1,-h2,-h3,-h4]])
#    elements2 = np.array([q1,q2,q3,q4],
#                         [-p1,-p2,-p3,-p4])
#
#    resulting_vectors1 = A_matrix.dot(elements1)
#    resulting_vectors2 = B_matrix.dot(elements2)
#
#    return






