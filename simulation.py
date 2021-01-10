import numpy as np
from data import create_pdh
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from numba import njit
import Orbitdrawer as Od
from matplotlib import animation
import Orbitdrawer as Od
import math


class simulation():
    '''Class to compute simulation of the orbital elements'''

    def __init__(self, file_name, etnos_file='', **kwargs):
        '''NOTE: Initial function that sets up all the different attributes of our class.
        :param file_name: File_name directing to data file for planethandler.
        :type file_name: str
        :param etnos_file: File_name for the ETNO's.
        :type etnos_file: str
        :param **kuiperbelt: If True the function creates a kuiperbelt. For this all the following parameters should be
        given.
        :type **kuiperbelt: bool
        :param **hom_mode: If True built a homogeneous kuiperbelt in are, if False divide evenly over all orbits.
        :type **hom_mode: bool
        :param **total_mass: Parameter specifying the total mass of the kuiperbelt.
        :type **total_mass: int
        :param **r_res: How many orbits the kuiperbelt contains.
        :type r_res: int
        :param **range_min: Minimum distance of kuiperbelt.
        :type **range_min: float
        :param **range_max: Maximum distance of kuiperbelt.
        :type **range_max: float
        '''


        if not etnos_file:
            pdh = create_pdh(file_name)
        else:
            pdh = create_pdh(file_name, etnos_file)

        # G = 6.67430 * 10**(-11)
        G = 39.47692641  # units van au^3/y^2 M_sun
        self.j = pdh.jupiter
        self.s = pdh.saturn
        self.u = pdh.uranus
        self.n = pdh.neptune
        self.sun = pdh.sun
        self.smaxis_vector = np.array([self.j['smaxis'], self.s['smaxis'], self.u['smaxis'], self.n['smaxis']])
        self.mass_vector = np.array([self.j['mass'], self.s['mass'], self.u['mass'], self.n['mass']])

        if 'planet9' in kwargs:
            if kwargs['planet9']:
                d = kwargs['planet9'] # for shorter notation below
                pdh.createnewplanet(d['mass'], d['radius'], d['loanode'], d['eccentricity'], d['smaxis'],
                    d['argperiapsis'], d['orbital inclination'], d['mean longitude'])
                # , smaxis, argperiapsis, orbital_inclination, mean_longitude
                self.planet9 = pdh.planet9
                self.smaxis_vector = np.append(self.smaxis_vector, self.planet9['smaxis'])
                self.mass_vector = np.append(self.mass_vector, self.planet9['mass'])



        self.J_2_vector = np.array([14736, 16298, 3343, 3411]) * 10 ** (-6)
        self.J_2 = 2.198 * 10 ** (-7)  # J_2 of the sun
        self.J_4 = -4.805 * 10 ** (-9)  # J_4 of the sun
        self.J_4_vector = np.array([-587, -915, -29, -35]) * 10 ** (-6)
        self.n_vector = np.sqrt(G * self.sun['mass'] / (self.smaxis_vector**3))
        self.planet_number = len(self.smaxis_vector)


        if 'kuiperbelt' in kwargs:
            if kwargs['kuiperbelt']:
                if 'hom_mode' in kwargs:
                    hom_mode = kwargs['hom_mode']
                else:
                    hom_mode = False
                pdh.add_kuiperbelt(kwargs['total_mass'], kwargs['r_res'], kwargs['range_min'], kwargs['range_max'],
                                   hom_mode)

        if 'etnos' in kwargs:
            if kwargs['etnos']:
                pdh.add_etnos()

        if 'kuiperbelt' in kwargs or 'etnos' in kwargs:
            self.asteroid_smaxis = pdh.asteroid_attributes['smaxis']
            self.asteroid_number = len(self.asteroid_smaxis)
            self.asteroid_mass = pdh.asteroid_attributes['mass']
            self.asteroid_argperiapsis = pdh.asteroid_attributes['argperiapsis']
            self.asteroid_big_omega = pdh.asteroid_attributes['loanode']
            self.asteroid_inclination = pdh.asteroid_attributes['orbital inclination']
            self.asteroid_eccentricity = pdh.asteroid_attributes['eccentricity']
            self.free_n_vector = np.sqrt(G * self.sun['mass'] / (self.asteroid_smaxis**3))
            # self.free_n_vector = 2 * np.pi / (self.asteroid_smaxis ** (3 / 2))

    def alpha_matrix(self, kuiperbelt=False):
        '''NOTE: Function to compute the alpha matrix and the alpha bar and alpha product matrix.
        :param kuiperbelt: If False, no kuiperbelt simulation; If True built the matrices for kuiperbelt simulation.
        :type kuiperbelt: True
        '''
        alpha_matrix = np.zeros(self.planet_number,dtype=np.float64)
        alpha_bar_times_alpha_matrix = np.zeros(self.planet_number,dtype=np.float64)
        for i in range(self.planet_number):
            current_alpha = np.fmin(self.smaxis_vector[i] / np.delete(self.smaxis_vector, i),
                                    np.delete(self.smaxis_vector, i) / self.smaxis_vector[i])
            current_alpha = np.insert(current_alpha, obj=i, values=0)
            alpha_matrix = np.vstack((alpha_matrix, current_alpha))

            alpha_bar_times_alpha_matrix = np.vstack((alpha_bar_times_alpha_matrix,
                                                      np.concatenate((np.repeat(1, i), current_alpha[i:]))
                                                      * current_alpha))

        alpha_matrix = alpha_matrix[1:]
        alpha_bar_times_alpha_matrix = alpha_bar_times_alpha_matrix[1:]

        # If the kuiperbelt is specified it will execute this if statement.
        if kuiperbelt:

            free_alpha = np.zeros((self.asteroid_number, self.planet_number),dtype=np.float64)
            free_alpha_bar_times_alpha = np.zeros((self.asteroid_number, self.planet_number),dtype=np.float64)

            free_alpha = free_alpha + np.fmin(self.asteroid_smaxis[:, np.newaxis] / self.smaxis_vector,
                                              self.smaxis_vector / self.asteroid_smaxis[:, np.newaxis])

            free_alpha_bar_times_alpha = free_alpha_bar_times_alpha + np.fmin(self.asteroid_smaxis[:, np.newaxis] /
                                                                              self.smaxis_vector,
                                                                              self.smaxis_vector /
                                                                              self.asteroid_smaxis[:, np.newaxis])

            free_smaxis_array = np.array([self.asteroid_smaxis, ] * self.planet_number, dtype=np.float64).transpose()
            smaxis_array = np.array([self.smaxis_vector, ] * self.asteroid_number, dtype=np.float64)

            free_alpha_bar_times_alpha[free_smaxis_array < smaxis_array] = 1

            return np.array([alpha_matrix, alpha_bar_times_alpha_matrix, free_alpha, free_alpha_bar_times_alpha], dtype=np.ndarray)

        else:
            return np.array([alpha_matrix, alpha_bar_times_alpha_matrix], dtype=np.ndarray)

    @staticmethod
    def beta_values(alpha_matrix, kuiperbelt=False, **kwargs):
        '''NOTE: function to compute the b_{2/3} values needed for the A and B matrices.
        :param alpha_matrix: alpha matrix, consisting of all the different alphas, specified by the equations 7.128 and
        7.129 in Solar System dynamics.
        :type alpha_matrix: np.ndarray
        :param kuiperbelt: If True run kuiperbelt simulation, if False do not.
        :type kuiperbelt: bool
        :param **free_alpha: The free particle alpha matrix
        :type **free_alpha: np.ndarray
        '''

        def beta1(alpha):
            def g(phi, alpha):
                # Angle phi
                if alpha != 0:
                    return (1 / np.pi) * np.cos(phi) / (1 - 2 * alpha * np.cos(phi) + alpha**2)**(3/2)
                else:
                    return 0

            return quad(g, 0, 2 * np.pi, args=(alpha))[0]

        def beta2(alpha):
            def f(phi, alpha):
                # Angle 2*phi
                if alpha != 0:
                    return (1 / np.pi) * np.cos(2 * phi) / (1 - 2 * alpha * np.cos(phi) + alpha**2)**(3/2)
                else:
                    return 0

            return quad(f, 0, 2 * np.pi, args=(alpha))[0]

        vec_beta1 = np.vectorize(beta1)
        vec_beta2 = np.vectorize(beta2)

        if kuiperbelt:
            if 'free_alpha' not in kwargs:
                raise ValueError('Trying to run kuiperbelt simulation without free_alpha matrix')
            return np.array([vec_beta1(alpha_matrix), vec_beta2(alpha_matrix), vec_beta1(kwargs['free_alpha']),
                             vec_beta2(kwargs['free_alpha'])], dtype=np.ndarray)  # Heb de laatste entry van de array veranderd van vec_beta2(kwargs['free_alpha_bar']) naar vec_beta2(kwargs['free_alpha'])

        else:
            return np.array([vec_beta1(alpha_matrix), vec_beta2(alpha_matrix)], dtype=np.ndarray)

    def a_b_matrices(self, alpha_bar_times_alpha_matrix, beta, kuiperbelt=False, **kwargs):
        '''NOTE: Function to compute the A and B matrices needed to build the differential equations.
        :param alpha_bar_times_alpha_matrix: Outputted as the second argument of function alpha_matrix. This is the
        product of the regular alpha_matrix and the alpha_bar_matrix as specified by the equations 7.128 and
        7.129 in Solar System dynamics.
        :param beta: The array of two beta matrices as generated by the function beta_values. If we have a free particle
        simulation then beta contains 4 matrices, with the first two containing the regular betas, whilst the third and
        fourth index contain the beta matrices for the free particle simulation.
        :type alpha_bar_times_alpha_matrix: np.ndarray.
        :type beta: np.ndarray.
        :param kuiperbelt: If True run kuiperbelt simulation, if False do not run kuiperbelt simulation.
        :type kuiperbelt: bool
        :param **free_alpha_bar_matrix: Matrix of the product of alpha and alpha bar for the free particles.
        :type **free_alpha_bar_matrix: np.ndarray
        '''

        # Off-diagonal components
        sub_matrix = 1 / 4 * self.mass_vector / (self.sun['mass'] + self.mass_vector[:, np.newaxis]) \
                     * alpha_bar_times_alpha_matrix * self.n_vector[:, np.newaxis]

        a = sub_matrix
        b = sub_matrix * beta[0]

        a_d = self.n_vector * ((self.sun['radius'] / self.smaxis_vector)**2 * (3/2 * self.J_2 -
                               (self.sun['radius'] / self.smaxis_vector)**2 * (15/4 * self.J_4 +
                                                                               9/8 * self.J_2**2)))

        a_new = a * beta[0]
        a_d = a_d + a_new.sum(axis=1)

        b_d = -self.n_vector * ((self.sun['radius'] / self.smaxis_vector)**2 * (3/2 * self.J_2 -
                                (self.sun['radius'] / self.smaxis_vector)**2 * (15/4 * self.J_4 +
                                                                                27/8 * self.J_2**2)))
        b_d = b_d - b.sum(axis=1)

        a_matrix = np.diag(a_d) - a * beta[1]
        b_matrix = np.diag(b_d) + b
        if kuiperbelt:
            if 'free_alpha_bar_matrix' not in kwargs:
                raise ValueError('Trying to run kuiperbelt simulation without the free_alpha_bar_matrix.')

            sub_matrix = 1 / 4 * self.mass_vector / self.sun['mass'] * kwargs['free_alpha_bar_matrix'] * \
                         self.free_n_vector[:, np.newaxis]

            a = sub_matrix
            b = sub_matrix * beta[2]

            a_d = self.free_n_vector * ((self.sun['radius'] / self.asteroid_smaxis)**2 * (3/2 * self.J_2 -
                                        (self.sun['radius'] / self.asteroid_smaxis)**2 * (15/4 * self.J_4 +
                                                                                          9/8 * self.J_2**2)))


            a_new = a * beta[2]
            a_d = a_d + a_new.sum(axis=1)

            b_d = -self.free_n_vector * ((self.sun['radius'] / self.asteroid_smaxis)**2 * (3/2 * self.J_2 -
                                         (self.sun['radius'] / self.asteroid_smaxis)**2 * (15/4 * self.J_4 +
                                                                                           27/8 * self.J_2**2)))

            b_d = b_d - b.sum(axis=1)
            free_a_matrix = np.zeros((self.asteroid_number, self.planet_number+1), dtype=np.float64)
            free_a_matrix[:, 0] = a_d
            free_a_matrix[:, 1:] = a * beta[3]
            free_b_matrix = np.zeros((self.asteroid_number, self.planet_number + 1), dtype=np.float64)
            free_b_matrix[:, 0] = b_d
            free_b_matrix[:, 1:] = b

            return np.array([a_matrix, b_matrix, free_a_matrix, free_b_matrix], dtype=np.ndarray)

        else:

            return np.array([np.array([np.array(i,dtype=np.float64) for i in a_matrix]),
                             np.array([np.array(i,dtype=np.float64) for i in b_matrix]),
                             np.array([],dtype=np.ndarray)], dtype=np.ndarray)[0:2]

    @staticmethod
    def initial_condition_builder(e, var_omega, I, big_omega, radians):
        '''NOTE: Initial condition maker, if the initial conditions are not in h, k, p and q form, then this function
        can be called to transfer them.
        :param e: The eccentricities of the orbits.
        :type e: np.ndarray
        :param var_omega: The difference between big_omega and omega for every orbit.
        :type var_omega: np.ndarray
        :param I: The angles of the orbit w.r.t. the reference frame.
        :type I: np.ndarray
        :param big_omega: The angles between the ascending node and the reference frame x-hat direction for every orbit.
        :type big_omega: np.ndarray
        :param radians: If True values are in radians, if False they are in degrees.
        :type radians: bool
        '''
        if not radians:
            var_omega = var_omega / 180 * np.pi
            I = I / 180 * np.pi
            big_omega = big_omega / 180 * np.pi

        h = e * np.sin(var_omega)
        k = e * np.cos(var_omega)
        p = I * np.sin(big_omega)
        q = I * np.cos(big_omega)

        return np.concatenate((h, k, p, q))

    @staticmethod
    def variable_transformations(h, k, p, q):
        '''NOTE: Function to transfer the new made coordinates h, k, p, q to e, I, var_omega and big_omega'''
        return np.array([np.sqrt(h ** 2 + k ** 2), np.sqrt(p ** 2 + q ** 2), np.arcsin(h / np.sqrt(h ** 2 + k ** 2)),
                         np.arcsin(p / np.sqrt(p ** 2 + q ** 2))])

    # There are two functions, to get numba to work this was needed.

    @staticmethod
    @njit
    def orbital_calculator_kuiperbelt(t, vector, *args):
        '''NOTE: function to run the solver with, vector contains h, k, p, q in that order. Furthermore, the values are
        taken to go from closest to the sun to most outward.
        :param *args: Array of matrices, because scipy is stupid.
        :type *args: np.ndarray
        '''
        a_matrix = args[0]
        b_matrix = args[1]

        # Setting up the differential equations according to 7.25 and 7.26 from Solar System Dynamics. Since the a and
        # b matrices are both square and contain as many entries as planets, we use this for splitting our data vector.
        planet_number = a_matrix.shape[0]
        h_vector = vector[:planet_number]
        k_vector = vector[planet_number:2 * planet_number]
        p_vector = vector[2 * planet_number: 3 * planet_number]
        q_vector = vector[3 * planet_number: 4 * planet_number]

        # Differential equations.
        d_h_matrix = a_matrix * k_vector
        d_h_vec = d_h_matrix.sum(axis=1)
        d_k_matrix = a_matrix * h_vector
        d_k_vec = -d_k_matrix.sum(axis=1)
        d_p_matrix = b_matrix * q_vector
        d_p_vec = d_p_matrix.sum(axis=1)
        d_q_matrix = b_matrix * p_vector
        d_q_vec = -d_q_matrix.sum(axis=1)

        free_a_matrix = args[2]
        free_b_matrix = args[3]
        vector = vector[4 * planet_number:]
        amount_of_particles = int(len(vector)/4)
        free_h_vector = vector[:amount_of_particles]
        free_k_vector = vector[amount_of_particles:2 * amount_of_particles]
        free_p_vector = vector[2 * amount_of_particles: 3 * amount_of_particles]
        free_q_vector = vector[3 * amount_of_particles: 4 * amount_of_particles]

        # Redefine to be useful
        f_h = np.zeros((amount_of_particles, planet_number+1))
        f_k = np.zeros((amount_of_particles, planet_number+1))
        f_p = np.zeros((amount_of_particles, planet_number+1))
        f_q = np.zeros((amount_of_particles, planet_number+1))

        f_h[:, 0] = free_h_vector
        f_k[:, 0] = free_k_vector
        f_p[:, 0] = free_p_vector
        f_q[:, 0] = free_q_vector

        f_h[:, 1:] = h_vector.repeat(amount_of_particles).reshape(-1, amount_of_particles).T
        f_k[:, 1:] = k_vector.repeat(amount_of_particles).reshape(-1, amount_of_particles).T
        f_p[:, 1:] = p_vector.repeat(amount_of_particles).reshape(-1, amount_of_particles).T
        f_q[:, 1:] = q_vector.repeat(amount_of_particles).reshape(-1, amount_of_particles).T

        # Differential equations.
        d_h_free_matrix = free_a_matrix * f_k
        d_h_free_vec = d_h_free_matrix.sum(axis=1)
        d_k_free_matrix = free_a_matrix * f_h
        d_k_free_vec = -d_k_free_matrix.sum(axis=1)
        d_p_free_matrix = free_b_matrix * f_q
        d_p_free_vec = d_p_free_matrix.sum(axis=1)
        d_q_free_matrix = free_b_matrix * f_p
        d_q_free_vec = -d_q_free_matrix.sum(axis=1)

        return np.concatenate((d_h_vec, d_k_vec, d_p_vec, d_q_vec, d_h_free_vec, d_k_free_vec, d_p_free_vec,
                               d_q_free_vec))

    @staticmethod
    @njit
    def orbital_calculator_no_kuiperbelt(t, vector, *args):
        '''
        NOTE: function to run the solver with, vector contains h, k, p, q in that order. Furthermore, the values are
        taken to go from closest to the sun to most outward.
        :param *args: Array of matrices, because scipy is stupid.
        :type *args: np.ndarray
        '''
        a_matrix = args[0]
        # print(np.array_repr(a_matrix))
        b_matrix = args[1]

        # Setting up the differential equations according to 7.25 and 7.26 from Solar System Dynamics. Since the a and
        # b matrices are both square and contain as many entries as planets, we use this for splitting our data vector.
        planet_number = a_matrix.shape[0]
        h_vector = vector[:planet_number]
        k_vector = vector[planet_number:2 * planet_number]
        p_vector = vector[2 * planet_number: 3 * planet_number]
        q_vector = vector[3 * planet_number: 4 * planet_number]

        # Differential equations.
        d_h_matrix = a_matrix * k_vector
        d_h_vec = d_h_matrix.sum(axis=1)
        d_k_matrix = a_matrix * h_vector
        d_k_vec = -d_k_matrix.sum(axis=1)
        d_p_matrix = b_matrix * q_vector
        d_p_vec = d_p_matrix.sum(axis=1)
        d_q_matrix = b_matrix * p_vector
        d_q_vec = -d_q_matrix.sum(axis=1)

        return np.concatenate((d_h_vec, d_k_vec, d_p_vec, d_q_vec))

    @staticmethod
    def Euler_backward_calculator(dt, vector, *args):
        """NOTE: Function to compute the Euler backward implementation of our problem. Futhermore,
        Euler backward has y_n+1 = y_n + dt * f(t_n+1, y_n+1) = y_n + dt * A y_n+1, when considering matrices. Aside 
        from this, note that for the planets we have h_n+1 = h_n + dt * A * k_n+1, additionally we have k_n+1 = k_n 
        - dt * A h_n+1 => h_n+1 = h_n + dt * A * (k_n - dt * A * h_n+1) => 
        h_n+1 = (I_[number_of_planets] + dt^2 A^2)^-1 (h_n + dt * A * k_n). Aside from this, consider for the free 
        particle, every particle is described by h_n+1 = h_n + A_free[0] k_n+1 (asteroid) + A_free[1:] h_n+1(planets)
        => h_n+1 = (1 - A_free[0])^-1*(A_free[1:]*(I_[number_of_planets] + dt^2 A^2)^-1 (h_n + dt * A * k_n)) """
        # Matrices
        a_matrix = args[0]
        b_matrix = args[1]
        free_a_matrix = args[2]
        free_b_matrix = args[3]

        # Current data
        planet_number = a_matrix.shape[0]
        identity = np.identity(planet_number)
        h_vector = vector[:planet_number]
        k_vector = vector[planet_number:2 * planet_number]
        p_vector = vector[2 * planet_number: 3 * planet_number]
        q_vector = vector[3 * planet_number: 4 * planet_number]

        amount_of_particles = int(len(vector) / 4) - planet_number
        free_identity = np.ones(amount_of_particles)

        first_a_column = free_a_matrix[:, 0]
        a_prime = free_a_matrix[:, 1:]
        first_b_column = free_b_matrix[:, 0]
        b_prime = free_b_matrix[:, 1:]

        free_h_vector = vector[:amount_of_particles]
        free_k_vector = vector[amount_of_particles:2 * amount_of_particles]
        free_p_vector = vector[2 * amount_of_particles: 3 * amount_of_particles]
        free_q_vector = vector[3 * amount_of_particles: 4 * amount_of_particles]

        # Build the n+1 terms
        d_h = np.dot(np.linalg.inv(identity + dt**2 * np.dot(a_matrix, a_matrix)), (h_vector + dt * np.dot(a_matrix, k_vector)))
        d_k = np.dot(np.linalg.inv(identity + dt**2 * np.dot(a_matrix, a_matrix)), (k_vector - dt * np.dot(a_matrix, h_vector)))
        d_p = np.dot(np.linalg.inv(identity + dt**2 * np.dot(b_matrix, b_matrix)), (p_vector + dt * np.dot(b_matrix, q_vector)))
        d_q = np.dot(np.linalg.inv(identity + dt**2 * np.dot(b_matrix, b_matrix)), (q_vector - dt * np.dot(b_matrix, p_vector)))

        # We do not need to invert with the matrices, since these operations are 1-D
        d_h_free = 1 / (free_identity + dt**2 * first_a_column**2) * (free_h_vector +
                                                                      dt * first_a_column * free_k_vector -
                                                                      dt**2 * first_a_column * np.dot(a_prime, d_h) +
                                                                      dt * np.dot(a_prime, d_k))
        d_k_free = 1 / (free_identity + dt**2 * first_a_column**2) * (free_k_vector -
                                                                      dt * first_a_column * free_h_vector -
                                                                      dt**2 * first_a_column * np.dot(a_prime, d_k) -
                                                                      dt * np.dot(a_prime, d_h))
        d_p_free = 1 / (free_identity + dt**2 * first_b_column**2) * (free_p_vector +
                                                                      dt * first_b_column * free_q_vector -
                                                                      dt**2 * first_b_column * np.dot(b_prime, d_p) +
                                                                      dt * np.dot(b_prime, d_q))
        d_q_free = 1 / (free_identity + dt**2 * first_b_column**2) * (free_q_vector -
                                                                      dt * first_a_column * free_p_vector -
                                                                      dt**2 * first_b_column * np.dot(b_prime, d_q) -
                                                                      dt * np.dot(b_prime, d_p))

        return np.concatenate((d_h, d_k, d_p, d_q, d_h_free, d_k_free, d_p_free, d_q_free))

    def Euler_backward_method(self, time_scale, initial_conditions, max_step, kuiperbelt, *args):

        time = np.array([max_step], dtype=np.float64)

        # preliminary calculation first.

        if kuiperbelt:
            solutions = np.zeros((4*(self.planet_number+self.asteroid_number), 1))

        else:
            solutions = np.zeros((4*self.planet_number, 1))

        new_solution = self.Euler_backward_calculator(max_step, initial_conditions, *args)
        solutions = np.hstack((solutions, new_solution[:, np.newaxis]))

        for time_step in np.arange(time_scale[0]+max_step, time_scale[1], max_step):
            new_solution = self.Euler_backward_calculator(max_step, solutions[:, -1], *args)
            solutions = np.hstack((solutions, new_solution[:, np.newaxis]))
            time = np.append(time, time_step)

        solutions = solutions[:, 1:]

        return solutions, time

    def order_of_error(self, time_scale, form_of_ic, initial_conditions, max_step, method, relative_tolerance,
                       absolute_tolerance, kuiperbelt=False):
        '''NOTE: Function to compute the order of error, we still need to figure out error in which part of the method,
        for now lets assume the error is in the step size, not fully done yet, actively being worked on.'''

        multipliers = np.array([1, 2, 4], dtype=np.float64)*max_step
        size = np.array([], dtype=np.float64)
        make_array = True
        for multiplier in multipliers[::-1]:
                solution = self.run(time_scale, form_of_ic, initial_conditions, multiplier, method, relative_tolerance,
                                    absolute_tolerance, kuiperbelt)

                solution = solution[:-1]

                size_of_solution = math.floor(len(solution[0][0, :])/10)*10  # Modulo 10 such that the calculations are
                # nicer.
                size = np.append(size, size_of_solution)  # Remember the sizes to check correctness.
                new_size = size_of_solution // size[0]      # Step size for our solution.
                solution = np.array([solution[i][:, :size_of_solution:int(new_size)] for i in range(len(solution))],
                                    dtype=np.ndarray)
                if make_array:
                    solutions = solution
                    make_array = False
                else:
                    solutions = np.concatenate((solutions, solution), axis=0)

        # Notice that the for loop is reversed in order, hence the first four elements represent the largest step size.
        # Also cut off the initial conditions, they result in automatic errors.
        q_4h = solutions[:4, :, 1].astype('float64')
        q_2h = solutions[4:8, :, 1].astype('float64')
        q_h = solutions[8:12, :, 1].astype('float64')
        # Hardcoded sizes, since we always use 4 attributes.
        # Q(2h)-Q(4h)/Q(h)-Q(2h) = 2^p Richardson error formula, now reversed to find p.
        q_lower = q_h - q_2h
        q_upper = q_2h - q_4h
        q_lower[q_lower==0] = 0.1  # Low number to eliminate it from the p array
        q_upper[q_upper==0] = 1.0*10**3  # High number to eliminate it from the p-array
        order = np.round(np.log(np.abs(q_upper / q_lower)) / np.log(2))
        if np.any(order<0):
            print('shiiitt')

        return size, solutions, order

    def run(self, time_scale, form_of_ic, initial_conditions, max_step, method, relative_tolerance, absolute_tolerance,
            kuiperbelt=False):
        '''NOTE: Function to run this class and compute the simulation, returns the ode_solver solution.
         :param time_scale: The time interval over which to be simulated.
         :type time_scale: list
         :param form_of_ic: The form of the initial conditions. On the first entry if True, h, k, p, q coordinates, if
         False, e, var_omega I and big_omega coordinates. On the second entry, True implies big_omega and inclination,
         var_omega are in radians if False they are in degrees.
         :type form_of_ic: np.ndarray
         :param initial_conditions: The initial conditions given in the parameters h, k, p, q or e, var_omega, I and
         big_omega.
         :type initial_conditions: np.ndarray
         :param max_step: The maximum allowed step size for the simulator.
         :type max_step: int
         :param method: Method of the solver.
         :type method: str
         :param relative_tolerance: The relative accuracy of the solver.
         :type relative_tolerance: float
         :param absolute_tolerance: The absolute tolerance of the solver.
         :type absolute_tolerance: float
         :param kuiperbelt: If False, run regular simulation, if True run kuiperbelt simulations.
         :type kuiperbelt: bool
         '''

        if kuiperbelt:
            if not form_of_ic[0]:
                initial_conditionsp = self.initial_condition_builder(*initial_conditions[0], form_of_ic[0])
                initial_conditionsk = self.initial_condition_builder(*initial_conditions[1], form_of_ic[1])
            initial_conditions = np.concatenate((initial_conditionsp.flatten(),initial_conditionsk.flatten()))
            orbital_calculator = self.orbital_calculator_kuiperbelt

        else:
            if not form_of_ic[0]:
                initial_conditions = self.initial_condition_builder(*initial_conditions, form_of_ic[0])
            initial_conditions = initial_conditions.flatten()
            orbital_calculator = self.orbital_calculator_no_kuiperbelt
        alpha_array = self.alpha_matrix(kuiperbelt)

        if kuiperbelt:
            beta = self.beta_values(alpha_array[0], kuiperbelt, free_alpha=alpha_array[2])
            matrix_array = self.a_b_matrices(alpha_array[1], beta, kuiperbelt, free_alpha_bar_matrix=alpha_array[3])
        else:
            beta = self.beta_values(alpha_array[0], kuiperbelt)
            matrix_array = self.a_b_matrices(alpha_array[1], beta, kuiperbelt)

        if method != 'Euler':
            solution = solve_ivp(orbital_calculator, t_span=time_scale, y0=initial_conditions, args=(*matrix_array,),
                                 method=method, rtol=relative_tolerance, atol=absolute_tolerance, max_step=max_step,
                                 first_step=max_step)
            data = solution['y']

        else:
            solution, time = self.Euler_backward_method(time_scale, initial_conditions, max_step, kuiperbelt,
                                                        *matrix_array)

            data = solution
        planet_number = self.planet_number

        h = data[:planet_number, :]
        k = data[planet_number:2 * planet_number, :]
        p = data[2 * planet_number: 3 * planet_number, :]
        q = data[3 * planet_number: 4 * planet_number, :]
        e, I, var_omega, big_omega = self.variable_transformations(h, k, p, q)

        if kuiperbelt:
            free_data = data[4 * planet_number:]
            free_h = free_data[:self.asteroid_number, :]
            free_k = free_data[self.asteroid_number:2 * self.asteroid_number, :]
            free_p = free_data[2 * self.asteroid_number: 3 * self.asteroid_number, :]
            free_q = free_data[3 * self.asteroid_number: 4 * self.asteroid_number, :]
            free_e, free_I, free_var_omega, free_big_omega = self.variable_transformations(free_h, free_k, free_p,
                                                                                           free_q)
            print('kuiperbelt simulation ran succesfully, returning array of parameters seperated for planets and then '
                  'kuiperbelt')
            e = np.concatenate((e, free_e))
            I = np.concatenate((I, free_I))
            var_omega = np.concatenate((var_omega, free_var_omega))
            big_omega = np.concatenate((big_omega, free_big_omega))

        else:
            print('Regular simulation ran succesfully returning values.')
        if method != "Euler":
            return np.array([e, I, var_omega, big_omega, solution], dtype=np.ndarray)
        else:
            return np.array([e, I, var_omega, big_omega, solution, time], dtype=np.ndarray)


if __name__ == '__main__':
    kuiperbelt = True
    order_test = False
    etnos = True
    planet9 = {'mass':15e-6, 'radius':1e-5, 'loanode':100/180*np.pi,'eccentricity':0.6,'smaxis':50,
               'argperiapsis':140/180*np.pi,'orbital inclination':30/180*np.pi,'mean longitude':0}

    if kuiperbelt:
        file_name = 'data.json'
        etnos_file_name = 'etnos.json'
        sim = simulation(file_name=file_name,etnos_file=etnos_file_name, kuiperbelt=kuiperbelt,etnos=etnos, hom_mode=True, total_mass=10000, r_res=8, range_min=40,
                         range_max=100, planet9 = planet9)

        omega = np.array([sim.j['argperiapsis'], sim.s['argperiapsis'], sim.n['argperiapsis'], sim.u['argperiapsis']])
        big_omega = np.array([sim.j['loanode'], sim.s['loanode'], sim.n['loanode'], sim.u['loanode']])
        inclination = np.array([sim.j['orbital inclination'], sim.s['orbital inclination'], sim.n['orbital inclination'],
                                sim.u['orbital inclination']])
        eccentricity = np.array([sim.j['eccentricity'], sim.s['eccentricity'], sim.n['eccentricity'], sim.u['eccentricity']])
        smallaxis = [sim.j['smaxis'], sim.s['smaxis'], sim.n['smaxis'], sim.u['smaxis']]


        #Toevoegen van planet9 parameters
        if planet9:
            omega = np.append(omega, sim.planet9['argperiapsis'])
            big_omega = np.append(big_omega, sim.planet9['loanode'])
            inclination = np.append(inclination, sim.planet9['orbital inclination'])
            eccentricity = np.append(eccentricity, sim.planet9['eccentricity'])
            smallaxis = np.append(smallaxis, sim.planet9['smaxis'])


        omegak = sim.asteroid_argperiapsis
        big_omegak = sim.asteroid_big_omega
        inclinationk = sim.asteroid_inclination
        eccentricityk = sim.asteroid_eccentricity
        var_omegak = omegak + big_omegak
        smallaxis = np.append(smallaxis, sim.asteroid_smaxis)

        var_omega = omega + big_omega
        initial_conditions = np.vstack((eccentricity, var_omega, inclination, big_omega))
        initial_conditionsk = np.vstack((eccentricityk, var_omegak, inclinationk, big_omegak))
        t_eval = [0, 40*10**4]
        max_step = 1000
        form_of_ic = np.array([False, False])
        method = "Euler"#'DOP853'
        r_tol = 10 ** -4
        a_tol = 10 ** -3

        if not order_test:
            if method != "Euler":
                e, I, var_omega, big_omega, solution = sim.run(time_scale=t_eval, form_of_ic=form_of_ic,
                                                       initial_conditions=(initial_conditions,initial_conditionsk), max_step=max_step,
                                                       method=method,
                                                       relative_tolerance=r_tol, absolute_tolerance=a_tol,
                                                       kuiperbelt=kuiperbelt)
            else:
                e, I, var_omega, big_omega, solution, time = sim.run(time_scale=t_eval, form_of_ic=form_of_ic,
                                                               initial_conditions=(initial_conditions, initial_conditionsk),
                                                               max_step=max_step,
                                                               method=method,
                                                               relative_tolerance=r_tol, absolute_tolerance=a_tol,
                                                               kuiperbelt=kuiperbelt)
                tekenen = Od.visualisatie()
                tekenen.PlotParamsVsTijd((e), time, ('e'), alleenplaneten=True, planet9=planet9)
        else:
            length, solutions, order = sim.order_of_error(time_scale=t_eval, form_of_ic=form_of_ic,
                                           initial_conditions=(initial_conditions, initial_conditionsk),
                                           max_step=max_step, method=method, relative_tolerance=r_tol,
                                           absolute_tolerance=a_tol, kuiperbelt=kuiperbelt)
    else:
        file_name = 'data.json'
        sim = simulation(file_name=file_name, kuiperbelt=kuiperbelt, hom_mode=True, total_mass=10000, r_res=2, range_min=10 ** 12,
                         range_max=10 ** 14)
        omega = np.array([sim.j['argperiapsis'], sim.s['argperiapsis'], sim.n['argperiapsis'], sim.u['argperiapsis']])
        big_omega = np.array([sim.j['loanode'], sim.s['loanode'], sim.n['loanode'], sim.u['loanode']])
        inclination = np.array(
            [sim.j['orbital inclination'], sim.s['orbital inclination']
                 , sim.n['orbital inclination'],
              sim.u['orbital inclination']])
        eccentricity = np.array(
            [sim.j['eccentricity'], sim.s['eccentricity']
        , sim.n['eccentricity'], sim.u['eccentricity']])



        var_omega = omega + big_omega
        initial_conditions = np.vstack((eccentricity, var_omega, inclination, big_omega))
        t_eval = [0, 10 ** 5]
        max_step = 10 ** 3
        form_of_ic = np.array([False, False])
        method = 'RK23'
        a_tol = 10 ** -6
        r_tol = 10 ** -10
        e, I, var_omega, big_omega, solution = sim.run(time_scale=t_eval, form_of_ic=form_of_ic,
                                                       initial_conditions=(initial_conditions), max_step=max_step,
                                                       method=method,
                                                       relative_tolerance=r_tol, absolute_tolerance=a_tol,
                                                       kuiperbelt=kuiperbelt)

        smallaxis = [sim.j['smaxis'], sim.s['smaxis'], sim.n['smaxis'], sim.u['smaxis']]


    # fig1, animate, plotobjecten = Od.animatieN(e, I, var_omega, big_omega, smallaxis)
    # anim = animation.FuncAnimation(fig1, animate, fargs=(e, I, var_omega, big_omega, smallaxis),
    #                                 frames=round(t_eval[1] / max_step), interval=10, blit=False)

    # tekenen = Od.visualisatie()
    # # #tekenen.animatieN(e, I, var_omega, big_omega, smallaxis)
    # tekenen.PlotParamsVsTijd((e), solution.t, ('e'), alleenplaneten = True, planet9 = planet9)
    # tekenen.PlotParamsVsTijd((I), solution.t, ('I'), splitsen = False)
    # tekenen.PlotParamsVsTijd((var_omega), solution.t, ('$\overline{\omega}$'), splitsen = False)
    # tekenen.PlotParamsVsTijd((big_omega), solution.t, ('$\Omega$'), splitsen = False)

    # #testen:
    # # alpha, alpha_bar_times_alpha = sim.alpha_matrix(kuiperbelt=False)    #, free_alpha, free_alpha_bar_times_alpha
    # # beta = sim.beta_values(alpha, kuiperbelt=False)#, free_alpha=free_alpha)
    # # a, b = sim.a_b_matrices(alpha_bar_times_alpha, beta, kuiperbelt=False)#, , free_a, free_b
    # #                                         #free_alpha_bar_matrix=free_alpha_bar_times_alpha)  # TODO add in richardson extrapolation Q(2h)-Q(4h)/Q(h)-Q(2h) = 2^p with p the order, for h we take the step size probably
    #
    #
    # # Longtitude of ascending node loanode in renze ding is de big omega
    # # argperiapsis is de argument van de periapsis dat is de omega
    # # orbital inclination is I
    # omega = np.array([sim.j['argperiapsis'], sim.s['argperiapsis'], sim.n['argperiapsis'], sim.u['argperiapsis']])
    # big_omega = np.array([sim.j['loanode'], sim.s['loanode'], sim.n['loanode'], sim.u['loanode']])
    # inclination = np.array([sim.j['orbital inclination'], sim.s['orbital inclination'], sim.n['orbital inclination'],
    #                         sim.u['orbital inclination']])
    # eccentricity = np.array([sim.j['eccentricity'], sim.s['eccentricity'], sim.n['eccentricity'], sim.u['eccentricity']])
    #
    # # omega = np.concatenate((omega, sim.asteroid_argperiapsis))
    # # big_omega = np.concatenate((big_omega, sim.asteroid_big_omega))
    # # inclination = np.concatenate((inclination, sim.asteroid_inclination))
    # # eccentricity = np.concatenate((eccentricity, sim.asteroid_eccentricity))
    #
    #
    # var_omega = omega + big_omega
    # initial_conditions = np.vstack((eccentricity, var_omega, inclination, big_omega))
    # t_eval = [0, 365.25 * 24 * 3600 * 10 ** 12]
    # max_step = 365.25 * 24 * 3600 * 10 ** 9
    # form_of_ic = False
    # method = 'RK23'
    # a_tol = 10 ** 4
    # r_tol = 10 ** 3
    # e, I, var_omega, big_omega, solution = sim.run(time_scale=t_eval, form_of_ic=form_of_ic,
    #                                          initial_conditions=initial_conditions, max_step=max_step, method=method,
    #                                          relative_tolerance=r_tol, absolute_tolerance=a_tol, kuiperbelt=False)   #, free_e, free_I, free_var_omega, free_big_omega
    #
    # smallaxis = [sim.j['smaxis'],sim.s['smaxis'],sim.n['smaxis'],sim.u['smaxis']]
    #
    # # e = np.concatenate((e,free_e))
    # # I = np.concatenate((I,free_I))
    # # var_omega = np.concatenate((var_omega,free_var_omega))
    # # big_omega = np.concatenate((big_omega,free_big_omega))
    # # smallaxis = np.concatenate((np.array(smallaxis),sim.asteroid_smaxis))
    #
    # fig1, animate, plotobjecten = Od.animatieN(e,I,var_omega,big_omega,smallaxis)
    # anim = animation.FuncAnimation(fig1, animate, fargs=(e, I, var_omega, big_omega, smallaxis),
    #                                frames=round(t_eval[1]/max_step), interval=10, blit=False)
    # # Writer = animation.writers['ffmpeg']
    # # writer = Writer(fps=100)
    # # anim.save('yay.mp4',writer=writer)
