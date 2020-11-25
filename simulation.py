import numpy as np
from data import create_pdh
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from numba import njit
import Orbitdrawer as Od
from matplotlib import animation


class simulation():
    '''Class to compute simulation of the orbital elements'''

    def __init__(self, file_name, **kwargs):
        '''NOTE: Initial function that sets up all the different attributes of our class.
        :param file_name: File_name directing to data file for planethandler.
        :type file_name: str
        :param **kyperbelt: If True the function creates a kyperbelt. For this all the following parameters should be
        given.
        :type **kyperbelt: bool
        :param **hom_mode: If True built a homogeneous kyperbelt in are, if False divide evenly over all orbits.
        :type **hom_mode: bool
        :param **total_mass: Parameter specifying the total mass of the kyperbelt.
        :type **total_mass: int
        :param **r_res: How many orbits the kyperbelt contains.
        :type r_res: int
        :param **range_min: Minimum distance of kyperbelt.
        :type **range_min: float
        :param **range_max: Maximum distance of kyperbelt.
        :type **range_max: float
        '''

        pdh = create_pdh(file_name)

        if 'kyperbelt' in kwargs:
            if 'hom_mode' in kwargs:
                hom_mode = kwargs['hom_mode']
            else:
                hom_mode = False

            pdh.set_kuyperbelt(kwargs['total_mass'], kwargs['r_res'], kwargs['range_min'], kwargs['range_max'],
                               hom_mode)

            self.asteroid_smaxis = pdh.asteroid_attributes['smaxis']
            self.asteroid_number = len(self.asteroid_smaxis)
            self.asteroid_mass = pdh.asteroid_attributes['mass']
            self.asteroid_argperiapsis = pdh.asteroid_attributes['argperiapsis']
            self.asteroid_big_omega = pdh.asteroid_attributes['loanode']
            self.asteroid_inclination = pdh.asteroid_attributes['orbital inclination']
            self.asteroid_eccentricity = pdh.asteroid_attributes['eccentricity']
            self.free_n_vector = 2 * np.pi / (self.asteroid_smaxis ** (3 / 2))

        self.j = pdh.jupiter
        self.s = pdh.saturn
        self.u = pdh.uranus
        self.n = pdh.neptune
        self.sun = pdh.sun
        self.smaxis_vector = np.array([self.j['smaxis'], self.s['smaxis'], self.u['smaxis'], self.n['smaxis']])
        self.mass_vector = np.array([self.j['mass'], self.s['mass'], self.u['mass'], self.n['mass']])
        self.J_2_vector = np.array([14736, 16298, 3343, 3411])
        self.J_4_vector = np.array([-587, -915, -29, -35])
        self.n_vector = 2 * np.pi / (self.smaxis_vector ** (3 / 2))
        self.planet_number = len(self.smaxis_vector)

    def alpha_matrix(self, kyperbelt=False):
        '''NOTE: Function to compute the alpha matrix and the alpha bar and alpha product matrix.
        :param kyperbelt: If False, no kyperbelt simulation; If True built the matrices for kyperbelt simulation.
        :type kyperbelt: True
        '''
        alpha_matrix = np.zeros(self.planet_number)
        alpha_bar_times_alpha_matrix = np.zeros(self.planet_number)
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

        # If the kyperbelt is specified it will execute this if statement.
        if kyperbelt:

            free_alpha = np.zeros((self.asteroid_number, self.planet_number))
            free_alpha_bar_times_alpha = np.zeros((self.asteroid_number, self.planet_number))

            free_alpha = free_alpha + np.fmin(self.asteroid_smaxis / self.smaxis_vector,
                                              self.smaxis_vector / self.asteroid_smaxis)

            free_alpha_bar_times_alpha = free_alpha_bar_times_alpha + np.fmin(self.asteroid_smaxis / self.smaxis_vector,
                                                                              self.smaxis_vector / self.asteroid_smaxis)

            free_smaxis_array = np.array([self.asteroid_smaxis, ] * self.planet_number).transpose()
            smaxis_array = np.array([self.smaxis_vector, ] * self.asteroid_number)

            free_alpha_bar_times_alpha[free_smaxis_array < smaxis_array] = 1

            return np.array([alpha_matrix, alpha_bar_times_alpha_matrix, free_alpha, free_alpha_bar_times_alpha])

        else:
            return np.array([alpha_matrix, alpha_bar_times_alpha_matrix])

    @staticmethod
    def beta_values(alpha_matrix, kyperbelt=False, **kwargs):
        '''NOTE: function to compute the b_{2/3} values needed for the A and B matrices.
        :param alpha_matrix: alpha matrix, consisting of all the different alphas, specified by the equations 7.128 and
        7.129 in Solar System dynamics.
        :type alpha_matrix: np.ndarray
        :param kyperbelt: If True run kyperbelt simulation, if False do not.
        :type kyperbelt: bool
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
                    return (1 / np.pi) * np.cos(2 * phi) / (1 - 2 * alpha * np.cos(2 * phi) + alpha**2)**(3/2)
                else:
                    return 0

            return quad(f, 0, 2 * np.pi, args=(alpha))[0]

        vec_beta1 = np.vectorize(beta1)
        vec_beta2 = np.vectorize(beta2)

        if kyperbelt:
            if 'free_alpha' not in kwargs:
                raise ValueError('Trying to run kyperbelt simulation without free_alpha matrix')
            return np.array([vec_beta1(alpha_matrix), vec_beta2(alpha_matrix), vec_beta1(kwargs['free_alpha']),
                             vec_beta2(kwargs['free_alpha'])])  # Heb de laatste entry van de array verandert van vec_beta2(kwargs['free_alpha_bar']) naar vec_beta2(kwargs['free_alpha'])

        else:
            return np.array([vec_beta1(alpha_matrix), vec_beta2(alpha_matrix)])

    def a_b_matrices(self, alpha_bar_times_alpha_matrix, beta, kyperbelt=False, **kwargs):
        '''NOTE: Function to compute the A and B matrices needed to build the differential equations.
        :param alpha_bar_times_alpha_matrix: Outputted as the second argument of function alpha_matrix. This is the
        product of the regular alpha_matrix and the alpha_bar_matrix as specified by the equations 7.128 and
        7.129 in Solar System dynamics.
        :param beta: The array of two beta matrices as generated by the function beta_values. If we have a free particle
        simulation then beta contains 4 matrices, with the first two containing the regular betas, whilst the third and
        fourth index contain the beta matrices for the free particle simulation.
        :type alpha_bar_times_alpha_matrix: np.ndarray.
        :type beta: np.ndarray.
        :param kyperbelt: If True run kyperbelt simulation, if False do not run kyperbelt simulation.
        :type kyperbelt: bool
        :param **free_alpha_bar_matrix: Matrix of the product of alpha and alpha bar for the free particles.
        :type **free_alpha_bar_matrix: np.ndarray
        '''

        # Off-diagonal components
        sub_matrix = 1 / 4 * self.mass_vector / (self.sun['mass'] + self.mass_vector[:, np.newaxis]) \
                     * alpha_bar_times_alpha_matrix * self.n_vector[:, np.newaxis]

        a = sub_matrix
        b = sub_matrix * beta[0]

        # Diagonal elements i.e. A_{jj} and B_{jj}, first line calculates extra part, second line the sum part
        a_d = self.n_vector * ((self.sun['radius'] / self.smaxis_vector)**2 * (3/2 * self.J_2_vector -
                               (self.sun['radius'] / self.smaxis_vector)**2 * (15/4 * self.J_4_vector +
                                                                               9/8 * self.J_2_vector**2)))
        a_new = a * beta[0]
        a_d = a_d + a_new.sum(axis=1)

        b_d = -self.n_vector * ((self.sun['radius'] / self.smaxis_vector)**2 * (3/2 * self.J_2_vector -
                                (self.sun['radius'] / self.smaxis_vector)**2 * (15/4 * self.J_4_vector +
                                                                                27/8 * self.J_2_vector**2)))
        b_d = b_d - b.sum(axis=1)

        a_matrix = np.diag(a_d) - a * beta[1]
        b_matrix = np.diag(b_d) + b
        if kyperbelt:
            if 'free_alpha_bar_matrix' not in kwargs:
                raise ValueError('Trying to run kyperbelt simulation without the free_alpha_bar_matrix.')

            sub_matrix = 1 / 4 * self.mass_vector / self.sun['mass'] * kwargs['free_alpha_bar_matrix'] * \
                         self.free_n_vector[:, np.newaxis]

            a = sub_matrix
            b = sub_matrix * beta[2]

            a_d = self.free_n_vector * ((self.sun['radius'] / self.asteroid_smaxis)**2 * (3/2 * self.J_2_vector -
                                        (self.sun['radius'] / self.asteroid_smaxis)**2 * (15/4 * self.J_4_vector +
                                                                                          9/8 * self.J_2_vector**2)))
            a_new = a * beta[2]
            a_d = a_d + a_new.sum(axis=1)

            b_d = -self.free_n_vector * ((self.sun['radius'] / self.asteroid_smaxis)**2 * (3/2 * self.J_2_vector -
                                         (self.sun['radius'] / self.asteroid_smaxis)**2 * (15/4 * self.J_4_vector +
                                                                                           27/8 * self.J_2_vector**2)))

            b_d = b_d - b.sum(axis=1)

            free_a_matrix = np.diag(a_d) - a * beta[3]
            free_b_matrix = np.diag(b_d) + b

            return np.array([a_matrix, b_matrix, free_a_matrix, free_b_matrix])

        else:

            return np.array([a_matrix, b_matrix])

    @staticmethod
    def initial_condition_builder(e, var_omega, I, big_omega):
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
        '''

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

    @staticmethod
    # @njit
    def orbital_calculator(t, vector, *args):
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

        if 4 * planet_number != len(vector):  # More objects than just the planets. Implies we have free particles.
            free_a_matrix = args[2]
            free_b_matrix = args[3]
            vector = vector[4 * planet_number:]
            amount_of_particles = int(len(vector)/4)
            free_h_vector = vector[:amount_of_particles]
            free_k_vector = vector[amount_of_particles:2 * amount_of_particles]
            free_p_vector = vector[2 * amount_of_particles: 3 * amount_of_particles]
            free_q_vector = vector[3 * amount_of_particles: 4 * amount_of_particles]

            # Differential equations.
            d_h_free_matrix = free_a_matrix * free_k_vector
            d_h_free_vec = d_h_free_matrix.sum(axis=1)
            d_k_free_matrix = free_a_matrix * free_h_vector
            d_k_free_vec = -d_k_free_matrix.sum(axis=1)
            d_p_free_matrix = free_b_matrix * free_q_vector
            d_p_free_vec = d_p_free_matrix.sum(axis=1)
            d_q_free_matrix = free_b_matrix * free_p_vector
            d_q_free_vec = -d_q_free_matrix.sum(axis=1)
            return np.concatenate((d_h_vec, d_k_vec, d_p_vec, d_q_vec, d_h_free_vec, d_k_free_vec, d_p_free_vec,
                                   d_q_free_vec))

        else:
            return np.concatenate((d_h_vec, d_k_vec, d_p_vec, d_q_vec))

    def order_of_error(self, time_scale, form_of_ic, initial_conditions, max_step, method, relative_tolerance,
                       absolute_tolerance, kyperbelt=False):
        '''NOTE: Function to compute the order of error, we still need to figure out error in which part of the method,
        for now lets assume the error is in the step size, not fully done yet, actively being worked on.'''

        solutions = np.array([], dtype=float)
        for i in range(1, 5):
            if i == 3:
                pass
            elif i == 4:
                pass
            else:
                solution = self.run(time_scale, form_of_ic, initial_conditions, max_step, method, relative_tolerance,
                                    absolute_tolerance, kyperbelt)
                solutions = np.vstack(solutions, solution)

        order = np.round(np.log((solutions[1] - solutions[2]) / (solutions[0] - solutions[1])) / np.log(2))

        return order

    def run(self, time_scale, form_of_ic, initial_conditions, max_step, method, relative_tolerance, absolute_tolerance,
            kyperbelt=False):
        '''NOTE: Function to run this class and compute the simulation, returns the ode_solver solution.
         :param time_scale: The time interval over which to be simulated.
         :type time_scale: list
         :param form_of_ic: The form of the initial conditions. If True, h, k, p, q coordinates, if False, e, var_omega
         I and big_omega coordinates.
         :type form_of_ic: bool
         :param initial_conditions: The initial conditions given in the parameters h, k, p, q or e, var_omega, I and
         big_omega.
         :type initial_conditions: 1darry
         :param max_step: The maximum allowed step size for the simulator.
         :type max_step: int
         :param method: Method of the solver.
         :type method: str
         :param relative_tolerance: The relative accuracy of the solver.
         :type relative_tolerance: float
         :param absolute_tolerance: The absolute tolerance of the solver.
         :type absolute_tolerance: float
         :param kyperbelt: If False, run regular simulation, if True run kyperbelt simulations.
         :type kyperbelt: bool
         '''

        if not form_of_ic:
            initial_conditions = self.initial_condition_builder(*initial_conditions)
        initial_conditions = initial_conditions.flatten()
        alpha_array = self.alpha_matrix(kyperbelt)

        if kyperbelt:
            beta = self.beta_values(alpha_array[0], kyperbelt, free_alpha=alpha_array[2])
            matrix_array = self.a_b_matrices(alpha_array[1], beta, kyperbelt, free_alpha_bar_matrix=alpha_array[3])
        else:
            beta = self.beta_values(alpha_array[0], kyperbelt)
            matrix_array = self.a_b_matrices(alpha_array[1], beta, kyperbelt)

        solution = solve_ivp(self.orbital_calculator, t_span=time_scale, y0=initial_conditions, args=(*matrix_array,),
                             method=method, rtol=relative_tolerance, atol=absolute_tolerance, max_step=max_step)

        planet_number = self.planet_number
        data = solution['y']
        h = data[:planet_number, :]
        k = data[planet_number:2 * planet_number, :]
        p = data[2 * planet_number: 3 * planet_number, :]
        q = data[3 * planet_number: 4 * planet_number, :]
        e, I, var_omega, big_omega = self.variable_transformations(h, k, p, q)

        if kyperbelt:
            free_data = data[4 * planet_number:]
            free_h = free_data[:self.asteroid_number, :]
            free_k = free_data[self.asteroid_number:2 * self.asteroid_number, :]
            free_p = free_data[2 * self.asteroid_number: 3 * self.asteroid_number, :]
            free_q = free_data[3 * self.asteroid_number: 4 * self.asteroid_number, :]
            free_e, free_I, free_var_omega, free_big_omega = self.variable_transformations(free_h, free_k, free_p,
                                                                                           free_q)
            print('Kyperbelt simulation ran succesfully, returning array of parameters seperated for planets and then '
                  'kyperbelt')
            return np.array([e, I, var_omega, big_omega, free_e, free_I, free_var_omega, free_big_omega, solution],
                            dtype=np.ndarray)

        else:
            print('Regular simulation ran succesfully returning values.')
            return np.array([e, I, var_omega, big_omega, solution], dtype=np.ndarray)


if __name__ == '__main__':
    file_name = 'data.json'
    kyperbelt = True
    sim = simulation(file_name=file_name, kyperbelt=kyperbelt, hom_mode=True, total_mass=10000, r_res=4,
                     range_min=10**12, range_max=10**14)

    # Test matrices
    # alpha, alpha_bar_times_alpha = sim.alpha_matrix(kyperbelt=False)    #, free_alpha, free_alpha_bar_times_alpha
    # beta = sim.beta_values(alpha, kyperbelt=False)#, free_alpha=free_alpha)
    # a, b = sim.a_b_matrices(alpha_bar_times_alpha, beta, kyperbelt=False)#, , free_a, free_b
    #                                         #free_alpha_bar_matrix=free_alpha_bar_times_alpha)  # TODO add in richardson extrapolation Q(2h)-Q(4h)/Q(h)-Q(2h) = 2^p with p the order, for h we take the step size probably



    # Longtitude of ascending node loanode in renze ding is de big omega
    # argperiapsis is de argument van de periapsis dat is de omega
    # orbital inclination is I
    # Simulation
    omega = np.array([sim.j['argperiapsis'], sim.s['argperiapsis'], sim.n['argperiapsis'], sim.u['argperiapsis']])
    omega = np.concatenate((omega, sim.asteroid_argperiapsis))
    big_omega = np.array([sim.j['loanode'], sim.s['loanode'], sim.n['loanode'], sim.u['loanode']])
    big_omega = np.concatenate((big_omega, sim.asteroid_big_omega))
    inclination = np.array([sim.j['orbital inclination'], sim.s['orbital inclination'], sim.n['orbital inclination'],
                            sim.u['orbital inclination']])
    inclination = np.concatenate((inclination, sim.asteroid_inclination))
    eccentricity = np.array(
        [sim.j['eccentricity'], sim.s['eccentricity'], sim.n['eccentricity'], sim.u['eccentricity']])
    eccentricity = np.concatenate((eccentricity, sim.asteroid_eccentricity))

    var_omega = omega + big_omega
    initial_conditions = np.vstack((eccentricity, var_omega, inclination, big_omega))
    t_eval = [0, 365.25 * 24 * 3600 * 10 ** 12]
    max_step = 365.25 * 24 * 3600 * 10 ** 9
    form_of_ic = False
    method = 'RK23'
    a_tol = 10 ** 4
    r_tol = 10 ** 3
    e, I, var_omega, big_omega, free_e, free_I, free_var_omega, free_big_omega, solution = sim.run(time_scale=t_eval,
                                                                                                   form_of_ic=form_of_ic,
                                             initial_conditions=initial_conditions, max_step=max_step, method=method,
                                             relative_tolerance=r_tol, absolute_tolerance=a_tol, kyperbelt=kyperbelt)

    # e = np.concatenate((e,free_e))
    # I = np.concatenate((I,free_I))
    # var_omega = np.concatenate((var_omega,free_var_omega))
    # big_omega = np.concatenate((big_omega,free_big_omega))

    smallaxis = [sim.j['smaxis'],sim.s['smaxis'],sim.n['smaxis'],sim.u['smaxis']]
    # smallaxis = np.concatenate((np.array(smallaxis),sim.asteroid_smaxis))

    fig1, animate, plotobjecten = Od.animatieN(e,I,var_omega,big_omega,smallaxis)
    anim = animation.FuncAnimation(fig1, animate, fargs=(e, I, var_omega, big_omega, smallaxis),
                                   frames=round(t_eval[1]/max_step), interval=10, blit=False)
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=100)
    # anim.save('yay.mp4',writer=writer)
