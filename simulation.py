import numpy as np
from data import create_pdh
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from numba import njit

class simulation():
    '''Class to compute simulation of the orbital elements'''

    def __init__(self, file_name):
        '''NOTE: Initial function that sets up all the different attributes of our class.
        :param file_name: File_name directing to data file for planethandler.
        :type file_name: str
        '''

        pdh = create_pdh(file_name)
        self.j = pdh.jupiter
        self.s = pdh.saturn
        self.u = pdh.uranus
        self.n = pdh.neptune
        self.sun = pdh.sun
        self.smaxis_vector = np.array([self.j['smaxis'], self.s['smaxis'], self.u['smaxis'], self.n['smaxis']])
        self.mass_vector = np.array([self.j['mass'], self.s['mass'], self.u['mass'], self.n['mass']])
        self.J_2_vector = np.array([14736, 16298, 3343, 3411])
        self.J_4_vector = np.array([-587, -915, -29, -35])
        self.n_vector = 2 * np.pi / (self.smaxis_vector**(3 / 2))

    def alpha_matrix(self):
        '''NOTE: Function to compute the alpha matrix and the alpha bar and alpha product matrix. '''
        alpha_matrix = np.zeros(4)
        alpha_bar_times_alpha_matrix = np.zeros(4)
        for i in range(len(self.smaxis_vector)):
            current_alpha = np.fmin(self.smaxis_vector[i] / np.delete(self.smaxis_vector, i),
                                    np.delete(self.smaxis_vector, i) / self.smaxis_vector[i])
            current_alpha = np.insert(current_alpha, obj=i, values=0)
            alpha_matrix = np.vstack((alpha_matrix, current_alpha))

            alpha_bar_times_alpha_matrix = np.vstack((alpha_bar_times_alpha_matrix,
                                                      np.concatenate((np.repeat(1, i), current_alpha[i:]))
                                                      * current_alpha))

        alpha_matrix = alpha_matrix[1:]
        alpha_bar_times_alpha_matrix = alpha_bar_times_alpha_matrix[1:]

        return alpha_matrix, alpha_bar_times_alpha_matrix

    @staticmethod
    def beta_values(alpha_matrix):
        '''NOTE: function to compute the b_{2/3} values needed for the A and B matrices.
        :param alpha_matrix: alpha matrix, consisting of all the different alphas, specified by the equations 7.128 and
        7.129 in Solar System dynamics.
        :type alpha_matrix: ndarray.
        '''

        def beta1(alpha):
            def g(phi, alpha):
                # Angle phi
                if alpha != 0:
                    return (1 / np.pi) * np.cos(phi) / (1 - 2 * alpha * np.cos(phi) + alpha**2)**(3 / 2)
                else:
                    return 0

            return quad(g, 0, 2 * np.pi, args=(alpha))[0]

        def beta2(alpha):
            def f(phi, alpha):
                # Angle 2*phi
                if alpha != 0:
                    return (1 / np.pi) * np.cos(2 * phi) / (1 - 2 * alpha * np.cos(2 * phi) + alpha**2)**(3 / 2)
                else:
                    return 0

            return quad(f, 0, 2 * np.pi, args=(alpha))[0]

        vec_beta1 = np.vectorize(beta1)
        vec_beta2 = np.vectorize(beta2)

        return np.array([vec_beta1(alpha_matrix), vec_beta2(alpha_matrix)])

    def a_b_matrices(self, alpha_bar_times_alpha_matrix, beta):
        '''NOTE: Function to compute the A and B matrices needed to build the differential equations.
        :param alpha_bar_times_alpha_matrix: Outputted as the second argument of function alpha_matrix. This is the
        product of the regular alpha_matrix and the alpha_bar_matrix as specified by the equations 7.128 and
        7.129 in Solar System dynamics.
        :param beta: The array of two beta matrices as generated by the function beta_values.
        :type alpha_bar_times_alpha_matrix: ndarray.
        :type beta: ndarray.
        '''

        # Off-diagonal components
        sub_matrix = 1/4 * self.mass_vector / (self.sun['mass'] + self.mass_vector[:, np.newaxis])\
                     * alpha_bar_times_alpha_matrix * self.n_vector[:, np.newaxis]

        a = -sub_matrix
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
        b_d = b_d + b.sum(axis=1)

        a_matrix = np.diag(a_d) + a * beta[1]
        b_matrix = np.diag(b_d) + b

        return a_matrix, b_matrix

    @staticmethod
    def initial_condition_builder(e, var_omega, I, big_omega):
        '''NOTE: Initial condition maker, if the initial conditions are not in h, k, p and q form, then this function
        can be called to transfer them.
        :param e: The eccentricities of the orbits.
        :type e: ndarray
        :param var_omega: The difference between big_omega and omega for every orbit.
        :type var_omega: ndarray
        :param I: The angles of the orbit w.r.t. the reference frame.
        :type I: ndarray
        :param big_omega: The angles between the ascending node and the reference frame x-hat direction for every orbit
        :type big_omega: ndarray
        '''

        h = e * np.sin(var_omega)
        k = e * np.cos(var_omega)
        p = I * np.sin(big_omega)
        q = I * np.cos(big_omega)

        return np.concatenate((h, k, p, q))

    @staticmethod
    def variable_transformations(h, k, p, q):
        '''NOTE: Function to transfer the new made coordinates h, k, p, q to e, I, var_omega and big_omega'''
        return np.array([np.sqrt(h**2 + k**2), np.sqrt(p**2 + q**2), np.arcsin(h/np.sqrt(h**2 + k**2)),
                         np.arcsin(p/np.sqrt(p**2 + q**2))])

    @staticmethod
    @njit
    def orbital_calculator(t, vector, a_matrix, b_matrix):
        '''NOTE: function to run the solver with, vector contains h, k, p, q in that order. Furthermore, the values are
        taken to go from closes to the sun to most outward.
        :param a_matrix: The A matrix of the system of equations in eq 7.136 of Solar System Dynamics.
        :param b_matrix: The B matrix of the system of equations mentioned before.
        :type a_matrix: ndarray
        :type b_matrix: ndarray
        '''

        # Setting up the differential equations according to 7.25 and 7.26 from Solar System Dynamics. Since the a and
        # b matrices are both square and contain as many entries as planets, we use this for splitting our data vector.
        planet_number = a_matrix.shape[0]
        h_vector = vector[:planet_number]
        k_vector = vector[planet_number:2 * planet_number]
        p_vector = vector[2 * planet_number: 3 * planet_number]
        q_vector = vector[3 * planet_number:]

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

    def run(self, time_scale, form_of_ic, initial_conditions, max_step, method, relative_tolerance, absolute_tolerance):
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
         '''

        if not form_of_ic:
            initial_conditions = np.transpose(initial_conditions)
            initial_conditions = self.initial_condition_builder(*initial_conditions)
        initial_conditions = initial_conditions.flatten()
        alpha_matrix, alpha_times_alpha_bar_matrix = self.alpha_matrix()
        beta = self.beta_values(alpha_matrix)
        a_matrix, b_matrix = self.a_b_matrices(alpha_times_alpha_bar_matrix, beta)

        solution = solve_ivp(self.orbital_calculator, t_span=time_scale, y0=initial_conditions, args=(a_matrix,
                            b_matrix), method=method, rtol=relative_tolerance, atol=absolute_tolerance,
                            max_step=max_step)

        planet_number = a_matrix.shape[0]
        data = solution['y']
        h = data[:planet_number, :]
        k = data[planet_number:2 * planet_number, :]
        p = data[2 * planet_number: 3 * planet_number, :]
        q = data[3 * planet_number:, :]
        e, I, var_omega, big_omega = self.variable_transformations(h, k, p, q)

        return e, I, var_omega, big_omega, solution

if __name__ == '__main__':
    file_name = 'data.json'
    sim = simulation(file_name=file_name)
    alpha, alpha_bar_times_alpha = sim.alpha_matrix()
    beta = sim.beta_values(alpha)
    a, b = sim.a_b_matrices(alpha_bar_times_alpha, beta) # TODO add in richardson extrapolation Q(2h)-Q(4h)/Q(h)-Q(2h) = 2^p with p the order, for h we take the step size probably
    # Longtitude of ascending node loanode in renze ding is de big omega
    # argperiapsis is de argument van de periapsis dat is de omega
    # orbital inclination is I
    omega = np.array([sim.j['argperiapsis'], sim.s['argperiapsis'], sim.n['argperiapsis'], sim.u['argperiapsis']])
    big_omega = np.array([sim.j['loanode'], sim.s['loanode'], sim.n['loanode'], sim.u['loanode']])
    inclination = np.array([sim.j['orbital inclination'], sim.s['orbital inclination'], sim.n['orbital inclination'],
                            sim.u['orbital inclination']])
    eccentricity = np.array([sim.j['eccentricity'], sim.s['eccentricity'], sim.n['eccentricity'], sim.u['eccentricity']])
    var_omega = omega+big_omega
    initial_conditions = np.vstack((eccentricity, var_omega, inclination, big_omega))

    t_eval = [0, 365.25*24*3600*10**6]
    max_step = 365.25*24*3600*10**3
    form_of_ic = False
    method = 'RK23'
    a_tol = 10**4
    r_tol = 10**3
    e, I, var, big_omega, solution = sim.run(time_scale=t_eval, form_of_ic=form_of_ic,
                                             initial_conditions=initial_conditions, max_step=max_step, method=method,
                                             relative_tolerance=r_tol, absolute_tolerance=a_tol)