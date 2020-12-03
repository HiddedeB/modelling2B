import simulation as ST
import numpy as np
from data import create_pdh
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from numba import njit
import Orbitdrawer as Od
from matplotlib import animation
from matplotlib import pyplot as plt

kuiperbelt = True
if kuiperbelt:
    file_name = 'data.json'
    sim = ST.simulation(file_name=file_name, kuiperbelt=kuiperbelt, hom_mode=True, total_mass=10000, r_res=5, range_min=10**12,
                     range_max=10 ** 13)
    omega = np.array([sim.j['argperiapsis'], sim.s['argperiapsis'], sim.n['argperiapsis'], sim.u['argperiapsis']])
    big_omega = np.array([sim.j['loanode'], sim.s['loanode'], sim.n['loanode'], sim.u['loanode']])
    inclination = np.array(
        [sim.j['orbital inclination'], sim.s['orbital inclination'], sim.n['orbital inclination'],
         sim.u['orbital inclination']])
    eccentricity = np.array(
        [sim.j['eccentricity'], sim.s['eccentricity'], sim.n['eccentricity'], sim.u['eccentricity']])
    var_omega = omega + big_omega


    omegak = sim.asteroid_argperiapsis
    big_omegak = sim.asteroid_big_omega
    inclinationk = sim.asteroid_inclination
    eccentricityk = sim.asteroid_eccentricity
    var_omegak = omegak+big_omegak

    initial_conditions = np.vstack((eccentricity, var_omega, inclination, big_omega))
    initial_conditionsk = np.vstack((eccentricityk, var_omegak, inclinationk, big_omegak))

    t_eval = [0, 14 * 365.25 * 24 * 3600 * 10 ** 9]
    max_step = 365.25 * 24 * 3600 * 10 ** 6
    form_of_ic = np.array([False, False])
    method = 'Radau'
    a_tol = 10 ** 4
    r_tol = 10 ** 3
    e, I, var_omega, big_omega, free_e, free_I, free_var_omega, free_big_omega, solution = sim.run(time_scale=t_eval, form_of_ic=form_of_ic,
                                                   initial_conditions=(initial_conditions,initial_conditionsk), max_step=max_step,
                                                   method=method,
                                                   relative_tolerance=r_tol, absolute_tolerance=a_tol,
                                                   kuiperbelt=kuiperbelt)  # , free_e, free_I, free_var_omega, free_big_omega

    smallaxis = [sim.j['smaxis'], sim.s['smaxis'], sim.n['smaxis'], sim.u['smaxis']]

    e = np.concatenate((e,free_e))
    I = np.concatenate((I,free_I))
    var_omega = np.concatenate((var_omega,free_var_omega))
    big_omega = np.concatenate((big_omega,free_big_omega))
    smallaxis = np.concatenate((np.array(smallaxis),sim.asteroid_smaxis))

else:
    file_name = 'data.json'
    sim = ST.simulation(file_name=file_name, kuiperbelt=kuiperbelt, hom_mode=True, total_mass=10000, r_res=5, range_min=10 ** 13,
                     range_max=10 ** 14)
    omega = np.array([sim.j['argperiapsis'], sim.s['argperiapsis'], sim.n['argperiapsis'], sim.u['argperiapsis']])
    big_omega = np.array([sim.j['loanode'], sim.s['loanode'], sim.n['loanode'], sim.u['loanode']])
    inclination = np.array(
        [sim.j['orbital inclination'], sim.s['orbital inclination'], sim.n['orbital inclination'],
         sim.u['orbital inclination']])
    eccentricity = np.array(
        [sim.j['eccentricity'], sim.s['eccentricity'], sim.n['eccentricity'], sim.u['eccentricity']])



    var_omega = omega + big_omega
    initial_conditions = np.vstack((eccentricity, var_omega, inclination, big_omega))
    t_eval = [0, 365.25 * 24 * 3600 * 10 ** 14]
    max_step = 365.25 * 24 * 3600 * 10 ** 10
    form_of_ic = np.array([False])
    method = 'RK23'
    a_tol = 10 ** 4
    r_tol = 10 ** 3
    e, I, var_omega, big_omega, solution = sim.run(time_scale=t_eval, form_of_ic=form_of_ic,
                                                   initial_conditions=initial_conditions, max_step=max_step,
                                                   method=method,
                                                   relative_tolerance=r_tol, absolute_tolerance=a_tol,
                                                   kuiperbelt=kuiperbelt)

    smallaxis = [sim.j['smaxis'], sim.s['smaxis'], sim.n['smaxis'], sim.u['smaxis']]


tekenen = Od.visualisatie()
#tekenen.animatieN(e, I, var_omega, big_omega, smallaxis)
tekenen.PlotParamsVsTijd((e,I,var_omega,big_omega),solution.t,('e','I','var_omega','big_omega'))
#tekenen.ParamVsA(e,smallaxis,('e'))

plt.show()