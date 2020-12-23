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
# specificaties van kuiperbelt objecten kan je in sim = ST.simulation( .... ) vinden.
planet9 = False
etnos = True


file_name = 'data.json'
etnos_file_name = False

if etnos:
    etnos_file_name = 'etnos.json'

sim = ST.simulation(file_name=file_name, etnos_file=etnos_file_name, kuiperbelt=kuiperbelt, etnos=etnos, hom_mode=True,
                 total_mass=10000, r_res=5, range_min=40,
                 range_max=100, planet9=planet9)


omega = np.array([sim.j['argperiapsis'], sim.s['argperiapsis'], sim.n['argperiapsis'], sim.u['argperiapsis']])
big_omega = np.array([sim.j['loanode'], sim.s['loanode'], sim.n['loanode'], sim.u['loanode']])
inclination = np.array([sim.j['orbital inclination'], sim.s['orbital inclination'], sim.n['orbital inclination'],
     sim.u['orbital inclination']])
eccentricity = np.array([sim.j['eccentricity'], sim.s['eccentricity'], sim.n['eccentricity'], sim.u['eccentricity']])
smallaxis = [sim.j['smaxis'], sim.s['smaxis'], sim.n['smaxis'], sim.u['smaxis']]

if planet9:
    omega = np.append(omega, sim.planet9['argperiapsis'])
    big_omega = np.append(big_omega, sim.planet9['loanode'])
    inclination = np.append(inclination, sim.planet9['orbital inclination'])
    eccentricity = np.append(eccentricity, sim.planet9['eccentricity'])
    smallaxis = np.append(smallaxis, sim.planet9['smaxis'])

if kuiperbelt or etnos:
    omegak = sim.asteroid_argperiapsis
    big_omegak = sim.asteroid_big_omega
    inclinationk = sim.asteroid_inclination
    eccentricityk = sim.asteroid_eccentricity
    var_omegak = omegak + big_omegak
    smallaxis = np.append(smallaxis, sim.asteroid_smaxis)

var_omega = omega + big_omega
initial_conditions = np.vstack((eccentricity, var_omega, inclination, big_omega))
if kuiperbelt or etnos:
    initial_conditionsk = np.vstack((eccentricityk, var_omegak, inclinationk, big_omegak))
t_eval = [0, 4*10**5]
max_step = 10 ** 2
form_of_ic = np.array([False, False])
method = 'DOP853'
a_tol = 10 ** -6
r_tol = 10 ** -10

if kuiperbelt or etnos:
    e, I, var_omega, big_omega, solution = sim.run(time_scale=t_eval, form_of_ic=form_of_ic,
                                               initial_conditions=(initial_conditions,initial_conditionsk), max_step=max_step,
                                               method=method,
                                               relative_tolerance=r_tol, absolute_tolerance=a_tol,
                                               kuiperbelt=True)
else:
    e, I, var_omega, big_omega, solution = sim.run(time_scale=t_eval, form_of_ic=form_of_ic,
                                               initial_conditions=(initial_conditions), max_step=max_step,
                                               method=method,
                                               relative_tolerance=r_tol, absolute_tolerance=a_tol,
                                               kuiperbelt=kuiperbelt)


tekenen = Od.visualisatie()
# tekenen.animatieN(e, I, var_omega, big_omega, smallaxis, plot_range=[-700,700])
# tekenen.PlotParamsVsTijd((e), solution.t, ('e'), alleenplaneten=False, planet9=planet9)
tekenen.PlotParamsVsTijd((e, I, var_omega, big_omega), solution.t, (r'$e$', r'$I$ (rad)',
                                                                    r'$\varpi$ (rad)',r'$\Omega$ (rad)'),
                          alleenplaneten=False, planet9=planet9, legend=False)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=100)
# tekenen.anim.save('filmpje massa x1000.mp4',writer=writer)