import simulation as ST
import numpy as np
from data import create_pdh
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from numba import njit
import Orbitdrawer as Od
from matplotlib import animation
from matplotlib import pyplot as plt
import itertools as it
from matplotlib import use
use('Agg')



kuiperbelt = True
# specificaties van kuiperbelt objecten kan je in sim = ST.simulation( .... ) vinden.
planet9 = True
varyplanet9mass = True
etnos = True


file_name = 'data.json'
etnos_file_name = False

if etnos:
    etnos_file_name = 'etnos.json'

sim = ST.simulation(file_name=file_name, etnos_file=etnos_file_name, kuiperbelt=kuiperbelt, etnos=etnos, hom_mode=True,
                 total_mass=10000, r_res=5, range_min=40,
                 range_max=100, planet9=planet9)

msun = 1.989*10**30
mearth = 5.972*10**24
mplanet9 = mearth*10/msun
eplanet9 = sim.planet9['eccentricity']
aplanet9 = sim.planet9['smaxis']
massvariation = np.linspace(mplanet9/2,mplanet9, 6)
evariation = np.linspace(eplanet9 - 0.2, eplanet9 + 0.2, 5)
avariation = np.linspace(aplanet9 - 50, aplanet9 + 50, 6)
combinations = list(it.product(massvariation, evariation, avariation))

# j = 0
# for i in combinations:
#     print(i)
#     j+=1
# print(j)
k = 0
for i in combinations:
    sim.mass_vector[-1] = i[0]
    sim.planet9['eccentricity'] = i[1]
    sim.planet9['smaxis'] = i[2]
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
    t_eval = [0, 4*10**7]
    max_step = 2 * 10 ** 2
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
    tekenen.PlotParamsVsTijdKuiper((e,I), solution.t, ('e','I'), alleenplaneten = False, planet9=planet9, paramset = i)
    tekenen.PlotParamsVsTijd((e,I), solution.t, ('e','I'), alleenplaneten = False, planet9=planet9, paramset = i)

    # tekenen.PlotParamsVsTijd((e, I, var_omega, big_omega), solution.t, (r'$e$', r'$I$ (rad)',
    #                                                                     r'$\varpi$ (rad)',r'$\Omega$ (rad)'),
    #                           alleenplaneten=False, planet9=planet9, legend=False)
    # tekenen.ParamVsA(e,smallaxis,'e')

    # Writer = animation.writers['pillow']
    # writer = Writer(fps=100)
    # tekenen.anim.save('filmpje massa x1000.mp4',writer=writer)
    k += 1
    print(k)