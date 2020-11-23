import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3


def r(theta, eccentricity, smaxis, omega):
    b_squared = (1 - eccentricity ** 2) * smaxis ** 2
    c = eccentricity * smaxis

    return b_squared / (smaxis + c * np.cos(theta - omega))


# The x,y,z-coordinates of planetary motion for the right lagrange parameters
def xyz(theta, eccentricity, smaxis, Omega, omega, I):
    x = r(theta, eccentricity, smaxis, omega) * (np.cos(Omega) * np.cos(omega + theta) -
                                                 np.sin(Omega) * np.sin(omega + theta) * np.cos(I))
    y = r(theta, eccentricity, smaxis, omega) * (np.sin(Omega) * np.cos(omega + theta) +
                                                 np.cos(Omega) * np.sin(omega + theta) * np.cos(I))
    z = r(theta, eccentricity, smaxis, omega) * (np.sin(omega + theta) * np.sin(I))

    return x, y, z, np.array([x, y, z])


# Plot function for 3d plots of the ellipses
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# theta_values = np.linspace(0,2*np.pi,10**3)


def plot_elips_3d(theta, eccentricity, smaxis, Omega, omega, I, input, figure):
    ''' Functie die de elipsbaan van een planeet tekent in 3D. Als je veel plotjes wilt opslaan kan je beter plt.show() weghalen
    NOTE:
    :param theta: hoeken van ellips die getekend moeten worden. voor gehele ellipse vector van 0 tot 360 graden
    :param eccentricity: eccentriciteit van de ellipse
    :param smaxis: to do
    :param Omega: to do
    :param omega: to do
    :param I: to do
    :param input: kleur voor getekende ellipse
    :param figure: figure waarin je ellipse wilt tekenen
    '''
    X, Y, Z, vector = xyz(theta, eccentricity, smaxis, Omega, omega, I)

    figure.plot(X, Y, Z, c=input)

    # plt.show()
    return vector


# def animatie(planeetparameters, smallaxis, steps):
#     '''
#     Functie voor het animeren van de planeetbanen.
#     :param planeetparameters: List met de lengte van het aantal planeten dat
#     geanimeerd moet worden. elementen in list bevatten een lijst met de 4
#     getransformeerde parameters van de ellipsbaan voor elke tijdstap.
#     :param smallaxis: List met de smallaxis parameter van elke planeet die
#     geanimeerd moet worden.
#
#
#     '''
#
#     # figuur definieren
#     fig1 = plt.figure()
#     ax = p3.Axes3D(fig1)
#     ax.set_xlim3d([-2 * 10 ** 11, 2 * 10 ** 11])
#     ax.set_xlabel('X')
#     ax.set_ylim3d([-2 * 10 ** 11, 2 * 10 ** 11])
#     ax.set_ylabel('Y')
#     ax.set_zlim3d([-2 * 10 ** 10, 2 * 10 ** 10])
#     ax.set_zlabel('Z')
#
#     plotobjecten = []
#     print(np.size(planeetparameters))
#     for i in range(np.shape(planeetparameters)[0]):
#         plotobjecten.append(ax.plot([], [], []))
#     print('plotobjecten = ' + str(plotobjecten))
#     print('param = ' + str(planeetparameters))
#     print('smallaxis = ' + str(smallaxis))
#
#     def animate(i, plotobjecten, planeetparameters, smallaxis):
#         print('begin')
#         theta_values = np.linspace(0, 2 * np.pi, 10 ** 3)
#
#         for (line, planeetparam, korteas) in zip(plotobjecten, planeetparameters, smallaxis):
#             print('midden')
#
#             X, Y, Z, vector = xyz(theta_values, planeetparam[i][0], korteas,
#                                   planeetparam[i][3], planeetparam[i][2],
#                                   planeetparam[i][1])
#             print(line)
#             line[0].set_data([X, Y])
#             line[0].set_3d_properties(Z)
#         print('einde')
#         print(i)
#
#     print('a')
#     print('steps = {}'.format(steps))
#     animate(i,plotobjecten, planeetparameters, smallaxis)
#     plt.show()
#     anim = animation.FuncAnimation(fig1, animate, fargs=(plotobjecten, planeetparameters, smallaxis),
#                                    frames=steps, interval=10, blit=False)
#
#     plt.show()

def animatien(planeetparameters, smallaxis, steps):
    '''
    Functie voor het animeren van de planeetbanen.
    :param planeetparameters: List met de lengte van het aantal planeten dat
    geanimeerd moet worden. elementen in list bevatten een lijst met de 4
    getransformeerde parameters van de ellipsbaan voor elke tijdstap.
    :param smallaxis: List met de smallaxis parameter van elke planeet die
    geanimeerd moet worden.


    '''

    # figuur definieren
    fig1 = plt.figure()
    ax = p3.Axes3D(fig1)
    ax.set_xlim3d([-2 * 10 ** 11, 2 * 10 ** 11])
    ax.set_xlabel('X')
    ax.set_ylim3d([-2 * 10 ** 11, 2 * 10 ** 11])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-2 * 10 ** 10, 2 * 10 ** 10])
    ax.set_zlabel('Z')

    plotobjecten = []
    for i in range(np.shape(planeetparameters)[0]):
        plotobjecten.append(ax.plot([], [], []))

    def animate(i, plotobjecten, planeetparameters, smallaxis):
        theta_values = np.linspace(0, 2 * np.pi, 10 ** 3)

        for (line, planeetparam, korteas) in zip(plotobjecten, planeetparameters, smallaxis):
            X, Y, Z, vector = xyz(theta_values, planeetparam[i][0], korteas,
                                  planeetparam[i][3], planeetparam[i][2],
                                  planeetparam[i][1])
            line[0].set_data([X, Y])
            line[0].set_3d_properties(Z)

    return fig1, animate, plotobjecten


def animatieN(e, I, var, big_omega, smallaxis):
    '''
    Functie voor het animeren van de planeetbanen.
    '''

    # figuur definieren
    fig1 = plt.figure()
    ax = p3.Axes3D(fig1)
    ax.set_xlim3d([-2 * 10 ** 18, 2 * 10 ** 18])
    ax.set_xlabel('X')
    ax.set_ylim3d([-2 * 10 ** 18, 2 * 10 ** 18])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-2 * 10 ** 18, 2 * 10 ** 18])
    ax.set_zlabel('Z')

    plotobjecten = []
    for i in range(np.shape(e)[0]):
        plotobjecten.append(ax.plot([], [], []))

    def animate(i, e, I, var, big_omega, smallaxis):
        theta_values = np.linspace(0, 2 * np.pi, 10 ** 3)

        for (line, eccentriciteit, Inclination, var_omega, Omega, korteas) in zip(plotobjecten, e, I, var, big_omega,
                                                                                  smallaxis):
            X, Y, Z, vector = xyz(theta_values, eccentriciteit[i], korteas,
                                  Omega[i], var_omega[i] - Omega[i],
                                  Inclination[i])
            line[0].set_data([X, Y])
            line[0].set_3d_properties(Z)

    return fig1, animate, plotobjecten
