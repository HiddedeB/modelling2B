import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.gridspec as gridspec

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
    ax.set_xlim3d([-2 * 10 ** 12, 2 * 10 ** 12])
    ax.set_xlabel('X')
    ax.set_ylim3d([-2 * 10 ** 12, 2 * 10 ** 12])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-2 * 10 ** 12, 2 * 10 ** 12])
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


class visualisatie():
    '''Class to plot/animate orbits '''

    def __init__(self):
        '''NOTE: '''

    def r(self, theta, eccentricity, smaxis, omega):
        b_squared = (1 - eccentricity ** 2) * smaxis ** 2
        c = eccentricity * smaxis

        return b_squared / (smaxis + c * np.cos(theta - omega))

    # The x,y,z-coordinates of planetary motion for the right lagrange parameters
    def xyz(self, theta, eccentricity, smaxis, Omega, omega, I):
        x = self.r(theta, eccentricity, smaxis, omega) * (np.cos(Omega) * np.cos(omega + theta) -
                                                     np.sin(Omega) * np.sin(omega + theta) * np.cos(I))
        y = self.r(theta, eccentricity, smaxis, omega) * (np.sin(Omega) * np.cos(omega + theta) +
                                                     np.cos(Omega) * np.sin(omega + theta) * np.cos(I))
        z = self.r(theta, eccentricity, smaxis, omega) * (np.sin(omega + theta) * np.sin(I))

        return x, y, z, np.array([x, y, z])

    def plot_elips_3d(self, theta, eccentricity, smaxis, Omega, omega, I, input):
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
        self.X, self.Y, self.Z, vector = self.xyz(theta, eccentricity, smaxis, Omega, omega, I)

        self.figure = plt.figure()
        p3.Axes3D(self.figure)
        plt.plot(self.X,self.Y,self.Z)

        # plt.show()
        return vector

    def animateN(self, i, e, I, var, big_omega, smallaxis):
        '''Update functie voor animatieN functie'''
        self.theta_values = np.linspace(0, 2 * np.pi, 10 ** 3)

        for (line, eccentriciteit, Inclination, var_omega, Omega, korteas) in zip(self.plotobjecten, e, I, var,
                                                                                  big_omega,
                                                                                  smallaxis):
            self.X, self.Y, self.Z, vector = self.xyz(self.theta_values, eccentriciteit[i], korteas,
                                  Omega[i], var_omega[i] - Omega[i],
                                  Inclination[i])
            line[0].set_data([self.X, self.Y])
            line[0].set_3d_properties(self.Z)

    def animatieN(self, e, I, var, big_omega, smallaxis, plot_range=[-10,10]):
        '''
        Functie voor het animeren van de planeetbanen.
        '''

        # figuur definieren
        self.figureN = plt.figure()
        self.ax = p3.Axes3D(self.figureN)
        self.ax.set_xlim3d(plot_range)
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d(plot_range)
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d(plot_range)
        self.ax.set_zlabel('Z')

        self.plotobjecten = []
        for i in range(np.shape(e)[0]+1):
            self.plotobjecten.append(self.ax.plot([], [], []))


        self.anim = animation.FuncAnimation(self.figureN, self.animateN, fargs=(e, I, var, big_omega, smallaxis),
                                       frames=round(np.shape(e)[1]), interval=1, blit=False)
        #plt.show()

    def PlotParamsVsTijd(self,param, tijd, paramname, alleen = '', planet9 = False, legend=True, savefigure = False, **kwargs):
        '''NOTE: a function to plot multiple parameters against time.
        :param param: tuple of parameters to plot. formatted as outputted by simulation.run()
        :type param: tuple
        :param tijd: time of simulation
        :type *args: np.array
        :param paramname: tuple containing names of parameters as strings
        :type paramname: tuple
        :param alleen: Three options ''/'planeten'/'objecten' plots both planets and objects, only planets or only objects respectively
        :type alleen: string
        :param planet9: boolean die aangeeft of planet 9 wel of niet in de simulatie zit.
        :type planet9: boolean
        :param savefigure: If True and paramset in **kwargs, then the figure will be saved and not shown. location depends on value of alleen. Also only works for alleen = '' or 'objecten'
        :type savefigure: Boolean
        :param paramset: **kwargs argument. Contains mass, eccentricity and semimajor axis values for planet 9. Used in variation of parameters
        :type paramset: tuple
        '''
        if alleen == '':
            self.figureT = plt.figure()
            self.grid = self.figureT.add_gridspec(len(paramname),2)
            self.axs = self.grid.subplots()
            if planet9:
                nrplanets = 5
            else:
                nrplanets = 4
            if len(paramname) == 1:
                self.axs[0].plot(tijd,param.T[:,0:nrplanets])
                self.axs[0].set(xlabel='Time (y)', ylabel = paramname[0], title='Planets')

                self.axs[1].plot(tijd,param.T[:,nrplanets:])
                self.axs[1].set(xlabel='Time (y)', title='Objects')
            else:
                j = 1
                for i in param:
                    if j==1:
                        tits = ['Planets','Objects']
                    else:
                        tits = ['','']
                    self.axs[j-1,0].plot(tijd,i.T[:,0:nrplanets])
                    self.axs[j - 1, 0].set(xlabel='Time (y)', ylabel = paramname[j-1], title=tits[0])

                    self.axs[j-1,1].plot(tijd,i.T[:,nrplanets:])
                    self.axs[j - 1, 1].set(xlabel='Time (y)', title=tits[1])

                    j += 1
            if legend:
                if planet9:
                    self.figureT.legend(['Jupiter','Saturn','Uranus','Neptune','planet9'],loc = 'upper left')
                else:
                    self.figureT.legend(['Jupiter','Saturn','Uranus','Neptune'],loc = 'upper left')

            if savefigure == True and 'paramset' in kwargs:
                paramset = kwargs.get("paramset")
                self.figureT.suptitle('mass: {:.2E}, eccentricity: {:.2f}, a: {}'.format(paramset[0], paramset[1], paramset[2]))
                plt.savefig('plotjes/combi/mass{:.2E}eccentricity{:.2f}a{}.png'.format(paramset[0], paramset[1], paramset[2]))
                plt.close()
        elif alleen == 'planeten':

            if planet9:
                nrplanets = 5
            else:
                nrplanets = 4

            if len(paramname) == 1:
                plt.plot(tijd, param.T[:, 0:nrplanets])
                plt.xlabel('Time (y)')
                plt.ylabel(paramname)
            else:
                j = 1
                for i in param:
                    plt.subplot(len(param), 1, j)
                    plt.plot(tijd, i.T[:, 0:nrplanets])
                    plt.xlabel('Time (y)')
                    plt.ylabel(paramname[j - 1])

                    j += 1
            if legend:
                if planet9:
                    plt.legend(['Jupiter', 'Saturn', 'Uranus', 'Neptune', 'planet9'], loc='upper left')
                else:
                    plt.legend(['Jupiter', 'Saturn', 'Uranus', 'Neptune'], loc='upper left')
        elif alleen == 'objecten':

            if planet9:
                nrplanets = 5
            else:
                nrplanets = 4

            if len(paramname) == 1:
                plt.plot(tijd, param.T[:, nrplanets:])
                plt.xlabel('Time (y)')
                plt.ylabel(paramname)
            else:
                j = 1
                for i in param:
                    plt.subplot(len(param), 1, j)
                    plt.plot(tijd, i.T[:, nrplanets:])
                    plt.xlabel('Time (y)')
                    plt.ylabel(paramname[j - 1])

                    j += 1
            if savefigure == True and 'paramset' in kwargs:
                paramset = kwargs.get("paramset")
                plt.suptitle('mass: {:.2E}, eccentricity: {:.2f}, a: {}'.format(paramset[0], paramset[1], paramset[2]))
                plt.savefig('plotjes/kuiperonly/kleinesimulatie/massavariatie/mass{:.2E}eccentricity{:.2f}a{}.png'.format(paramset[0], paramset[1], paramset[2]))
                plt.close()

    # def PlotParamsVsTijdKuiper(self,param, tijd, paramname, alleenplaneten = False, planet9 = False, legend=True, paramset = (0,0,0)):
    #     '''NOTE: a function to plot multiple parameters against time.
    #     :param param: tuple of parameters to plot. formatted as outputted by simulation.run()
    #     :type param: tuple
    #     :param tijd: time of simulation
    #     :type *args: np.array
    #     :param paramname: tuple containing names of parameters as strings
    #     :type paramname: tuple
    #     :param alleenplaneten: True geeft de parameter geplot tegen de tijd voor alleen de planeten (4 of 5). False geeft
    #     twee kolommen waarbij de linker de planeten bevat en de rechter de andere objecten
    #     :type alleenplaneten: Boolean
    #     :param planet9: boolean die aangeeft of planet 9 wel of niet in de simulatie zit.
    #     :type planet9: boolean
    #     '''
    #     if not alleenplaneten:
    #         self.figureT = plt.figure()
    #         self.grid = self.figureT.add_gridspec(len(paramname),1)
    #         self.axs = self.grid.subplots()
    #         if planet9:
    #             nrplanets = 5
    #         else:
    #             nrplanets = 4
    #         if len(paramname) == 1:
    #             self.axs.plot(tijd,param.T[:,nrplanets:])
    #             self.axs.set(xlabel='Time (y)', title='Objects')
    #         else:
    #             j = 1
    #             for i in param:
    #                 if j==1:
    #                     tits = ['Planets','Objects']
    #                 else:
    #                     tits = ['','']
    #                 self.axs[j-1].plot(tijd,i.T[:,nrplanets:])
    #                 self.axs[j - 1].set(xlabel='Time (y)')#, title=tits[1])
    #                 j += 1
    #         # self.figureT.suptitle('mass: {}, eccentricity: {}, a: {}'.format(paramset[0],paramset[1],paramset[2]))
    #         # plt.savefig('plotjes/kuiperonly/konly_mass_{}_eccentricity_{}_a_{}.png'.format(paramset[0],paramset[1],paramset[2]))
    #         # plt.close()
    #
    #
    #
    #         # if legend:
    #         #     if planet9:
    #         #         self.figureT.legend(['Jupiter','Saturn','Uranus','Neptune','planet9'],loc = 'upper left')
    #         #     else:
    #         #         self.figureT.legend(['Jupiter','Saturn','Uranus','Neptune'],loc = 'upper left')
    #
    #     elif alleen == 'planeten':
    #
    #         if planet9:
    #             nrplanets = 5
    #         else:
    #             nrplanets = 4
    #
    #         if len(paramname) == 1:
    #             plt.plot(tijd, param.T[:,0:nrplanets])
    #             plt.xlabel('Time (y)')
    #             plt.ylabel(paramname)
    #         else:
    #             j = 1
    #             for i in param:
    #
    #                 plt.subplot(len(param),1,j)
    #                 plt.plot(tijd, i.T[:,0:nrplanets])
    #                 plt.xlabel('Time (y)')
    #                 plt.ylabel(paramname[j-1])
    #
    #                 j += 1
    #         if legend:
    #             if planet9:
    #                 plt.legend(['Jupiter','Saturn','Uranus','Neptune','planet9'],loc = 'upper left')
    #             else:
    #                 plt.legend(['Jupiter','Saturn','Uranus','Neptune'],loc = 'upper left')
    #     elif alleen == 'objecten':
    #
    #         if planet9:
    #             nrplanets = 5
    #         else:
    #             nrplanets = 4
    #
    #         if len(paramname) == 1:
    #             plt.plot(tijd, param.T[:,nrplanets:])
    #             plt.xlabel('Time (y)')
    #             plt.ylabel(paramname)
    #         else:
    #             j = 1
    #             for i in param:
    #
    #                 plt.subplot(len(param),1,j)
    #                 plt.plot(tijd, i.T[:,nrplanets:])
    #                 plt.xlabel('Time (y)')
    #                 plt.ylabel(paramname[j-1])
    #
    #                 j += 1



    def animate(self,i , param, smallaxis):
        '''NOTE: function used to animate the plot object in ParamVsA'''
        for (line,parameter, a) in zip(self.plotobjecten,param,smallaxis):
            line[0].set_data([a,parameter[i]])

    def ParamVsA(self,param,smallaxis,paramnaam):
        ''' NOTE: a function to plot a specific parameter against the size of the small axis.
        :param param: parameter that needs to be plotted against a. an (n,t) array with n objects plotted over time t
        :type param: np.ndarray()
        :param smallaxis: list containing the smallaxis (a) parameters
        :type smallaxis: list
        :param paramnaam: string of parameter name that needs to be plotted
        :type paramnaam: str
        '''
        self.figureP = plt.figure()

        j = 1
        # for i in param:
        plt.subplot(len(param),1,j)
        self.ax = plt.axes(xlim=(smallaxis.min(), smallaxis.max()), ylim=(min(0,param.min()),param.max()))
        self.ax.set_ylabel(paramnaam)
        self.ax.set_xlabel('a')

        self.plotobjecten = []
        for i in range(np.shape(param)[0]):
            self.plotobjecten.append(self.ax.plot([], [],'o'))
        plt.legend(['Jupiter','Saturn','Uranus','Neptune'])

        self.anim = animation.FuncAnimation(self.figureP, self.animate, fargs=(param,smallaxis),
                                       frames=round(np.shape(param)[1]), interval=1, blit=False)




