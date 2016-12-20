# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 15:35:52 2016

@author: S. Bertrand
"""

import numpy as np
import matplotlib.pyplot as plt



class MesuresAuRepos:
    
    def __init__(self):
        nbPoints = 10000

        # mesures
        np.random.seed(1)

        # bruit vitesse
        stdDevVel = 0.05
        meanVel = 0.0
        self.vitesse = np.random.normal(meanVel, stdDevVel, nbPoints)

        # bruit position
        stdDevPos = 0.10
        meanPos = 0.0
        self.position = np.random.normal(meanPos, stdDevPos, nbPoints)


        self.Te = 0.2

        self.temps = np.linspace(0,nbPoints-1, nbPoints) * self.Te


    def plotVitesse(self):
        plt.figure()
        plt.plot(self.temps, self.vitesse, 'k-x')
        plt.xlabel('temps (s)')
        plt.ylabel('vitesse (m/s)')
        plt.grid()

    def plotPosition(self):
        plt.figure()
        plt.plot(self.temps, self.position, 'k-x')
        plt.xlabel('temps (s)')
        plt.ylabel('position (m)')
        plt.grid()



class TrajectoireVraie:

    def __init__(self):   
        
        nbPoints = 100
        
        velocity = np.zeros(nbPoints)
        velocity[0:nbPoints/5] = np.linspace(0.0, 0.1, nbPoints/5)
        velocity[nbPoints/5:nbPoints*4/5] = np.linspace(0.1, 0.1, nbPoints*3/5)
        velocity[4*nbPoints/5:nbPoints] = np.linspace(0.1, 0.0, nbPoints/5)
        self.vitesse = velocity
        
        self.Te = 0.2
        
        self.temps = np.linspace(0,nbPoints-1, nbPoints) * self.Te

        position = np.zeros(nbPoints)
        position[0] = -1.0
        for i in range(0,nbPoints-1):
            position[i+1] = position[i] + self.Te * velocity[i]
        self.position = position
        
    def plotVitesse(self):
        plt.figure()
        plt.plot(self.temps, self.vitesse, 'r-')
        plt.xlabel('temps (s)')
        plt.ylabel('vitesse (m/s)')
        plt.grid()

    def plotPosition(self):
        plt.figure()
        plt.plot(self.temps, self.position, 'r-')
        plt.xlabel('temps (s)')
        plt.ylabel('position (m)')
        plt.grid()  


class MesuresTrajectoire:
    
    def __init__(self, pasMesures=1):
        
        nbPoints = 100
        self.nbPoints = nbPoints        
        
        trajVraie = TrajectoireVraie()          
        
        self.Te = trajVraie.Te
        self.temps = trajVraie.temps
        
        # mesures
        np.random.seed(1)

        #vitesse bruitée
        stdDevVel = 0.05
        meanVel = 0.0
        noiseVel = np.random.normal(meanVel, stdDevVel, nbPoints)
        self.vitesse = trajVraie.vitesse + noiseVel


        # position bruitée
        stdDevPos = 0.10
        meanPos = 0.0
        noisePos = np.random.normal(meanPos, stdDevPos, nbPoints)
        self.position = trajVraie.position + noisePos

        
        # supression d'échantillons de mesures en position
        #nOverM = 1        
        if (pasMesures!=1):
            self.position[0] = np.NaN
            for i in range(1, nbPoints-1):
                if (np.mod(i,pasMesures)!=0):
                    self.position[i] = np.NaN        
        
        

    def plotVitesse(self):
        plt.figure()
        plt.plot(self.temps, self.vitesse, 'kx-')
        plt.xlabel('temps (s)')
        plt.ylabel('vitesse (m/s)')
        plt.grid()

    def plotPosition(self):
        plt.figure()
        plt.plot(self.temps, self.position, 'kx-')
        plt.xlabel('temps (s)')
        plt.ylabel('position (m)')
        plt.grid()

#nOverM = 1
#
## supression d'échantillons de mesures en position
#if (nOverM!=1):
#    positionMeas[0] = np.NaN
#for i in range(1, nbPoints-1):
#    if (np.mod(i,nOverM)!=0):
#        positionMeas[i] = np.NaN