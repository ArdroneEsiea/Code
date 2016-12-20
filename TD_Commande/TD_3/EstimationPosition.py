# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 15:45:07 2016

@author: S. Bertrand
"""

import numpy as np
import matplotlib.pyplot as plt
import donnees
import KalmanFilter as kf



# caractéristiques des capteurs (cf. question II.1)
# ------------------------------------------------------

# écart type des mesures de position
# ****** A MODIFIER EN TD ******
stdDevPos = 0.100118927317
# *******************************

# écart type des mesures de vitesse
# ****** A MODIFIER EN TD ******
stdDevVel = 0.0499392921844
# *******************************



# chargement des données
# ------------------------------------------

# trajectoire réelle du robot (pour vérification seulement, non connue en pratique)
trajectoireVraie = donnees.TrajectoireVraie()
#trajectoireVraie.plotPosition()
#trajectoireVraie.plotVitesse()


# mesures de la trajectoire du robot
mesuresTrajectoire = donnees.MesuresTrajectoire(pasMesures=10)
mesuresTrajectoire.plotPosition()
mesuresTrajectoire.plotVitesse()


# nombre de points
nbPoints = mesuresTrajectoire.nbPoints

# période d'échantillonnage 
Te = mesuresTrajectoire.Te

# temps
temps = mesuresTrajectoire.temps



# hypothèse de position initiale du robot 
# -----------------------------------------
x0Est = np.array([0.0])



# estimation de la position par intégration numérique des mesures de vitesse
# -----------------------------------------------------------------------------

# structures de données pour sauvegarder les données
integratedPositionFromVelMeas = np.zeros_like(trajectoireVraie.position)
integratedPositionFromVelMeas[0] = x0Est[0]

# calcul de la position estimée tout au long de la trajectoire
for i in range(1, nbPoints):

    # ****** A MODIFIER EN TD ********************
    integratedPositionFromVelMeas[i] = integratedPositionFromVelMeas[i-1] + Te * mesuresTrajectoire.vitesse[i-1]

    # ********************************************


# affichages
plt.close('all')

plt.figure()
plt.plot(temps, trajectoireVraie.position, 'r-')
plt.plot(temps, integratedPositionFromVelMeas, 'g-')
plt.xlabel('t (s)')
plt.ylabel('position (m)')
plt.grid()


plt.figure()
plt.plot(temps, integratedPositionFromVelMeas - trajectoireVraie.position, 'g-')
plt.xlabel('t (s)')
plt.ylabel('erreur en position (m)')
plt.grid()





# estimation de la position par un filtre de Kalman
# -----------------------------------------------------
        
# dimensions
# ******* A MODIFIER EN TD *****
nx = 1
nu = 1
ny = 1
# ******************************
positionKF = kf.KalmanFilter(nx, nu, ny) 


# matrices représentant l'équation d'état et de mesure    
# ******* A MODIFIER EN TD *****
Ak = np.array( [[1.0]] )
Bk = np.array( [[Te]] )
# ******************************
positionKF.setStateEquation(Ak, Bk)      

# matrice représentant l'équation de mesure
# ******* A MODIFIER EN TD *****
Ck = np.array( [[1.0]]) 
# ******************************
positionKF.setCk(Ck)


# matrice de covariance du bruit d'état
# ******* A FAIRE VARIER EN TD *****
Qk = np.array([[stdDevVel**2]])
# ******************************
positionKF.setQk(Qk)

# matrice de covariance du bruit de mesure
# ******* A FAIRE VARIER EN TD *****
Rk = np.array([[stdDevPos**2]])
# ******************************
positionKF.setRk(Rk)

# matrice de covariance associée à la position initiale supposée a priori
stdDevPos0 = 0.5
# ******* A FAIRE VARIER EN TD *****
P0 = np.array([[(stdDevPos0)**2]])
# ******************************
positionKF.initFilter(x0Est , P0) 

# structures de données pour sauvegarder les résultats
# ------------------------------------------------------
# position 
filteredPosition = np.zeros_like(trajectoireVraie.position)
filteredPosition[0] = x0Est
# écart stype sur la position
filteredPosStdDev = np.zeros_like(trajectoireVraie.position)
filteredPosStdDev[0] = np.sqrt(P0[0])

uk = np.zeros(1)
yk = np.zeros(1)


# calculs tout le long de la trajectoire
for i in range(1, nbPoints):

    uk[0] = mesuresTrajectoire.vitesse[i-1]   
    positionKF.predict(uk)
    
    if not(np.isnan(mesuresTrajectoire.position[i])):
        #print 'update'
        yk[0] = mesuresTrajectoire.position[i]
        positionKF.update(yk)
        
    filteredPosition[i] = positionKF.xk
    filteredPosStdDev[i] = np.sqrt(positionKF.Pk[0])



# affichages
# --------------

# position vraie, mesure et position filtrée
plt.figure()
plt.plot(temps, filteredPosition, 'b-')
plt.plot(temps, mesuresTrajectoire.position, 'kx-')
plt.plot(temps, trajectoireVraie.position, 'r-')
#plt.plot(temps, integratedPositionFromVelMeas, 'g-')
plt.xlabel('t (s)')
plt.ylabel('position (m)')
plt.grid()

# écart en position (erreur d'estimation)
plt.figure()
plt.plot(temps,filteredPosition - trajectoireVraie.position, 'b-')
#plt.plot(temps, integratedPositionFromVelMeas - trajectoireVraie.position, 'g-')
plt.plot(temps, 3.0*filteredPosStdDev, 'k-')
plt.plot(temps, -3.0*filteredPosStdDev, 'k-')
plt.xlabel('t (s)')
plt.ylabel('erreur en position (m)')
plt.grid()

plt.show()



