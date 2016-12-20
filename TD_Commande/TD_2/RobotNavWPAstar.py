# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 18:05:06 2015
@author: S. Bertrand
"""

import math
import Robot as rob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Timer as tmr
import AStar



# close all figures
plt.close("all")

dimX = 20
dimY = 10
    
occupancyGrid = np.zeros([dimX, dimY])
occupancyGrid[2:4,2:4]=1
occupancyGrid[6,1:3]=1
occupancyGrid[4:7,3]=1
occupancyGrid[8:10,2:6]=1
occupancyGrid[4:6,6:8]=1
occupancyGrid[6,5:7]=1
occupancyGrid[10:12,7:10]=1
occupancyGrid[11,0:3]=1
occupancyGrid[11:15,4:6]=1
occupancyGrid[13:15,2:5]=1
occupancyGrid[15,3:5]=1
occupancyGrid[16:18,3]=1
occupancyGrid[18:20,5]=1
occupancyGrid[14,7:9]=1
occupancyGrid[16,7:10]=1


adjacency = 8

carte = AStar.Map(dimX,dimY, adjacency)
carte.initCoordinates()
carte.loadOccupancy(occupancyGrid)
carte.generateGraph()
carte.plot(10)


  
epsilon = 1

print "A* algorithm running ..."
closedList, successFlag = carte.AStarFindPath(0,199, epsilon)
#print closedList
if (successFlag==True):
    print "  - A* terminated with success: path found"

    path = carte.builtPath(closedList)

    carte.plotPathOnMap(path, 20)

i = 0

path.reverse()

WPlist =[]

# parcourir la liste path

for noNoeud in path:
    
    
    # recuperer les coordoonnées x et y du noeud
    xNoeud = carte.graph.listOfNodes[noNoeud].x
    yNoeud = carte.graph.listOfNodes[noNoeud].y
    
    # les ajouter à WPList
    WPlist.append(  [xNoeud, yNoeud] )



# robot
x0 = 0.0
y0 = 0.0
theta0 = 0.0
robot = rob.Robot(x0, y0, theta0)
k1 = 4.5
k2 = 12

# position control loop timer
positionCtrlPeriod = 0.2
timerPositionCtrl = tmr.Timer(positionCtrlPeriod)

# orientation control loop timer
orientationCtrlPeriod = 0.05
timerOrientationCtrl = tmr.Timer(orientationCtrlPeriod)


# list of way points: list of [x coord, y coord]
#WPlist = [ [2.0,2.0] ]  
#Test de Passage d'un point de passage à l'autre
#WPlist = [ [2.0,2.0] , [2.0, -2.0] ,[-2.0, -2.0],[-2.0, 2.0],[2.0, 2.0],[0.0, 0.0]]
#threshold for change to next WP
epsilonWP = 1
# init WPManager
WPManager = rob.WPManager(WPlist, epsilonWP)



# duration of scenario and time step for numerical integration
t0 = 0.0
tf = 25.0
dt = 0.01
simu = rob.RobotSimulation(robot, t0, tf, dt)


# initialize control inputs
#Valeurs à calculer
Vr = 0.0
thetar = 0.0
omegar = 0.0
epsilon = 0.5



# loop on simulation time
for t in simu.t: 
   
   
    # WP navigation: switching condition to next WP of the list

        # !!!!! 
        # A COMPLETER EN TD
        # !!!!! 
    
    if np.sqrt((WPManager.xr - robot.x)**2+ (WPManager.yr - robot.y)**2)<= epsilon:
                        
            WPManager.switchToNextWP()
    
    # position control loop
    if timerPositionCtrl.isEllapsed(t):
        
        
        # !!!!! 
        # A COMPLETER EN TD : calcul de Vr
        Vr = k1 * (np.sqrt((WPManager.xr - robot.x)**2+ (WPManager.yr - robot.y)**2)) 
        # !!!!! 

        # reference orientation
        # !!!!! 
        # A COMPLETER EN TD : calcul de thetar
        thetar = np.arctan2(WPManager.yr-robot.y,WPManager.xr-robot.x)
        # !!!!! 
        
        # !!!!! 
        # A COMPRENDRE EN TD : quelle est l'utilité des deux lignes de code suivantes ?
        #     (à conserver après le calcul de thetar)
        if math.fabs(robot.theta-thetar)>math.pi:
            thetar = thetar + math.copysign(2*math.pi,robot.theta)        
        # !!!!! 
        
        
    # orientation control loop
    if timerOrientationCtrl.isEllapsed(t):
        # angular velocity control input        
        # !!!!! 
        # A COMPLETER EN TD : calcul de omegar
        omegar = k2 * (thetar- robot.theta)
        # !!!!! 

    
    # apply control inputs to robot
    robot.setV(Vr)
    robot.setOmega(omegar)    
    
    # integrate motion
    robot.integrateMotion(dt)

    # store data to be plotted   
    simu.addData(robot, WPManager, Vr, thetar, omegar)
    
# end of loop on simulation time



# generate plots
simu.plotXY(1)
simu.plotXYTheta(2)
simu.plotVOmega(3)

#simu.runAnimation(WPManager.epsilonWP, 5)

# show plots
plt.show()





# Animation *********************************
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
ax.grid()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

robotBody, = ax.plot([], [], 'o-', lw=2)
robotDirection, = ax.plot([], [], '-', lw=1, color='k')
wayPoint, = ax.plot([], [], 'o-', lw=2, color='b')
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
WPArea, = ax.plot([], [], ':', lw=1, color='b')

thetaWPArea = np.arange(0.0,2.0*math.pi+2*math.pi/30.0, 2.0*math.pi/30.0)
xWPArea = WPManager.epsilonWP*np.cos(thetaWPArea)
yWPArea = WPManager.epsilonWP*np.sin(thetaWPArea)

def initAnimation():
    robotDirection.set_data([], [])
    robotBody.set_data([], [])
    wayPoint.set_data([], [])
    WPArea.set_data([], [])
    robotBody.set_color('r')
    robotBody.set_markersize(20)    
    time_text.set_text('')
    return robotBody,robotDirection, wayPoint, time_text, WPArea  
    
def animate(i):  
    robotBody.set_data(simu.x[i], simu.y[i])          
    wayPoint.set_data(simu.xr[i], simu.yr[i])
    WPArea.set_data(simu.xr[i]+xWPArea.transpose(), simu.yr[i]+yWPArea.transpose())    
    thisx = [simu.x[i], simu.x[i] + 0.5*math.cos(simu.theta[i])]
    thisy = [simu.y[i], simu.y[i] + 0.5*math.sin(simu.theta[i])]
    robotDirection.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*simu.dt))
    return robotBody,robotDirection, wayPoint, time_text, WPArea

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(simu.t)),
    interval=4, blit=True, init_func=initAnimation, repeat=False)
#interval=25

#ani.save('robot.mp4', fps=15)

