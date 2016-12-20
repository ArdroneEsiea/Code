# -*- coding: utf-8 -*-
"""
Created on Fri Jul 03 2015

Potential Fields navigtion


@author: S. Bertrand
"""

import math
import Robot as rob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Timer as tmr
import Potentials as pot
import matplotlib.patches as patches

# robot
x0 = -4
y0 = 1.2
theta0 = 0.0
robot = rob.Robot(x0, y0, theta0)

# position control loop timer
positionCtrlPeriod = 0.2
timerPositionCtrl = tmr.Timer(positionCtrlPeriod)

# orientation control loop timer
orientationCtrlPeriod = 0.05
timerOrientationCtrl = tmr.Timer(orientationCtrlPeriod)


# goal way point
xr = 4.0
yr = 1.0
ka = 0.04
goal = pot.Goal(xr, yr, ka)

WPlist = [ [goal.xr, goal.yr] ]  
#threshold for change to next WP
epsilonWP = 0.2
# init WPManager
WPManager = rob.WPManager(WPlist, epsilonWP)

# obstacle 1
#!!!!!!!!!!
# A COMPLETER EN TD
#!!!!!!!!!!

# obstacle 2
#!!!!!!!!!!
# A COMPLETER EN TD
#!!!!!!!!!!


# duration of scenario and time step for numerical integration
t0 = 0.0
tf = 30.0
dt = 0.01
simu = rob.RobotSimulation(robot, t0, tf, dt)

#class Obstacle:
#    def _init_(self,xo,yo,do,sigmaX,signaY):
#        self.xo = xo
#        self.yo = yo
#        self.do = do 
#        self.sigmaX = sigmaX
#        self.sigmaY = sigmaY
#        
#    def evalRepulsionGaussianPtential(self,x,y):
#        U = 0.0
#        return U 
#    
#    def evalRepulsionGaussianForce(self,x,y):
#        Fx = 0
#        Fy = 0

# initialize control inputs
Vr = 0.0
thetar = 0.0
omegar = 0.0


# loop on simulation time
for t in simu.t: 
   

    # position control loop
    if timerPositionCtrl.isEllapsed(t):

        #!!!!!!!!!!
        # A COMPLETER EN TD
        # calcul de Vr
        Vr = 0.0
        #!!!!!!!!!!
        
        #!!!!!!!!!!
        # A COMPLETER EN TD        
        # reference orientation
        thetar = 0.0
        #!!!!!!!!!!
        
        
    # orientation control loop
    if timerOrientationCtrl.isEllapsed(t):

        #!!!!!!!!!!
        # A COMPLETER EN TD        
        # angular velocity control input        
        omegar = 0.0
        #!!!!!!!!!!
    
    
    # assign control inputs to robot
    robot.setV(Vr)
    robot.setOmega(omegar)    
    
    # integrate motion
    robot.integrateMotion(dt)

    # store data to be plotted   
    simu.addData(robot, WPManager, Vr, thetar, omegar)
    
# end of loop on simulation time


# close all figures
plt.close("all")

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

'''
# !!!!!!!!!!!!!!!!!!!!
# A DECOMMENTER EN TD POUR AFFICHER LES OBSTACLES
ax.add_patch(patches.Circle((obstacle1.xo, obstacle1.yo), obstacle1.do, alpha = 0.08, color='red'))#, color='none', linewidth=1, linecolor='blue'))
plt.plot(obstacle1.xo, obstacle1.yo, 'o', color='r', lw=2, markersize=10)
ax.add_patch(patches.Circle((obstacle2.xo, obstacle2.yo), obstacle2.do, alpha = 0.08, color='red'))#, color='none', linewidth=1, linecolor='blue'))
plt.plot(obstacle2.xo, obstacle2.yo, 'o', color='r', lw=2, markersize=10)
# !!!!!!!!!!!!!!!!!!!!
'''


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

