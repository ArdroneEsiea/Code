# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:43:28 2015

@author: Hoagie
"""

import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches


class Goal:
    def __init__(self, xr, yr, ka):
        self.xr = xr
        self.yr = yr
        self.ka = ka
    
    def evalAttractionPotential(self,x ,y):
        # !!!!!!!!!!!
        # A COMPLETER EN TD        
        U = (0.5)*ka*((x-xr)**2 + (y-yr)**2)
        # !!!!!!!!!!!
        return U

    def evalAttractionForce(self, x ,y):
        # !!!!!!!!!!!
        # A COMPLETER EN TD
        Ux = (0.5)*ka*((x-xr)**2
        Fx = -np.gradient(x)
        Fy = -np.gradient(y)
        # !!!!!!!!!!!
        return Fx, Fy


    def plotAttractionPotential(self, figNo=1, xmin=-3, xmax=3, ymin=-3, ymax=3, step=0.2):
        X,Y = np.meshgrid( np.arange(xmin,xmax,step), np.arange(ymin,ymax,step) )
        U = self.evalAttractionPotential(X, Y)
        # plot potential
        fig = plt.figure(figNo)
        ax = fig.add_subplot(111, projection='3d', aspect='equal')
        ax.plot_surface(X, Y, U, rstride=1, cstride=1, linewidth=0.5, antialiased=True, cmap=cm.coolwarm)
        ax.grid()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Potential')
        
    def plotAttractionForce(self, figNo=1, xmin=-3, xmax=3, ymin=-3, ymax=3, step=0.2):
        X,Y = np.meshgrid( np.arange(xmin,xmax,step), np.arange(ymin,ymax,step) )
        Fax, Fay = self.evalAttractionForce(X, Y)
        fig = plt.figure(figNo)
        ax = fig.add_subplot(111, autoscale_on=True, aspect='equal')
        plt.quiver(X,Y,Fax,Fay, color='g')
        ax.grid()
        plt.plot(self.xr, self.yr, 'o', color='g', lw=2, markersize=10)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')


class Obstacle:
    def __init__(self, xo, yo, do, sigmaX, sigmaY):
        self.xo = xo
        self.yo = yo
        self.do = do
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY

    def evalRepulsionGaussianPotential(self, x, y):
        
        # !!!!!!!!!!!
        # A COMPLETER EN TD        
        xo = 1.5
        yo = 0
        do = 5
        sigmaX = 0.5
        sigmaY = 0.5
        U = (1.0/(2.0*math.pi*sigmaX*sigmaY))*math.exp(-(0.5)*(((x-xo)/sigmaX)+((y-yo)/sigmaY)))
        print(U)
        
        # !!!!!!!!!!!
        return U        

    
    def evalRepulsionGaussianForce(self, x, y):
        # !!!!!!!!!!!
        # A COMPLETER EN TD        
        Fx = -np.gradient(x)
        Fy = -np.gradient(y)
        # !!!!!!!!!!!    
        return Fx, Fy   
        
        
    def plotRepulsionGaussianPotential(self, figNo=1, xmin=-3, xmax=3, ymin=-3, ymax=3, step=0.2):
        X,Y = np.meshgrid( np.arange(xmin,xmax,step), np.arange(ymin,ymax,step) )
        Ur = np.zeros_like(X)
        for i in range(0,np.shape(X)[0]):
            for j in range(0, np.shape(X)[1]):
                Ur[i,j] = self.evalRepulsionGaussianPotential(X[i,j], Y[i,j])
        # plot potential
        fig = plt.figure(figNo)
        ax = fig.add_subplot(111, projection='3d', aspect='equal')
        ax.plot_surface(X, Y, Ur, rstride=1, cstride=1, linewidth=0.5, antialiased=True, cmap=cm.coolwarm)
        ax.grid()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Potential')
  
        

    def plotRepulsionGaussianForce(self, figNo=1, xmin=-3, xmax=3, ymin=-3, ymax=3, step=0.2):
        X,Y = np.meshgrid( np.arange(xmin,xmax,step), np.arange(ymin,ymax,step) )
        Frx = np.zeros_like(X)
        Fry = np.zeros_like(Y)
        for i in range(0,np.shape(X)[0]):
            for j in range(0, np.shape(X)[1]):
                Frx[i,j], Fry[i,j] = self.evalRepulsionGaussianForce(X[i,j], Y[i,j])
        fig = plt.figure(figNo)
        ax = fig.add_subplot(111, autoscale_on=True, aspect='equal')
        plt.quiver(X,Y,Frx,Fry, color='r')
        ax.grid()
        plt.plot(self.xo, self.yo, 'o', color='r', lw=2, markersize=10)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')



if __name__=='__main__':
    
    xr = 0
    yr = -2.5
    ka = 0.04
    goal = Goal(xr, yr, ka)
    
    xo1 = -1.5
    yo1 = 1
    do1 = 5
    sigmaX1 = 0.5
    sigmaY1 = 0.5
    obstacle1 = Obstacle(xo1, yo1, do1, sigmaX1, sigmaY1)
    
    xo2 = 1.5
    yo2 = 0
    do2 = 5
    sigmaX2 = 0.5
    sigmaY2 = 0.5
    obstacle2 = Obstacle(xo2, yo2, do2, sigmaX2, sigmaY2)
    
    
    plt.close('all')    
    
    goal.plotAttractionPotential(10)    
    goal.plotAttractionForce(11)
    
    obstacle1.plotRepulsionGaussianPotential(20)
    obstacle1.plotRepulsionGaussianForce(21)    
    
    obstacle2.plotRepulsionGaussianPotential(30)
    obstacle2.plotRepulsionGaussianForce(31)
    
    plt.show()