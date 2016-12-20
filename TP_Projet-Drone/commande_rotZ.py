#!/usr/bin/env python

# *************************
#     MINEURE ROBOTIQUE 5A
#          ESIEA - ONERA
# *************************

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata # for receiving navdata feedback
#import time

# variables globales
altitudeRef = 0.80 # altitude ref en m
altitudeFinal = 1.4
commande = Twist()
k = 0.5
variable = 1
seuil = 0.1
x_max = 806
x_min = 146
y_max = 845
y_min = 233




# initialisation du noeud
rospy.init_node('commande_rotZ', anonymous=True)

# declaration d'un publisher sur le topic de commande
pubCommande = rospy.Publisher('/cmd_vel', Twist, queue_size=10)



# fonction de lecture de la mesure et d'envoi de la commande
def getAltitude(data):
    global altitudeRef,variable,tagsC
     
    tagsC = data.tags_count
    rotationRef = data.tags_orientation

    # lecture donnee altitude (en m)
    altitude = data.altd  / 1000.0
    # affichage sur la console (pour info)
    rospy.loginfo(rospy.get_caller_id() + "altitude (m) = %f", data.altd)
    rospy.loginfo("angle =%f\n", data.rotZ)
    rospy.loginfo("varaible =%f\n", variable)
    # calcul de la commande
    # *********** A MODIFIER EN TP *************

    commande.linear.z = 0.5*(altitudeRef - altitude)
    if abs(altitude - altitudeRef) < seuil :
                #time.sleep(2)        
            if altitudeRef  <  altitudeFinal :
                altitudeRef = altitudeRef + 0.300

    if(tagsC==1):
        xtag = data.tags_xc
        ytag = data.tags_yc
        rotz = data.rotZ
	#rospy.loginfo("x =%f\n", data.tags_xc)
	#rospy.loginfo("y =%f\n", data.tags_yc)

        if(xtag >= 0.8*x_min) and (xtag <= 1.2*x_max) and (ytag >= 0.8*y_min) and (ytag <= 1.2*y_max):


            if(variable == 1):
		rospy.loginfo("tag detecte")
                rotationRef = rotationRef + 45
                commande.angular.z = -0.5
                if(rotationRef == 360):
                    variable = 0
	    
            
            if(variable == 0): 
                rotationRef = rotationRef - 45 
                commande.angular.z = 0.5
                if(rotationRef == 0):
                    variable = 1

            commande.linear.x = 0.5*(data.tags_xc[0] - 400)
            commande.linear.y = 0.5*(data.tags_yc[0] - 400)
            
        else:
            commande.linear.z = 0.5*(altitudeRef - altitude)



        # ****************************************
        
    

    
# publication de la commande
pubCommande.publish(commande)    


# declaration d'un subscriber : appelle getAltitude a chq arrivee d'une donnee sur le topic Navdata
rospy.Subscriber("/ardrone/navdata", Navdata, getAltitude)



# fonction main executee en boucle 
if __name__ == '__main__':
    rospy.spin()

