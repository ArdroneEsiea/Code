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
boolean = False
seuil = 0.1

# initialisation du noeud
rospy.init_node('commande_z', anonymous=True)

# declaration d'un publisher sur le topic de commande
pubCommande = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


# fonction de lecture de la mesure et d'envoi de la commande
def getAltitude(data):
	global altitudeRef ,boolean	
	# lecture donnee altitude (en m)
	altitude = data.altd  / 1000.0
	# affichage sur la console (pour info)
	rospy.loginfo(rospy.get_caller_id() + "altitude (m) = %f", data.altd)

	# calcul de la commande
	# *********** A MODIFIER EN TP *************

	commande.linear.z = 1.5*(altitudeRef - altitude)
	# ****************************************

	if abs(altitude - altitudeRef) < seuil :
		#time.sleep(2)		
		if altitudeRef  <  altitudeFinal :
			altitudeRef = altitudeRef + 0.300
		
		#if altitudeRef  >  altitudeFinal and altitudeRef > abs(altitudeFinal -0.5):
		#	altitudeRef = altitudeRef - 0.300	
	

	
	# publication de la commande
	pubCommande.publish(commande)	


# declaration d'un subscriber : appelle getAltitude a chq arrivee d'une donnee sur le topic Navdata
rospy.Subscriber("/ardrone/navdata", Navdata, getAltitude)



# fonction main executee en boucle 
if __name__ == '__main__':
	rospy.spin()

