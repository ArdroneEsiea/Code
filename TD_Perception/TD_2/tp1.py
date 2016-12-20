# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:15:21 2015

@author: Aurélien Plyer
"""
# importation des modules qu'on va utiliser : 
import cv2                # OpenCV
import pylab as pl        # pylab pour l'affichage
import numpy as np        # numpy pour les calculs sur matrices
from scipy import ndimage # scipy pour les convolution
from pylab import figure, imshow

#%%

class Matcheur:
    # attribus de la classe Matcheur
    pts0 = None
    pts1 = None
    I0 = None
    I1 = None
    inlier_mask = None
    seuil_ransac = 20

    @staticmethod
    def InitStatic():
        Matcheur.constructeurs = {'akaze': cv2.AKAZE_create,
             'agast': cv2.AgastFeatureDetector_create,
             'brisk': cv2.BRISK_create,
             'fast' : cv2.FastFeatureDetector_create,
             'gftt' : cv2.GFTTDetector_create,
             'kaze' : cv2.KAZE_create,
             'mser' : cv2.MSER_create,
             'orb'  : cv2.ORB_create,
             'blob' : cv2.SimpleBlobDetector_create,
             'sgbm' : cv2.StereoSGBM_create,
             'bm'   : cv2.StereoBM_create,
             'matcher': cv2.DescriptorMatcher_create }
        Matcheur.descripteurs = []
        Matcheur.detecteurs = []
        fast = cv2.FastFeatureDetector_create()
        I = np.random.randint(100,size=512*512).reshape([512,512]).astype(np.uint8)
        I[10:20,10:20]=0
        kp= fast.detect(I)
        for key in Matcheur.constructeurs.keys():
            try:
                construct = Matcheur.constructeurs[key]()
                tmp_k = construct.detect(I)
#                if type(tmp_k[0]) is not type(cv2.KeyPoint()):
#                    raise Exception('resultats pas keypoiny!')
            except:
                a = 0
                #print key, ' n\'est pas un detecteur'
            else:
                Matcheur.detecteurs.append(key)
            try:
                construct = Matcheur.constructeurs[key]()
                tmp_k, tmp_desc = construct.compute(I,kp)
#                if type(tmp_k[0]) is not type(cv2.KeyPoint()):
#                    raise Exception('resultats pas keypoiny!')
            except:
                 a = 0
                 #print key, ' n\'est pas un descripteur'
            else:
                Matcheur.descripteurs.append(key)
        
    # Methodes statiques de la classe
    @staticmethod
    def GetAviableDetector():
        return Matcheur.detecteurs
    @staticmethod
    def GetAviableDescriptor():
        return Matcheur.descripteurs
    @staticmethod
    def getPointsFromMatch(match, kp0, kp1):
        pts0 = np.array([kp0[m.queryIdx].pt for m in match])
        pts1 = np.array([kp1[m.trainIdx].pt for m in match])
        return pts0, pts1
        
    
    def __init__(self,detecteur = 'fast', descripteur = 'brisk'):
        '''
             Constructeur de la classe, on y construit les classes membres que sont
             le détecteur, le descripteur ainsi que la classe d'affectation grâce aux
             vecteurs d'introspection précédement calculé  
        '''
        if self.detecteurs.count(detecteur) == 0 or self.descripteurs.count(descripteur) == 0:
                raise TypeError('descripteur ou detecteur inconnue')
        self.detecteur_name = detecteur
        self.descripteur_name = descripteur
        self.detector = self.constructeurs[self.detecteur_name]()
        self.descriptor = self.constructeurs[self.descripteur_name]()
        self.matcheur = cv2.BFMatcher(normType = self.descriptor.defaultNorm(), crossCheck = True )
    
    def match(self, I0, I1):
        '''
            Methode effectuant la mise en correspondance
        '''
        self.I0 = I0
        self.I1 = I1
        
        # Detecter les points dans I0 et I1
        self.kp0 = self.detector.detect(I0)
        self.kp1 = self.detector.detect(I1)
        
        # Calculer les descripteurs dans I0 et I1
        self.kp0, self.desc0 = self.descriptor.compute(I0,self.kp0)
        self.kp1, self.desc1 = self.descriptor.compute(I1,self.kp1)
        
        # Calcul de la mise en correspondance entre les descripteurs
        self.matches = self.matcheur.match(self.desc0, self.desc1)
        
        ###################################################################
        # Question : utiliser une lamda expression et la fonction sorted
        # pour trier les elements de self.matches en fonction de leurs
        # attribut 'distance'
        ####################################################################
        # Reponse : 
        self.matches = sorted(self.matches, key = lambda x: x.distance)
        ####################################################################
        
        self.pts0, self.pts1 = Matcheur.getPointsFromMatch(self.matches, self.kp0, self.kp1)
        return self.pts0, self.pts1
        
    def affine_match(self,motion_model = 'homography'):
        def stereoModel(pts0, pts1, compat1, compat2):
            error = np.abs(pts1[:,1]-pts0[:,1])
            mask = np.ones([pts1.shape[0],1])
            mask[error > 1] = 0
            return None, mask
        '''
            Fonction rafinant la mise en correspondance en utilisant un 
            model géométrique à priorit, les models sont :
             'homography' : model projectif correspondant au déplacement d'un plan
             'fundamental' : model correspondant a la contrainte epipolaire entre
                             deux vues
             'stereo' : n'estime pas un model, mais renvois que les points 
                        verifiant la contrainte stereo
            affine_match(self,motion_model = 'homography') -> flag, pts0, pts1, M
            flag = booleen si le model est valide
            pts0, pts1 = points inliers dans les images 0 et 1
            M = matrice du model geometrique estime
        '''
        if self.pts0 is None:
            raise RuntimeError('Veuillez appeler match(I0,I1) avant de rafiner')    
        if motion_model is 'homography':
            findModel = cv2.findHomography
        elif motion_model is 'fundamental':
            findModel = cv2.findFundamentalMat
        elif motion_model is 'stereo':
            findModel = stereoModel
        else:
            raise TypeError('model '+str(motion_model)+' inconus')
        
        pts0, pts1 = Matcheur.getPointsFromMatch(self.matches, self.kp0, self.kp1)
        M, mask = findModel(pts0, pts1, cv2.RANSAC, 5.0)
        ####################################################################
        # Question : compter le nombre d'inlier / outlier du model
        ####################################################################
        # Reponse : 
        inlier = np.sum(mask == 1)
        outlier = np.sum(mask == 0)
        ####################################################################
        if inlier > self.seuil_ransac:
            is_good = True
            #####################################################################
            #  Question : selectionner les points inliers en utilisant la 
            #             methode ravel de mask pour le serialiser
            ####################################################################
            # Reponse : 
            pts0 = pts0[mask.ravel() == 1]
            pts1 = pts1[mask.ravel() == 1]
            ####################################################################
            self.inlier_mask = mask
            self.pts0 = pts0
            self.pts1 = pts1
            return is_good, self.pts0, self.pts1, M
        else:
            self.inlier_mask = mask
            return False, None, None, None

    def print_parameters(self):
        print 'detecteur        : ', self.detecteur_name
        print 'descripteur      : ', self.descripteur_name
        print 'seuil good match : ', self.seuil_match
    def get_valid_kp(self):
        if self.inlier_mask is not None :
            valid = [ m for idx, m in enumerate(self.matches) if self.inlier_mask[idx] == 1]
        else:
            valid = self.matches
        kp0 = [self.kp0[m.queryIdx] for m in valid]
        kp1 = [self.kp1[m.trainIdx] for m in valid]
        pts0 = np.array([self.kp0[m.queryIdx].pt for m in valid])
        pts1 = np.array([self.kp1[m.trainIdx].pt for m in valid])
        desc0 = np.array([self.desc0[m.queryIdx] for m in valid])
        desc1 = np.array([self.desc1[m.trainIdx] for m in valid])
        return pts0, pts1, kp0, kp1, desc0, desc1
    def show_current(self):
        '''
            Fonction affichant la mise en correspondance actuelle
        '''
        if self.pts0 is not None:
            print 'nombre de match : ', len(self.matches)
            if self.inlier_mask is not None :
                valid = [ m for idx, m in enumerate(self.matches) if self.inlier_mask[idx] == 1]
            else:
                valid = self.matches
            img = cv2.drawMatches(self.I0,self.kp0,self.I1,self.kp1,valid, None)
            figure()            
            imshow(img)
        else:
            print 'pas de match!'
            
#%%
