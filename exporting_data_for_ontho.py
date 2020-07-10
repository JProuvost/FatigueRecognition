# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 14:44:44 2020

@author: Jean
"""

import numpy as np
import cv2
import os
import urllib.request as urlreq
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
import joblib
from datetime import datetime

os.chdir(os.getcwd())

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

LBFmodel_url= "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

LBFmodel = "lbfmodel.yaml"

urlreq.urlretrieve(LBFmodel_url, LBFmodel)
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])


hogify = HogTransformer(
    pixels_per_cell=(8, 8),
    cells_per_block=(2,2),
    orientations=9,
    block_norm='L2-Hys'
)


sgd_clf = joblib.load("svm_for_eyes.joblib")

def frame_extraction_v():
    print("ui3")
    cap= cv2.VideoCapture(0)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if ret == False:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imwrite('frame'+str(i)+'.jpg',frame)
        i+=1
    cap.release()
    cv2.destroyAllWindows()
    frame_nb=i
    return frame_nb

def delete_frame_file(a):
    i=0
    while (i<a):
        if os.path.exists("frame"+str(i)+".jpg") :
            os.remove("frame"+str(i)+".jpg")
        i = i + 1


def save_data_csv_format(facial_measures):

    key_generator = datetime.now()
    key_formated = key_generator.strftime("%d-%m-%Y-%H-%M-%S")
    np.savetxt('data_to_export/data_'+key_formated+'.csv', facial_measures, delimiter=',')

def data_obtention(frame_nb):
    print("ui2")
    #création de l'instance du détecteur de facial landmark sur le modèle ci-dessus ainsi que l'array pour les données obtenues
    print("ui4")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    print("ui5")
    landmark_detector = cv2.face.createFacemarkLBF()
    print("ui6")
    landmark_detector.loadModel(LBFmodel)
    print("ui7")

    facial_measures = np.zeros((frame_nb,5), dtype="int")
    i=0
    while (i<frame_nb):
        # Read the input image
        img = cv2.imread('frame'+str(i)+'.jpg')
        # Convert into grayscale
        print(i)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("ui8")
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        print("ui9")
        # Draw rectangle around the faces
        if(len(faces)!=0):
            print("ui10")
            _, landmarks = landmark_detector.fit(gray, faces)
            for landmark in landmarks:
                print("ui11")
                eye_l_v_1=np.linalg.norm(landmark[0][38]-landmark[0][42])
                eye_l_v_2=np.linalg.norm(landmark[0][39]-landmark[0][41])
                eye_1_h=np.linalg.norm(landmark[0][37]-landmark[0][40])
                eye_1_ear=(eye_l_v_1+eye_l_v_2)/2*eye_1_h

                print("ui12")
                eye_2_v_1=np.linalg.norm(landmark[0][44]-landmark[0][48])
                eye_2_v_2=np.linalg.norm(landmark[0][45]-landmark[0][47])
                eye_2_h=np.linalg.norm(landmark[0][43]-landmark[0][46])
                eye_2_ear=(eye_2_v_1+eye_2_v_2)/2*eye_2_h
                mouth_openess=np.linalg.norm(landmark[0][52]-landmark[0][63])
                print("ui13")
                img_eye_left= gray[int(landmark[0][20][1]):int(landmark[0][42][1]-5) , int(landmark[0][18][0]):int(landmark[0][22][0])].copy()
                img_eye_right= gray[int(landmark[0][48][1]-5):int(landmark[0][20][1]) , int(landmark[0][18][0]):int(landmark[0][22][0])].copy()
                print("ui14")
                img_eye_left=np.resize(img_eye_left,(1,54,54))
                img_eye_right=np.resize(img_eye_left,(1,54,54))
                print("ui15")
                img_eye_left_hog=hogify.fit_transform(img_eye_left)
                img_eye_right_hog=hogify.fit_transform(img_eye_right)
                print("ui16")
                left_eye_predict=sgd_clf.predict(img_eye_left_hog)
                right_eye_predict=sgd_clf.predict(img_eye_right_hog)

                facial_measures[i]=[eye_1_ear,eye_2_ear,mouth_openess,left_eye_predict,right_eye_predict]
        i+=1
    return facial_measures

print("ui")

a=frame_extraction_v()
print(a)
facial_measures=data_obtention(a)
save_data_csv_format(facial_measures)
delete_frame_file(a)
