import numpy as np
import pandas as pd
import cv2
import os

def extract_features(chemin_dossier, label):
    resultats = []

    for name in os.listdir(chemin_dossier):
       chemin_complet = os.path.join(chemin_dossier, name)
       if name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png"):
            image = cv2.imread(chemin_complet)

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            borne_basse = np.array([5, 50, 50])
            borne_haute = np.array([30, 255, 255])
            masque = cv2.inRange(hsv, borne_basse, borne_haute)

            # X1 : pct_rouille
            pct_rouille = (np.sum(masque)/255) / masque.size

            gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gris, cv2.CV_64F, dx=1, dy=0)
            sobel_y = cv2.Sobel(gris, cv2.CV_64F, dx=0, dy=1)
            magnitude = np.hypot(sobel_x, sobel_y)
            
            # X2 : rugosite 
            rugosite = np.var(magnitude)

            seuil_min = 50
            seuil_max = 150

            # X3 : Ma variable : nombre de bords distincts 
            canny = cv2.Canny(gris, seuil_min, seuil_max)
            nombre_bords = (np.sum(canny)/255) / canny.size

            resultats.append({'id':chemin_complet, 'pct_rouille': pct_rouille, 'rugosite':rugosite, 'canny':nombre_bords, 'label':label})

    return resultats

malades = extract_features("dataset/malades", 1)
saines = extract_features("dataset/saines", 0)
liste = malades + saines
df = pd.DataFrame(liste)
df.to_csv("features.csv", index=False)
df = pd.read_csv("features.csv")
print(df.drop(columns=['id']).corr()['label'])