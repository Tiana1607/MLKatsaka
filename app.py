import streamlit as st
import pickle
import cv2
import numpy as np
import os
from PIL import Image
from decision_tree import predict_forest

with open('modele.pkl', 'rb') as f:
    modele = pickle.load(f)

def extraire_features_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    borne_min = np.array([5, 50, 50])
    borne_max = np.array([30, 255, 255])
    masque_hsv = cv2.inRange(hsv, borne_min, borne_max)
    pct_rouille = (np.sum(masque_hsv) / 255) / masque_hsv.size

    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gris, cv2.CV_64F, dx=1, dy=0)
    sobel_y = cv2.Sobel(gris, cv2.CV_64F, dx=0, dy=1)
    magnitude = np.hypot(sobel_x, sobel_y)
    rugosite = np.var(magnitude)

    seuil_min = 50
    seuil_max = 150
    canny = cv2.Canny(gris, seuil_min, seuil_max)
    nombre_bords = (np.sum(canny)/255) / canny.size

    return pct_rouille, rugosite, nombre_bords


st.title("Diagnostic Rouille Polysora")

st.write("Application d'aide au diagnostic pour les techniciens agricoles malgaches.")

fichier = st.file_uploader("Téléversez une image de feuille de maïs", type=["jpg", "jpeg", "png"])

os.makedirs("uploads", exist_ok=True)

st.subheader("Galerie d'historique des détections")
cols = st.columns(3)

for i, nom_fichier in enumerate(os.listdir("uploads")):
    chemin = os.path.join("uploads", nom_fichier)
    image = Image.open(chemin)
    image_carree = image.resize((300, 300))
    cols[i%3].image(image_carree, width='stretch')   
    cols[i%3].write(nom_fichier)

if fichier is not None :
    image = Image.open(fichier)
    st.image(image)
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    pct_rouille, rugosite, nombre_bords = extraire_features_image(image_bgr)
    features = np.array([pct_rouille, rugosite, nombre_bords])
    prediction = predict_forest(modele, features)
    
    if prediction == 0:
        st.success("Feuille saine !")
    else :
        st.error("ATTENTION : Feuille Malade(Rouille Detected) !")

    chemin = os.path.join("uploads", fichier.name)
    image.save(chemin)
