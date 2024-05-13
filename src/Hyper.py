import streamlit as st
import cv2
import numpy as np
import spectral


def cargar_hyperimagenes():
    imagen = st.file_uploader("Selecciona una imagen hiperespectral", type=["bil", "hdr"], accept_multiple_files=True)
    return imagen

imagen = cargar_hyperimagenes()

nombre_hdr = "./hojas/ES 2 wet 110723.bil.hdr"

# Cargar la imagen hiperespectral
img = spectral.envi.open(nombre_hdr)

capa1= st.slider('Número de capa 1', 0, 299, 155)
color1=st.slider('Colormap 1', 0, 20, 0)
primera_capa = img.read_band(capa1)
primera_capa = cv2.normalize(primera_capa, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
primera_capa = cv2.applyColorMap(primera_capa, color1)
st.image(primera_capa, caption='Imagen 1',  width=200)

capa2= st.slider('Número de capa 2', 0, 299, 155)
color2=st.slider('Colormap 2', 0, 20, 0)
segunda_capa = img.read_band(capa2)
segunda_capa = cv2.normalize(segunda_capa, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
segunda_capa = cv2.applyColorMap(segunda_capa, color2)
st.image(segunda_capa, caption='Imagen 2', width=200)

alpha1 = st.slider('Transparencia Imagen 1', 0.0, 1.0, 0.5)
alpha2 = st.slider('Transparencia Imagen 2', 0.0, 1.0, 0.5)
superpuesta = cv2.addWeighted(primera_capa, alpha1, segunda_capa, alpha2, 0)
st.image(superpuesta, caption='Imagen Superpuesta', width=200)




