import streamlit as st
import cv2
import numpy as np

def cargar_imagenes():
    st.title('Cargar Imágenes')

    imagenes = []
    imagenes_grises =[]

    archivo_cargado = st.file_uploader("Selecciona una o más imágenes", type=["jpg", "jpeg", "png", "tiff", "tif"], accept_multiple_files=True)

    if archivo_cargado is not None:
        if isinstance(archivo_cargado, list):
            for img in archivo_cargado:
                st.image(img, width=200)
                imagenes.append((img))
                img_bytes = img.read()
                img_gris = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                st.image(img_gris, width=200)
                imagenes_grises.append((img_gris))
        else:
            st.image(archivo_cargado, width=200)
            imagenes.append((archivo_cargado))
            img_bytes = archivo_cargado.read()
            img_gris = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            img_gris=cv2.imread(archivo_cargado.img.uploaded_url)
            st.image(img_gris, width=200)
            imagenes_grises.append((img_gris))

    return imagenes, imagenes_grises

def binarizar_imagenes(imagenes_grises):
    imagenes_binarias=[]
    for img in imagenes_grises:
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        st.image(binary_img, width=200)
        imagenes_binarias.append(binary_img)

    return imagenes_binarias

imagenes, imagenes_grises=cargar_imagenes()
imagenes_binarias=binarizar_imagenes(imagenes_grises)