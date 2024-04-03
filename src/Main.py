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

def detectar_bordes(imagenes_binarias):
    imagenes_bordes=[]
    for img in imagenes_binarias:
        img_bordes = cv2.Canny(img, 100, 200) 
        st.image(img_bordes, width=200)
        imagenes_bordes.append(img_bordes)
    return imagenes_bordes

def detectar_contornos(imagenes_bordes):
    lista_contornos=[]
    for img in imagenes_bordes:
        contornos, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        lista_contornos.append(contornos)
    return lista_contornos

imagenes, imagenes_grises=cargar_imagenes()
imagenes_binarias=binarizar_imagenes(imagenes_grises)
imagenes_bordes=detectar_bordes(imagenes_binarias)
lista_contornos=detectar_contornos(imagenes_bordes)

for i in range(len(imagenes_grises)):
    imagen_con_contorno = cv2.drawContours(cv2.cvtColor(imagenes_grises[i], cv2.COLOR_GRAY2RGB), lista_contornos[i], -1, (0, 255, 0), 7)
    st.image(imagen_con_contorno, width=200)