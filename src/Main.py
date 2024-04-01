import streamlit as st
import cv2

def cargar_imagenes():
    st.title('Cargar Imágenes')

    imagenes = []

    archivo_cargado = st.file_uploader("Selecciona una o más imágenes", type=["jpg", "jpeg", "png", "tiff", "tif"], accept_multiple_files=True)

    if archivo_cargado is not None:
        if isinstance(archivo_cargado, list):
            for img in archivo_cargado:
                st.image(img, width=200)
                imagenes.append((img))
        else:
            st.image(archivo_cargado, width=200)
            imagenes.append((archivo_cargado))

    return imagenes

imagenes=cargar_imagenes()

imagen_DR = cv2.imread('./hojas/ESTEMPDR06072310549.tiff')
imagen_WE = cv2.imread('./hojas/ESTEMPWE06072310549.tiff')

st.image(imagen_DR, caption='Dry', width=150)
st.image(imagen_WE, caption='Wet', width=150) 