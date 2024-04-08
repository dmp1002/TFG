import streamlit as st
import cv2
import numpy as np

def cargar_imagenes():
    st.title('Cargar Imágenes')

    imagenes = []
    imagenes_grises =[]
    imagenes_color =[]

    archivo_cargado = st.file_uploader("Selecciona una o más imágenes", type=["jpg", "jpeg", "png", "tiff", "tif"], accept_multiple_files=True)

    if archivo_cargado is not None:
        if isinstance(archivo_cargado, list):
            for img in archivo_cargado:
                st.image(img, width=200)
                imagenes.append((img))
                img_bytes = img.read()
                img_gris = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
                img_color = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                imagenes_color.append((img_color))
                st.image(img_gris, width=200)
                imagenes_grises.append((img_gris))
        else:
            st.image(archivo_cargado, width=200)
            imagenes.append((archivo_cargado))
            img_bytes = archivo_cargado.read()
            img_gris = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            img_color = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            img_gris= cv2.imread(archivo_cargado.img.uploaded_url)
            img_color = cv2.imread(archivo_cargado.img.uploaded_url)
            imagenes_color.append((img_color))
            st.image(img_gris, width=200)
            imagenes_grises.append((img_gris))

    return imagenes, imagenes_grises, imagenes_color

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

def convertir_tipo_datos(img1, img2):
    if img1.dtype != img2.dtype:
        if img1.dtype == np.uint8 and img2.dtype == np.float32:
            img1 = cv2.convertScaleAbs(img1)
        elif img1.dtype == np.float32 and img2.dtype == np.uint8:
            img2 = cv2.convertScaleAbs(img2)

    return img1, img2
imagenes, imagenes_grises, imagenes_color=cargar_imagenes()
imagenes_binarias=binarizar_imagenes(imagenes_grises)
imagenes_bordes=detectar_bordes(imagenes_binarias)
lista_contornos=detectar_contornos(imagenes_bordes)


for i in range(len(imagenes_grises)):
    imagen_con_contorno = cv2.drawContours(cv2.cvtColor(imagenes_grises[i], cv2.COLOR_GRAY2RGB), lista_contornos[i], -1, (0, 255, 0), 7)
    st.image(imagen_con_contorno, width=200)

for i in range(len(imagenes_grises)):
    contornos = lista_contornos[i]

def alinear_imagenes(imagen_referencia, imagen_a_alinear):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    _, warp_matrix = cv2.findTransformECC(imagen_referencia, imagen_a_alinear, warp_matrix, cv2.MOTION_TRANSLATION)
    imagen_a_alinear_alineada = cv2.warpAffine(imagen_a_alinear, warp_matrix, (imagen_a_alinear.shape[1], imagen_a_alinear.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return imagen_a_alinear_alineada

imagen_alineada = alinear_imagenes(imagenes_grises[0], imagenes_grises[1])

imagen_combinada = cv2.addWeighted(imagenes_grises[0], 0.5, imagen_alineada, 0.5, 0)

st.image(imagen_combinada, width=200)

