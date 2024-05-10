import streamlit as st
import cv2
import numpy as np

def cargar_imagenes():
    st.title('Cargar Imágenes')

    archivo_cargado = st.file_uploader("Selecciona una o más imágenes", type=["jpg", "jpeg", "png", "tiff", "tif"], accept_multiple_files=True)
    if archivo_cargado is not None:
        imagenes, imagenes_grises, imagenes_color=procesar_imagenes_subidas(archivo_cargado)

        return imagenes, imagenes_grises, imagenes_color
    
def procesar_imagenes_subidas(archivo_cargado):
    imagenes = []
    imagenes_grises =[]
    imagenes_color =[]

    if isinstance(archivo_cargado, list):
        for img in archivo_cargado:
            imagenes.append((img))
            img_bytes = img.read()
            img_gris = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            img_color = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            imagenes_color.append((img_color))
            imagenes_grises.append((img_gris))
    else:
        imagenes.append((archivo_cargado))
        img_bytes = archivo_cargado.read()
        img_gris = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        img_color = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_gris= cv2.imread(archivo_cargado.img.uploaded_url)
        img_color = cv2.imread(archivo_cargado.img.uploaded_url)
        imagenes_color.append((img_color))
        imagenes_grises.append((img_gris))
    
    mostrar_imagenes(imagenes_color)

    return imagenes, imagenes_grises, imagenes_color

def binarizar_imagenes(imagenes_grises):
    imagenes_binarias = []
    for img in imagenes_grises:
        _, binary_img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        imagenes_binarias.append(binary_img)

    num_imagenes = len(imagenes_binarias)
    cols = st.columns(num_imagenes)

    for i in range(num_imagenes):
        with cols[i]:
            st.image(imagenes_binarias[i])
            porcentaje = calcular_porcentaje_hoja(imagenes_binarias[i])
            st.markdown(f"Porcentaje de hoja: {porcentaje}%")

    return imagenes_binarias

def mostrar_imagenes(imagenes):
    num_imagenes = len(imagenes)
    cols = st.columns(num_imagenes)

    for i in range(num_imagenes):
        with cols[i]:
            st.image(imagenes[i])
            
def calcular_porcentaje_hoja(imagen_binaria):
    total_pixeles = imagen_binaria.shape[0] * imagen_binaria.shape[1]
    pixeles_negros = total_pixeles - cv2.countNonZero(imagen_binaria)
    porcentaje= round((pixeles_negros / total_pixeles) * 100, 2)
    return porcentaje

def binarizar_imagenes_gotas(imagenes_rgb):
    imagenes_binarias_gotas=[]
    for img in imagenes_rgb:
        img=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        binary_img = cv2.bitwise_not(binary_img)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_img = cv2.dilate(binary_img, se, iterations=1)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, se)
        binary_img = cv2.erode(binary_img, se, iterations=1)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imagenes_binarias_gotas.append(binary_img)

    mostrar_imagenes(imagenes_binarias_gotas)

    return imagenes_binarias_gotas

def aplicar_mascaras(imagenes_color, imagenes_binarias):
    imagenes_recortadas=[]
    for i in range(len(imagenes_color)):
        img = imagenes_color[i]
        mask = imagenes_binarias[i]

        mask_inv = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(img, img, mask=mask_inv)
        result = img.copy()
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask_inv

        imagenes_recortadas.append(result)

    mostrar_imagenes(imagenes_recortadas)

    return imagenes_recortadas


def detectar_bordes(imagenes_binarias):
    imagenes_bordes=[]
    for img in imagenes_binarias:
        img_bordes = cv2.Canny(img, 100, 200) 
        imagenes_bordes.append(img_bordes)

    mostrar_imagenes(imagenes_bordes)

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


def rgb_cambiar(imagenes_color):
    imagenes_rgb=[]
    for img in imagenes_color:
        b,g,r=cv2.split(img)
        new_r= new_r = np.clip(r*(639.1), 55, 155).astype(np.uint8)
        #new_r = r
        new_g = np.clip(g*(801.4), 55, 155).astype(np.uint8)
        new_b= np.clip(b*(459.2), 55, 155).astype(np.uint8)
        #new_b = b

        new_image = cv2.merge((new_b, new_g, new_r))
        imagenes_rgb.append(new_image)

    mostrar_imagenes(imagenes_rgb)

    return imagenes_rgb

imagenes, imagenes_grises, imagenes_color=cargar_imagenes()
imagenes_binarias=binarizar_imagenes(imagenes_grises)
imagenes_recortadas=aplicar_mascaras(imagenes_color, imagenes_binarias)
imagenes_rgb=rgb_cambiar(imagenes_color)
imagenes_binarias_gotas=binarizar_imagenes_gotas(imagenes_rgb)
imagenes_recortadas_gotas=aplicar_mascaras(imagenes_rgb, imagenes_binarias_gotas)

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


def adjust_rgb(image, r_value, g_value, b_value):
    adjusted_image = np.copy(image)
    adjusted_image[:,:,0] += b_value  
    adjusted_image[:,:,1] += g_value  
    adjusted_image[:,:,2] += r_value  
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)

r_value = st.slider('Red', 0, 1000, 254)
g_value = st.slider('Green', 0, 1000, 0)
b_value = st.slider('Blue', 0, 1000, 595)

adjusted_img = adjust_rgb(imagenes_color[1], r_value, g_value, b_value)
st.image(adjusted_img, channels='BGR', width=500)