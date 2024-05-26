import streamlit as st
import cv2
import numpy as np
import spectral
import hydralit as hy
import os
import atexit
import csv
app = hy.HydraApp(title='Inicio')


if 'lista_imagenes' not in st.session_state:
    st.session_state.lista_imagenes = []
if 'archivos_subidos' not in st.session_state:
    st.session_state.archivos_subidos = []

carpetas = ['archivos_subidos', 'imagenes_guardadas', 'imagenes_trinarizadas']

for carpeta in carpetas:
    os.makedirs(carpeta, exist_ok=True)

def limpiar_carpeta():
    carpeta = 'archivos_subidos'
    for archivo in os.listdir(carpeta):
        ruta_archivo = os.path.join(carpeta, archivo)
        os.unlink(ruta_archivo)

atexit.register(limpiar_carpeta)

@app.addapp()
def Visualizar():

    def cargar_imagen(file_types):
        archivos_subidos = st.file_uploader("Sube tus archivos", accept_multiple_files=True, type=file_types)
        if option == "Hyperespectral":
            procesar_hyperespectral(archivos_subidos)
        elif option == "Imagen":
            procesar_imagen(archivos_subidos)
        elif option == "Trinarizada":
            procesar_trinarizada(archivos_subidos)

    def procesar_hyperespectral(archivos_subidos):
        if len(archivos_subidos) == 2:
            hdr_file = None
            bil_file = None

            for archivo in archivos_subidos:
                if archivo not in st.session_state.archivos_subidos:
                    st.session_state.archivos_subidos.append(archivo)
                ruta_archivos = os.path.join('archivos_subidos', archivo.name)
                with open(ruta_archivos, 'wb') as f:
                    f.write(archivo.read())
                if archivo.name.endswith('.bil.hdr'):
                    hdr_file = ruta_archivos
                elif archivo.name.endswith('.bil'):
                    bil_file = ruta_archivos

            if hdr_file and bil_file:
                img = spectral.open_image(hdr_file)
                valor = st.number_input("Elige una banda de 0 a 299:", min_value=0, max_value=299, value=40, step=1)
                banda = img.read_band(valor)
                banda = cv2.normalize(banda, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if st.button("Elegir banda"):
                    guardar_capa(banda)
            else:
                st.warning("Debes subir un archivo .bil y un archivo .bil.hdr de la misma imagen.")
        elif len(archivos_subidos) > 2:
            st.warning("Sólo debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")
        else:
            st.warning("Debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")


    def guardar_capa(imagen):
        nombre_img = f"Capa {len(st.session_state.lista_imagenes) + 1}"
        st.session_state.lista_imagenes.append((nombre_img, imagen))

    def procesar_imagen(archivos_subidos):
        for archivo in archivos_subidos:
                if archivo not in st.session_state.archivos_subidos:
                    st.session_state.archivos_subidos.append(archivo)
                    imagen = cv2.imdecode(np.frombuffer(archivo.read(), np.uint8), 1)
                    guardar_capa(imagen)
    
    def procesar_trinarizada(archivos_subidos):
        procesar_imagen(archivos_subidos)
        carpeta_trinarizadas = 'imagenes_trinarizadas'
        imagenes_trinarizadas = [f for f in os.listdir(carpeta_trinarizadas) if f.endswith('.png')]

        if imagenes_trinarizadas:
            imagen_seleccionada = st.selectbox("Selecciona una imagen trinarizada", imagenes_trinarizadas)
            if st.button("Cargar imagen trinarizada"):
                ruta_imagen = os.path.join(carpeta_trinarizadas, imagen_seleccionada)
                imagen = cv2.imread(ruta_imagen)
                guardar_capa(imagen)
    
    def convertir_a_color(imagen):
        if len(imagen.shape) == 2:
            return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            return imagen
        
    def mostrar_imagen():
        lista_img = [nombre_img for nombre_img, _ in st.session_state.lista_imagenes]
        st.header("Visualizar imágenes")
        seleccionadas = []
        transparencias = {}
        for nombre_img in lista_img:
            if st.checkbox(nombre_img):
                seleccionadas.append(nombre_img)
                transparencias[nombre_img] = st.slider(f"Transparencia para {nombre_img}", 0.0, 1.0, 1.0, step=0.01)

        if seleccionadas:

            imagen_capas = None
            for nombre_img in seleccionadas:
                _, img = next((n, i) for n, i in st.session_state.lista_imagenes if n == nombre_img)
                img = convertir_a_color(img)  
                transparencia = transparencias[nombre_img]
                if imagen_capas is None:
                    imagen_capas = np.zeros_like(img, dtype=np.float32)
                imagen_capas = cv2.addWeighted(imagen_capas, 1.0, img.astype(np.float32), transparencia, 0)

            
            imagen_capas = cv2.normalize(imagen_capas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(imagen_capas, caption="Imagen imagen_capas", width=333)
            
            
            if st.button("Guardar imagen"):
                guardar_imagen(imagen_capas)

    def guardar_imagen(imagen):
        nombre_imagen = f"Imagen{len(os.listdir('imagenes_guardadas')) + 1}.png"
        ruta_imagen = os.path.join('imagenes_guardadas', nombre_imagen)
        cv2.imwrite(ruta_imagen, cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR))
        st.success(f"Imagen guardada como {nombre_imagen} en la carpeta {'imagenes_guardadas'}")

    option = st.selectbox("Selecciona una opción para subir archivos", ["Hyperespectral", "Imagen", "Trinarizada"])

    if option == "Hyperespectral":
        file_types = ["bil", "bil.hdr"]
    elif option == "Imagen":
        file_types = ["jpg", "jpeg", "png", "tiff", "tif"]
    else:  
        file_types = ["png"]

    cargar_imagen(file_types)
    mostrar_imagen()
    

@app.addapp()
def Trinarizar():
    st.title("Nueva trinarización")


    def guardar_imagen(imagen):
        nombre_imagen = f"{nombre}.png"
        ruta_imagen = os.path.join('imagenes_trinarizadas', nombre_imagen)
        cv2.imwrite(ruta_imagen, cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR))
        st.success(f"Imagen guardada como {nombre_imagen} en la carpeta {'imagenes_trinarizadas'}")

    def cargar_hyper_bin():
        archivos_subidos = st.file_uploader("Sube tus archivos", accept_multiple_files=True, type=["bil", "bil.hdr"])
        
        if len(archivos_subidos) == 2:
            hdr_file = None
            bil_file = None

            for archivo in archivos_subidos:
                if archivo not in st.session_state.archivos_subidos:
                    st.session_state.archivos_subidos.append(archivo)
                ruta_archivos = os.path.join('archivos_subidos', archivo.name)
                with open(ruta_archivos, 'wb') as f:
                    f.write(archivo.read())
                if archivo.name.endswith('.bil.hdr'):
                    hdr_file = ruta_archivos
                elif archivo.name.endswith('.bil'):
                    bil_file = ruta_archivos

            if hdr_file and bil_file:
                img = spectral.open_image(hdr_file)
                banda = img.read_band(40)
                banda = cv2.normalize(banda, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                _, hoja_bin = cv2.threshold(banda, 127, 255, cv2.THRESH_BINARY)
                banda2 = img.read_band(222)
                banda2 = cv2.normalize(banda, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                _, gotas_bin = cv2.threshold(banda2, 12, 255, cv2.THRESH_BINARY)

                gotas = cv2.bitwise_xor(gotas_bin, hoja_bin)

                trinarizada = np.zeros((banda.shape[0], banda.shape[1], 3), dtype=np.uint8)
                trinarizada[hoja_bin == 0] = [255, 0, 0] 
                trinarizada[gotas == 255] = [0, 255, 0] 
                st.image(trinarizada)

                if st.button("Guardar imagen trinarizada y csv"):
                    guardar_imagen(trinarizada)

                    alto, ancho, _ = trinarizada.shape
                    ruta_archivos = os.path.join('imagenes_trinarizadas', f'{nombre}.csv')

                    with open(ruta_archivos, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        for x in range(alto):
                            fila = []
                            for y in range(ancho):
                                color_pixel = trinarizada[x, y]

                                if np.array_equal(color_pixel, [0, 0, 0]):
                                    valor = '00'
                                elif np.array_equal(color_pixel, [0, 255, 0]):
                                    valor = '01'
                                elif np.array_equal(color_pixel, [255, 0, 0]):
                                    valor = '10'
                                    
                                fila.append(valor)
                            writer.writerow(fila)
            else:
                st.warning("Debes subir un archivo .bil y un archivo .bil.hdr de la misma imagen.")
        elif len(archivos_subidos) > 2:
            st.warning("Sólo debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")
        else:
            st.warning("Debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")

    def mostrar_imagenes_trinarizadas():
        carpeta_imagenes = 'imagenes_trinarizadas'
        imagenes = [f for f in os.listdir(carpeta_imagenes) if f.endswith('.png')]
        
        if imagenes:
            st.write("Lista de imágenes trinarizadas:")
            for imagen in imagenes:
                st.write(imagen)
        else:
            st.write("No hay imágenes trinarizadas disponibles.")

    col1, col2 = st.columns([1,4])

    with col1:
        mostrar_imagenes_trinarizadas()
    
    with col2:
        nombre = st.text_input("Ingresa el nombre")
        cargar_hyper_bin()

app.run()



