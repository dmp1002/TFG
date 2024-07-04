import streamlit as st
import cv2
import numpy as np
import spectral
import hydralit as hy
import os, csv, datetime
import pandas as pd
from skimage import measure
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import filedialog
from streamlit_image_zoom import image_zoom

app = hy.HydraApp(title="TriVitis-VitiScan-VitisLab-VitisDropScan",use_loader=False,hide_streamlit_markers=False)

if 'lista_capas' not in st.session_state:
    st.session_state.lista_capas = []
if 'trinarizada' not in st.session_state:
    st.session_state.trinarizada = None
if 'archivos_subidos' not in st.session_state:
    st.session_state.archivos_subidos = []
if 'trinarizadas_cargadas' not in st.session_state:
    st.session_state.trinarizadas_cargadas = {}
if 'carpeta_origen' not in st.session_state:
    st.session_state.carpeta_origen = None
if 'carpeta_destino' not in st.session_state:
    st.session_state.carpeta_destino = None

carpetas = ['archivos_subidos','imagenes_guardadas', 'imagenes_trinarizadas']

for carpeta in carpetas:
    os.makedirs(carpeta, exist_ok=True)

@app.addapp(title="Visualizar", icon="🖥️")
def Visualizar():
    @st.experimental_dialog("Cargar capas:",width="large")
    def cargar_imagen():
        opcion = st.selectbox("Selecciona una opción", ["Imagen hiperespectral", "Imagen estándar", "Imagen trinarizada"])

        if opcion == "Imagen hiperespectral":
            archivos_subidos = st.file_uploader("Sube una imagen hiperespectral", accept_multiple_files=True, type=["bil", "bil.hdr"])
            with open("estilos.css") as f:
                css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
            procesar_hyperespectral(archivos_subidos)
        elif opcion == "Imagen estándar":
            archivos_subidos = st.file_uploader("Sube una imagen", accept_multiple_files=True, type=["jpg", "jpeg", "png", "tiff", "tif"])
            with open("estilos.css") as f:
                css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
            procesar_imagen(archivos_subidos)
        elif opcion == "Imagen trinarizada":
            procesar_trinarizada()

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
                nombre_base_hdr = os.path.splitext(os.path.splitext(os.path.basename(hdr_file))[0])[0]
                nombre_base_bil = os.path.splitext(os.path.basename(bil_file))[0]
                if nombre_base_hdr == nombre_base_bil:
                    img = spectral.open_image(hdr_file)
                    valor = st.number_input("Elige una banda de 0 a 299:", min_value=0, max_value=299, value=40, step=1)
                    if st.button("Elegir banda"):
                        banda = img.read_band(valor)
                        guardar_capa(banda, nombre_base_hdr)
                else:
                    st.error("Debes subir un archivo .bil y un archivo .bil.hdr de la misma imagen.")
            else:
                st.error("Debes subir un archivo .bil y un archivo .bil.hdr de la misma imagen.")
        elif len(archivos_subidos) > 2:
            st.error("Sólo debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")
        else:
            st.info("Debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")



    def guardar_capa(banda,nombre_hyper):
        imagen = cv2.normalize(banda, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        nombre_img = f"Capa {len(st.session_state.lista_capas) + 1} - {nombre_hyper}"
        st.session_state.lista_capas.append((nombre_img, imagen))
        st.rerun()

    def procesar_imagen(archivos_subidos):
        for archivo in archivos_subidos:
            imagen = cv2.imdecode(np.frombuffer(archivo.read(), np.uint8), 1)
            nombre = archivo.name
            guardar_capa(imagen,nombre)
    
    def procesar_trinarizada():
        imagenes_trinarizadas = list(st.session_state.trinarizadas_cargadas.keys())

        if imagenes_trinarizadas:
            imagen_seleccionada = st.selectbox("Selecciona una imagen trinarizada", imagenes_trinarizadas)
            if st.button("Cargar imagen trinarizada"):
                imagen = st.session_state.trinarizadas_cargadas[imagen_seleccionada]
                nombre_hyper=imagen_seleccionada
                guardar_capa(imagen,nombre_hyper)
        else:
            st.markdown("No hay imágenes trinarizadas cargadas")
        
    def convertir_a_color(imagen):
        if len(imagen.shape) == 2:
            return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            return imagen
        
    def mostrar_imagen():
        col1, col2 = st.columns([3, 7])

        with col2:
            seleccionadas, transparencias = mostrar_imagen2()

        with col1:
            menu_lateral(seleccionadas, transparencias)

    def menu_lateral(seleccionadas, transparencias):
        with st.container(height=600,border=False):
            estilo_centrado = """
                <style>
                .centered-text {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100%;
                    text-align: center;
                }
                </style>
                """
            ocultar_fs = '''
                    <style>
                    button[title="View fullscreen"]{
                        visibility: hidden;}
                    </style>
                    '''
            st.markdown(estilo_centrado, unsafe_allow_html=True)
                
            if not st.session_state.lista_capas:
                st.markdown('<div class="centered-text">Sin capas</div>', unsafe_allow_html=True)
            elif seleccionadas:
                imagen_capas = None
                for idx, nombre_img in enumerate(seleccionadas):
                    _, img = next((n, i) for n, i in st.session_state.lista_capas if n == nombre_img)
                    img = convertir_a_color(img)
                    transparencia = transparencias[nombre_img]
                    if idx == 0:
                        imagen_capas = img.astype(np.float32)
                    else:
                        imagen_capas = cv2.addWeighted(imagen_capas, 1.0, img.astype(np.float32), transparencia, 0)

                imagen_capas = cv2.normalize(imagen_capas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                image_zoom(
                        image=imagen_capas,
                        mode="both",
                        size=500,
                        zoom_factor=10,
                        increment=1,
                    )
                st.markdown(ocultar_fs, unsafe_allow_html=True)

                if st.button("💾Guardar imagen"):
                    guardar_imagen(imagen_capas)
            else:
                st.markdown('<div class="centered-text">Sin capas seleccionadas</div>', unsafe_allow_html=True)

    def mostrar_imagen2():
        with st.container(height=539):
            lista_img = [nombre_img for nombre_img, _ in st.session_state.lista_capas]
            seleccionadas = []
            transparencias = {}
            capas_a_eliminar = []

            for nombre_img in lista_img:
                col3, col4 = st.columns([9, 1])
                with col3:
                    if st.toggle(nombre_img):
                        seleccionadas.append(nombre_img)
                        transparencia = st.slider(f"Transparencia para {nombre_img}", 0.0, 1.0, 1.0, step=0.01)
                        transparencias[nombre_img] = transparencia
                with col4:
                    if st.button("🗑️", key=f"Eliminar_{nombre_img}"):
                        capas_a_eliminar.append(nombre_img)

            if capas_a_eliminar:
                for nombre_img in capas_a_eliminar:
                    st.session_state.lista_capas = [(n, i) for n, i in st.session_state.lista_capas if n != nombre_img]
                    st.rerun()
        col3,col4,_ = st.columns([1,1,3])
        with col3:
            if st.button("➕AÑADIR CAPAS"):
                cargar_imagen()
        with col4:
            if st.button("🗑️ELIMINAR TODAS"):
                st.session_state.lista_capas = []
                st.rerun()
        return seleccionadas,transparencias

    def guardar_imagen(imagen):
        nombre_imagen = f"Imagen{len(os.listdir('imagenes_guardadas')) + 1}.png"
        ruta_imagen = os.path.join('imagenes_guardadas', nombre_imagen)
        cv2.imwrite(ruta_imagen, cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR))
        st.success(f"{nombre_imagen} guardada en la carpeta {'imagenes_guardadas'}.")

    mostrar_imagen()


@app.addapp(title="Trinarizar", icon="🔼")
def Trinarizar():
    bandas_seleccionadas = False
    ocultar_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''
    def leer_valores_xml(ruta_xml):
        arbol = ET.parse(ruta_xml)
        raiz = arbol.getroot()

        banda1 = int(raiz.find('bandas_por_defecto/banda1').text)
        banda2 = int(raiz.find('bandas_por_defecto/banda2').text)

        return banda1, banda2
    
    def cargar_hyper_bin():
        nonlocal bandas_seleccionadas

        ruta_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_xml = os.path.join(ruta_actual, 'bandas.xml')
        
        valor_defecto_paso1, valor_defecto_paso2 = leer_valores_xml(ruta_xml)

        archivos_subidos = st.file_uploader("Sube una imagen hiperespectral", accept_multiple_files=True, type=["bil", "bil.hdr"])
        with open("estilos.css") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        if len(archivos_subidos) == 2:
            hdr_file = None
            bil_file = None

            for archivo in archivos_subidos:
                if archivo not in st.session_state.archivos_subidos:
                    st.session_state.archivos_subidos.append(archivo)
                ruta_archivos = os.path.join('archivos_subidos', archivo.name)
                ruta_archivos = os.path.join('archivos_subidos', archivo.name)
                with open(ruta_archivos, 'wb') as f:
                    f.write(archivo.read())
                if archivo.name.endswith('.bil.hdr'):
                    hdr_file = ruta_archivos
                elif archivo.name.endswith('.bil'):
                    bil_file = ruta_archivos
                    nombre_hyper = os.path.splitext(archivo.name)[0]  

            if hdr_file and bil_file:
                img = spectral.open_image(hdr_file)

                st.text("Paso 1: Detección de hoja")
                paso1 = st.number_input("Elige una banda de 0 a 299:", min_value=0, max_value=299, value=valor_defecto_paso1, step=1)
                
                st.text("Paso 2: Detección de gotas")
                paso2 = st.number_input("Elige una banda de 0 a 299:", min_value=0, max_value=299, value=valor_defecto_paso2, step=1)
                paso2_min, paso2_max = st.slider("Elige un rango del umbral de 0 a 1000:", min_value=0, max_value=1000, value=(50, 800), step=1)
                if not bandas_seleccionadas:
                    bandas_seleccionadas = st.button("Procesar imagen")
                if bandas_seleccionadas or st.session_state.trinarizada is not None:
                    banda = img.read_band(paso1)
                    banda = cv2.normalize(banda, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    _, hoja_bin = cv2.threshold(banda, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    banda2 = img.read_band(paso2)
                    banda2 = cv2.normalize(banda2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    gotas_bin = cv2.inRange(banda2, int(paso2_min * 255/1000), int(paso2_max * 255/1000))


                    trinarizada = np.zeros((banda.shape[0], banda.shape[1], 3), dtype=np.uint8)

                    trinarizada[(hoja_bin == 0) & (gotas_bin == 0)] = [255, 0, 0]
                    trinarizada[(hoja_bin == 0) & (gotas_bin == 255)] = [0, 255, 0] 
                    trinarizada[hoja_bin == 255] = [0, 0, 0]  

                    mascara_gotas = (trinarizada[:, :, 0] == 255) & (trinarizada[:, :, 1] == 0) & (trinarizada[:, :, 2] == 0)

                    mascara_dilatada = cv2.dilate(mascara_gotas.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

                    mascara_rellenada = cv2.morphologyEx(mascara_dilatada, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

                    mascara_final = cv2.erode(mascara_rellenada, np.ones((3, 3), np.uint8), iterations=1)

                    etiquetas = measure.label(mascara_final, connectivity=2)
                    propiedades = measure.regionprops(etiquetas)

                    area_min_umbral = 100
                    area_max_umbral = 500 

                    trinarizada[(mascara_final == 1) & ~mascara_gotas] = [255, 0, 0] 

                    for prop in propiedades:
                        if prop.area > area_max_umbral or prop.area < area_min_umbral:
                            for coord in prop.coords:
                                trinarizada[coord[0], coord[1]] = [0, 255, 0] 

                    st.session_state.trinarizada = trinarizada

                    total_pixeles = trinarizada.shape[0] * trinarizada.shape[1]
                    num_pixeles_hoja = np.count_nonzero(np.all(trinarizada == [0, 255, 0], axis=-1))
                    num_pixeles_gotas = np.count_nonzero(np.all(trinarizada == [255, 0, 0], axis=-1))

                    df = pd.DataFrame({
                        '': ['Pixeles Hoja', 'Pixeles Gota'],
                        '#': [num_pixeles_hoja, num_pixeles_gotas],
                        '%': [round(num_pixeles_hoja / total_pixeles * 100, 2), round(num_pixeles_gotas / num_pixeles_hoja * 100, 2) if num_pixeles_hoja != 0 else 0],
                    })

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(trinarizada, width=333)
                        st.markdown(ocultar_fs, unsafe_allow_html=True)
                    with col2:
                        st.write(df)

                    if st.session_state.trinarizada is not None and st.button("Guardar imagen trinarizada y csv"):
                        trinarizada = st.session_state.trinarizada
                        nombre_imagen = f"{nombre_hyper}.png"
                        ruta_imagen = os.path.join('imagenes_trinarizadas', nombre_imagen)
                        cv2.imwrite(ruta_imagen, cv2.cvtColor(trinarizada, cv2.COLOR_RGB2BGR))
                        
                        alto, ancho, _ = trinarizada.shape
                        ruta_archivos = os.path.join('imagenes_trinarizadas', f'{nombre_hyper}.csv')

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
                        
                        ruta_csv_porc = os.path.join('imagenes_trinarizadas', f'{nombre_hyper}_porcentajes.csv')
                        df.to_csv(ruta_csv_porc, index=False)
                        
                        st.success('Archivos exportados correctamente')
                    if st.session_state.trinarizada is not None and st.button("Cargar trinarizada"):
                        trinarizada = st.session_state.trinarizada
                        fecha = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        st.session_state.trinarizadas_cargadas[(nombre_hyper, fecha)] = trinarizada
                        st.rerun()
                        
        elif len(archivos_subidos) > 2:
            st.error("Sólo debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")
        else:
            st.info("Debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")

    def mostrar_imagenes_trinarizadas():
        imagenes = st.session_state.trinarizadas_cargadas
        
        if imagenes:
            st.write("Trinarizadas cargadas:")
            for imagen in imagenes:
                st.write(imagen)
        else:
            st.write("No hay imágenes trinarizadas disponibles.")

    col1, col2 = st.columns([7,2])

    with col1:
        nombre = st.text_input("Ingresa el nombre")
        cargar_hyper_bin()
        st.expander("hello")
    
    with col2:
        mostrar_imagenes_trinarizadas()

@app.addapp(title="Trinarizar por lotes", icon="🗂️")
def TrinarizarPorLotes():

    css = '''
    <style>
        .stNumberInput > div {
            width: 333px; 
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    def leer_valores_xml(ruta_xml):
        arbol = ET.parse(ruta_xml)
        raiz = arbol.getroot()

        banda1 = int(raiz.find('bandas_por_defecto/banda1').text)
        banda2 = int(raiz.find('bandas_por_defecto/banda2').text)

        return banda1, banda2

    def file_selector():
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(master=root)
        root.destroy()
        
        return folder_path
    
    def procesar_lote(carpeta_origen, carpeta_destino, paso1, paso2, paso2_min, paso2_max):
        archivos_hdr = [f for f in os.listdir(carpeta_origen) if f.endswith('.hdr') and 'wet' in f]
        archivos_bil = [f for f in os.listdir(carpeta_origen) if f.endswith('.bil') and 'wet' in f]

        total_imagenes = len(archivos_hdr)
        progreso = st.progress(0)

        for i, bil in enumerate(archivos_bil):
            nombre = os.path.splitext(bil)[0]
            hdr = f"{nombre}.bil.hdr"
            if hdr in archivos_hdr:
                ruta_hdr = os.path.join(carpeta_origen, hdr)
                
                img = spectral.open_image(ruta_hdr)
                
                banda = img.read_band(paso1)
                banda = cv2.normalize(banda, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                _, hoja_bin = cv2.threshold(banda, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                banda2 = img.read_band(paso2)
                banda2 = cv2.normalize(banda2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                gotas_bin = cv2.inRange(banda2, int(paso2_min * 255/1000), int(paso2_max * 255/1000))

                trinarizada = np.zeros((banda.shape[0], banda.shape[1], 3), dtype=np.uint8)

                trinarizada[(hoja_bin == 0) & (gotas_bin == 0)] = [255, 0, 0]
                trinarizada[(hoja_bin == 0) & (gotas_bin == 255)] = [0, 255, 0] 
                trinarizada[hoja_bin == 255] = [0, 0, 0]  

                mascara_gotas = (trinarizada[:, :, 0] == 255) & (trinarizada[:, :, 1] == 0) & (trinarizada[:, :, 2] == 0)

                mascara_dilatada = cv2.dilate(mascara_gotas.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)

                mascara_rellenada = cv2.morphologyEx(mascara_dilatada, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

                mascara_final = cv2.erode(mascara_rellenada, np.ones((3, 3), np.uint8), iterations=1)

                etiquetas = measure.label(mascara_final, connectivity=2)
                propiedades = measure.regionprops(etiquetas)

                area_min_umbral = 100
                area_max_umbral = 500 

                trinarizada[(mascara_final == 1) & ~mascara_gotas] = [255, 0, 0] 

                for prop in propiedades:
                    if prop.area > area_max_umbral or prop.area < area_min_umbral:
                        for coord in prop.coords:
                            trinarizada[coord[0], coord[1]] = [0, 255, 0] 

                nombre_imagen = f"{nombre}.png"
                ruta_imagen = os.path.join(carpeta_destino, nombre_imagen)
                cv2.imwrite(ruta_imagen, cv2.cvtColor(trinarizada, cv2.COLOR_RGB2BGR))

                progreso.progress((i + 1) / total_imagenes, text=f"Trinarizando las imágenes...({i + 1}/{total_imagenes})")

    if st.button("Selecciona la carpeta de origen"):
        st.session_state.carpeta_origen = file_selector()  
    if st.session_state.carpeta_origen:
        st.text(f"Carpeta de origen seleccionada: {st.session_state.carpeta_origen}")

    if st.button("Selecciona la carpeta de destino"):
        st.session_state.carpeta_destino = file_selector()
    if st.session_state.carpeta_destino:
        st.text(f"Carpeta de destino seleccionada: {st.session_state.carpeta_destino}")

    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_xml = os.path.join(ruta_actual, 'bandas.xml')
    valor_defecto_paso1, valor_defecto_paso2 = leer_valores_xml(ruta_xml)

    st.text("Paso 1: Detección de hoja")
    paso1 = st.number_input("Elige una banda de 0 a 299:", min_value=0, max_value=299, value=valor_defecto_paso1, step=1)
    st.text("Paso 2: Detección de gotas")
    paso2 = st.number_input("Elige una banda de 0 a 299:", min_value=0, max_value=299, value=valor_defecto_paso2, step=1)
    paso2_min, paso2_max = st.slider("Elige un rango del umbral de 0 a 1000:", min_value=0, max_value=1000, value=(50, 800), step=1)
    if st.button("Trinarizar por lotes"):
        if not st.session_state.carpeta_origen or not st.session_state.carpeta_destino:
            st.error("Por favor selecciona ambas carpetas antes de continuar.")
        else:
            procesar_lote(st.session_state.carpeta_origen, st.session_state.carpeta_destino, paso1, paso2, paso2_min, paso2_max)
            st.success("Trinarización por lotes completada")

app.run()