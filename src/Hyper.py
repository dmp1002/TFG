import streamlit as st
import cv2
import numpy as np
import spectral
import hydralit as hy
import os, atexit, csv, datetime
import pandas as pd

app = hy.HydraApp(title='Inicio',use_loader=False)


if 'lista_capas' not in st.session_state:
    st.session_state.lista_capas = []
if 'archivos_subidos' not in st.session_state:
    st.session_state.archivos_subidos = []
if 'trinarizada' not in st.session_state:
    st.session_state.trinarizada = None
if 'trinarizadas_cargadas' not in st.session_state:
    st.session_state.trinarizadas_cargadas = {}

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
    @st.experimental_dialog("Cargar capas:",width="large")
    def cargar_imagen():
        option = st.selectbox("Selecciona una opción para subir archivos", ["Hyperespectral", "Imagen", "Trinarizada"])

        if option == "Hyperespectral":
            file_types = ["bil", "bil.hdr"]
        elif option == "Imagen":
            file_types = ["jpg", "jpeg", "png", "tiff", "tif"]
        else:  
            file_types = ["png"]

        archivos_subidos = st.file_uploader("Sube tus archivos", accept_multiple_files=True, type=file_types)
        if option == "Hyperespectral":
            procesar_hyperespectral(archivos_subidos)
        elif option == "Imagen":
            procesar_imagen(archivos_subidos)
        elif option == "Trinarizada":
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
                img = spectral.open_image(hdr_file)
                valor = st.number_input("Elige una banda de 0 a 299:", min_value=0, max_value=299, value=40, step=1)
                banda = img.read_band(valor)
                if st.button("Elegir banda"):
                    guardar_capa(banda)
            else:
                st.warning("Debes subir un archivo .bil y un archivo .bil.hdr de la misma imagen.")
        elif len(archivos_subidos) > 2:
            st.warning("Sólo debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")
        else:
            st.warning("Debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")


    def guardar_capa(banda):
        imagen = cv2.normalize(banda, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        nombre_img = f"Capa {len(st.session_state.lista_capas) + 1}"
        st.session_state.lista_capas.append((nombre_img, imagen))
        st.rerun()

    def procesar_imagen(archivos_subidos):
        for archivo in archivos_subidos:
                if archivo not in st.session_state.archivos_subidos:
                    st.session_state.archivos_subidos.append(archivo)
                    imagen = cv2.imdecode(np.frombuffer(archivo.read(), np.uint8), 1)
                    guardar_capa(imagen)
    
    def procesar_trinarizada():
        imagenes_trinarizadas = list(st.session_state.trinarizadas_cargadas.keys())

        if imagenes_trinarizadas:
            imagen_seleccionada = st.selectbox("Selecciona una imagen trinarizada", imagenes_trinarizadas)
            if st.button("Cargar imagen trinarizada"):
                imagen = st.session_state.trinarizadas_cargadas[imagen_seleccionada]
                guardar_capa(imagen)
        else:
            st.markdown("No hay imágenes trinarizadas cargadas")
        
    def convertir_a_color(imagen):
        if len(imagen.shape) == 2:
            return cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            return imagen
        
    def mostrar_imagen():
        col1, col2 = st.columns([1, 2])

        with col2:
            with st.container(height=388):
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

        with col1:
            estilo_centrado = """
            <style>
            .centered-text {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100%;
                text-align: center;
                margin-top: 50%; /* Ajusta este valor según tu preferencia */
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
                for nombre_img in seleccionadas:
                    _, img = next((n, i) for n, i in st.session_state.lista_capas if n == nombre_img)
                    img = convertir_a_color(img)
                    transparencia = transparencias[nombre_img]
                    if imagen_capas is None:
                        imagen_capas = np.zeros_like(img, dtype=np.float32)
                    imagen_capas = cv2.addWeighted(imagen_capas, 1.0, img.astype(np.float32), transparencia, 0)

                imagen_capas = cv2.normalize(imagen_capas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                st.image(imagen_capas, width=333)
                st.markdown(ocultar_fs, unsafe_allow_html=True)

                if st.button("💾Guardar imagen"):
                    guardar_imagen(imagen_capas)
            else:
                st.markdown('<div class="centered-text">Sin capas seleccionadas</div>', unsafe_allow_html=True)

    def guardar_imagen(imagen):
        nombre_imagen = f"Imagen{len(os.listdir('imagenes_guardadas')) + 1}.png"
        ruta_imagen = os.path.join('imagenes_guardadas', nombre_imagen)
        cv2.imwrite(ruta_imagen, cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR))
        st.success(f"Imagen guardada como {nombre_imagen} en la carpeta {'imagenes_guardadas'}")


    mostrar_imagen()
    _,_,_,col3,col4,_,_,_ = st.columns(8)

    with col3:
        if st.button("➕AÑADIR CAPAS"):
            pass
            #cargar_imagen()
    with col4:
        if st.button("🗑️ELIMINAR TODAS"):
            st.session_state.lista_capas = []
            st.rerun()
    
    cargar_imagen()

@app.addapp()
def Trinarizar():
    bandas_seleccionadas = False
    ocultar_fs = '''
    <style>
    button[title="View fullscreen"]{
        visibility: hidden;}
    </style>
    '''

    def cargar_hyper_bin():
        nonlocal bandas_seleccionadas
        archivos_subidos = st.file_uploader(".",accept_multiple_files=True, type=["bil", "bil.hdr"],label_visibility="hidden")

        css = '''
        <style>
            [data-testid='stFileUploader'] {
                width: max-content; 
            }
            [data-testid='stFileUploader'] section > button{
                display: none;
            }
            [data-testid="stFileUploaderDropzone"] div div::before {
                content:"Cargar hipercubo"
            }
            [data-testid="stFileUploaderDropzone"] div div span{
                display:none;
            }
            [data-testid="stFileUploaderDropzone"] div div small{
                display:none;
            }
        </style>
        '''
    
        st.markdown(css, unsafe_allow_html=True)
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

                st.text("Paso 1: Detección de hoja")
                paso1 = st.number_input("Elige una banda de 0 a 299:", min_value=0, max_value=299, value=40, step=1)
                
                st.text("Paso 2: Detección de gotas")
                paso2 = st.number_input("Elige una banda de 0 a 299:", min_value=0, max_value=299, value=50, step=1)
                if not bandas_seleccionadas:
                    bandas_seleccionadas = st.button("Procesar imagen")
                if bandas_seleccionadas or st.session_state.trinarizada is not None:
                    banda=img.read_band(paso1)
                    banda = cv2.normalize(banda, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    _, hoja_bin = cv2.threshold(banda, 127, 255, cv2.THRESH_BINARY)
                    
                    banda2=img.read_band(paso2)
                    banda2 = cv2.normalize(banda2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    _, gotas_bin = cv2.threshold(banda2, 12, 255, cv2.THRESH_BINARY)

                    gotas = cv2.bitwise_xor(gotas_bin, hoja_bin)

                    trinarizada = np.zeros((banda.shape[0], banda.shape[1], 3), dtype=np.uint8)
                    trinarizada[hoja_bin == 0] = [255, 0, 0] 
                    trinarizada[gotas == 255] = [0, 255, 0] 
                    st.session_state.trinarizada = trinarizada

                    total_pixeles = trinarizada.shape[0] * trinarizada.shape[1]
                    num_pixeles_hoja = np.count_nonzero(np.all(trinarizada == [0, 255, 0], axis=-1))
                    num_pixeles_gotas = np.count_nonzero(np.all(trinarizada == [255, 0, 0], axis=-1))

                    df = pd.DataFrame({
                        '': ['Hojas', 'Gotas'],
                        '#': [num_pixeles_hoja, num_pixeles_gotas],
                        '%': [round(num_pixeles_hoja/total_pixeles*100, 2), round(num_pixeles_gotas/num_pixeles_hoja*100, 2)],
                    })
                    
                    col1,col2=st.columns([1,2])
                    with col1:
                        st.image(trinarizada, width=333)
                        st.markdown(ocultar_fs, unsafe_allow_html=True)
                    with col2:
                        st.write(df)
                    
                    if st.session_state.trinarizada is not None and st.button("Guardar imagen trinarizada y csv"):
                        trinarizada = st.session_state.trinarizada
                        nombre_imagen = f"{nombre}.png"
                        ruta_imagen = os.path.join('imagenes_trinarizadas', nombre_imagen)
                        cv2.imwrite(ruta_imagen, cv2.cvtColor(trinarizada, cv2.COLOR_RGB2BGR))
                        
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
                        
                        ruta_csv_porc = os.path.join('imagenes_trinarizadas', f'{nombre}_porcentajes.csv')
                        df.to_csv(ruta_csv_porc, index=False)
                        
                        st.success('Archivos exportados correctamente')
                    if st.session_state.trinarizada is not None and st.button("Cargar trinarizada"):
                        trinarizada = st.session_state.trinarizada
                        fecha = datetime.datetime.now()
                        st.session_state.trinarizadas_cargadas[fecha]=trinarizada
                        st.rerun()
                        
        elif len(archivos_subidos) > 2:
            st.warning("Sólo debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")
        elif len(archivos_subidos) == 1:
            st.warning("Debes subir dos archivos, el .bil y el .bil.hdr de la misma imagen.")

    def mostrar_imagenes_trinarizadas():
        imagenes = st.session_state.trinarizadas_cargadas
        
        if imagenes:
            st.write("Trinarizadas cargadas:")
            for imagen in imagenes:
                st.write(imagen)
        else:
            st.write("No hay imágenes trinarizadas disponibles.")

    col1, col2 = st.columns([4,1])

    with col1:
        nombre = st.text_input("Ingresa el nombre")
        cargar_hyper_bin()
    
    with col2:
        mostrar_imagenes_trinarizadas()

app.run()