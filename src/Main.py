import streamlit as st

st.title('Cargar Imágenes')

archivo_cargado = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "tiff", "tif"])

if archivo_cargado is not None:
    st.image(archivo_cargado, caption='Imagen cargada.', use_column_width=True)