import streamlit as st
from pathlib import Path
import PIL
import settings
from helper import load_model, get_image_download_buffer, draw_bounding_boxes

# Configuración del diseño de la página
st.set_page_config(
    page_title="UPD",
    page_icon="🦶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Barra lateral
st.sidebar.header("Configuración del modelo")

# Opciones del modelo
confidence = float(st.sidebar.slider(
    "Seleccionar confianza de detección", 0, 100, 30, help='Probabilidad de certeza en la detección de la úlcera')) / 100  # Control deslizante para la confianza del modelo

# NMS
iou_thres = 0.5

# Cargador de archivos para seleccionar imágenes
source_img = st.sidebar.file_uploader(
    label="Seleccionar una imagen", 
    help='Imagen del pie que desea analizar', 
    type=("jpg", "jpeg", "png"), 
    accept_multiple_files=False)

# Título de la página principal
st.title("Detección de UPD")

# Ruta del modelo de detección
detection_model_path = Path(settings.DETECTION_MODEL)

# Cargar el modelo
try:
    model = load_model(detection_model_path)
except Exception as ex:
    st.error(f"No se pudo cargar el modelo. Verifique la ruta especificada: {detection_model_path}")
    st.error(ex)

# Verificar si la imagen original ha cambiado
if 'uploaded_image' in st.session_state:
    if source_img is not None and st.session_state.uploaded_image != source_img:
        # st.session_state.clear()  # Limpia el estado de la sesión
        if 'res_plotted' in st.session_state:
            del st.session_state.res_plotted
        if 'boxes' in st.session_state:
            del st.session_state.boxes

if source_img is not None:
    st.session_state.uploaded_image = source_img

    # Crear dos columnas
    col1, col2 = st.columns(2)

    with col1:
        try:
            # Abrir y mostrar la imagen subida por el usuario
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Imagen original", use_column_width='auto')
        except Exception as ex:
            st.error("Ocurrió un error al abrir la imagen.")
            st.error(ex)

    with col2:
        detect_button = st.sidebar.button('Analizar imagen', use_container_width=True)  # Botón para iniciar la detección

        if 'res_plotted' not in st.session_state and detect_button:  # Verifica si la imagen detectada no está en el estado
            res = model.predict(uploaded_image, conf=confidence, iou=iou_thres)  # Realiza la detección utilizando el modelo
            st.session_state.boxes = res[0].boxes  # Almacena las cajas detectadas en el estado de la sesión
            st.session_state.res_plotted = draw_bounding_boxes(uploaded_image, res, {0: 'UPD'})

        if 'res_plotted' in st.session_state:  # Verifica si hay una imagen procesada
            st.image(st.session_state.res_plotted, caption='Ulceraciones detectadas', use_column_width='auto')  # Muestra la imagen procesada

            if st.session_state.boxes:  # Verifica si hay cajas detectadas
                try:
                    # Agrega un botón para descarga la imagen  
                    st.download_button(
                        use_container_width=True,
                        label="Descargar imagen",
                        data=get_image_download_buffer(st.session_state.res_plotted),  # Convierte la imagen a un buffer descargable
                        file_name=f"det_{source_img.name}",
                        mime="image/jpeg"
                    )
                except Exception as ex:
                    st.error("¡No se ha subido ninguna imagen aún!")
                    st.error(ex)
            else:
                st.info('No se han detectado ulceraciones', icon="ℹ️")
else:
    camera_svg = '''
        <svg xmlns="http://www.w3.org/2000/svg" fill="gray" viewBox="0 0 24 24" width="24" height="24">
            <circle cx="16" cy="8.011" r="2.5"/><path d="M23,16a1,1,0,0,0-1,1v2a3,3,0,0,1-3,3H17a1,1,0,0,0,0,2h2a5.006,5.006,0,0,0,5-5V17A1,1,0,0,0,23,16Z"/><path d="M1,8A1,1,0,0,0,2,7V5A3,3,0,0,1,5,2H7A1,1,0,0,0,7,0H5A5.006,5.006,0,0,0,0,5V7A1,1,0,0,0,1,8Z"/><path d="M7,22H5a3,3,0,0,1-3-3V17a1,1,0,0,0-2,0v2a5.006,5.006,0,0,0,5,5H7a1,1,0,0,0,0-2Z"/><path d="M19,0H17a1,1,0,0,0,0,2h2a3,3,0,0,1,3,3V7a1,1,0,0,0,2,0V5A5.006,5.006,0,0,0,19,0Z"/><path d="M18.707,17.293,11.121,9.707a3,3,0,0,0-4.242,0L4.586,12A2,2,0,0,0,4,13.414V16a3,3,0,0,0,3,3H18a1,1,0,0,0,.707-1.707Z"/>
        </svg>'''

    with st.container(border=True):
        st.markdown(
            f"<div style='font-size: 16px; display: flex; justify-content: center; align-items: center; padding: 0 0 10px 0; gap: 15px; border-radius: 8px;'>"
                f"{camera_svg}"
                "No ha seleccionado una imagen para su procesamiento"
            "</div>",
            unsafe_allow_html=True
        )
