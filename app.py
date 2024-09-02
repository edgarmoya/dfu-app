from pathlib import Path
import pandas as pd
import PIL
import streamlit as st
import settings
from helper import load_model, get_image_download_buffer

# Configuración del diseño de la página
st.set_page_config(
    page_title="Detección de DFU",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Barra lateral
st.sidebar.header("Configuración del modelo")

# Crear un grupo de botones de opción
option = st.sidebar.selectbox(
    "Seleccione un modelo",
    ('Yolov8n', 'Yolov8x')
)

# Ruta del modelo de detección
model_path = ''
if (option == 'Yolov8n'):
    model_path = Path(settings.DETECTION_MODEL_YOLOV8N)
if (option == 'Yolov8x'):
    model_path = Path(settings.DETECTION_MODEL_YOLOV8X)

# Cargar el modelo preentrenado
try:
    model = load_model(model_path)
except Exception as ex:
    st.error(f"No se pudo cargar el modelo. Verifique la ruta especificada: {model_path}")
    st.error(ex)

# Título de la página principal
st.title("Detección de DFU")

# Opciones del modelo
confidence = float(st.sidebar.slider(
    "Seleccionar confianza del modelo", 25, 100, 40)) / 100  # Control deslizante para la confianza del modelo

# Cargador de archivos para seleccionar imágenes
source_img = st.sidebar.file_uploader(
    "Elige una imagen", type=("jpg", "jpeg", "png", 'bmp'))

if source_img is not None:
    st.session_state.clear()  # Limpia el estado de la sesión

    # Crear dos columnas
    col1, col2 = st.columns(2)

    with col1:
        try:
            # Abrir y mostrar la imagen subida por el usuario
            uploaded_image = PIL.Image.open(source_img)
            st.image(source_img, caption="Imagen original",
                    use_column_width=True)
        except Exception as ex:
            st.error("Ocurrió un error al abrir la imagen.")
            st.error(ex)

    with col2:
        detect_button = st.sidebar.button('Detectar DFU')  # Botón para iniciar la detección
        if 'res_plotted' not in st.session_state and detect_button:  # Verifica si la imagen detectada no está en el estado
            res = model.predict(uploaded_image, conf=confidence)  # Realiza la detección utilizando el modelo
            st.session_state.boxes = res[0].boxes  # Almacena las cajas detectadas en el estado de la sesión
            st.session_state.res_plotted = res[0].plot()[:, :, ::-1]  # Almacena la imagen procesada

        if 'res_plotted' in st.session_state:  # Verifica si hay una imagen procesada
            st.image(st.session_state.res_plotted, caption='Ulceraciones detectadas',
                    use_column_width=True)  # Muestra la imagen procesada

            if st.session_state.boxes:  # Verifica si hay cajas detectadas
                try:
                    # Expande para mostrar los resultados de las cajas detectadas
                    with st.expander("Resultados de la detección"):
                        all_data = []  # Crear una lista vacía para almacenar laas coordenadas

                        # Iterar sobre cada caja y agregar a la lista
                        for box in st.session_state.boxes:
                            all_data.extend(box.data.tolist())

                        # Crear un DataFrame de pandas a partir de la lista de datos
                        df = pd.DataFrame(all_data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])

                        # Aplicar formato de dos decimales a las columnas flotantes
                        df['xmin'] = df['xmin'].map(lambda x: f"{x:.2f}")
                        df['ymin'] = df['ymin'].map(lambda x: f"{x:.2f}")
                        df['xmax'] = df['xmax'].map(lambda x: f"{x:.2f}")
                        df['ymax'] = df['ymax'].map(lambda x: f"{x:.2f}")
                        df['confidence'] = df['confidence'].map(lambda x: f"{x:.2f}")

                        # Mostrar la tabla en Streamlit con las conversiones aplicadas
                        st.table(df[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']])

                    # Agrega un botón para descarga la imagen
                    buffered = get_image_download_buffer(st.session_state.res_plotted)  # Convierte la imagen a un buffer descargable
                    st.download_button(
                        label="Descargar imagen",
                        data=buffered,
                        file_name=f"detected_{source_img.name}",
                        mime="image/jpeg"
                    )
                except Exception as ex:
                    st.error("¡No se ha subido ninguna imagen aún!")
                    st.error(ex)
            else:
                st.markdown(
                "<div style='background-color: #f0f2f8; font-size: 18px; display: flex; justify-content: center; align-items: center; padding: 12px 0; gap: 15px; border-radius: 8px;'>"
                    "No se han detectado ulceraciones"
                "</div>",
                unsafe_allow_html=True
    )
else:
    svg_code = '''
        <svg xmlns="http://www.w3.org/2000/svg" fill="gray" viewBox="0 0 24 24" width="30" height="30">
            <circle cx="16" cy="8.011" r="2.5"/><path d="M23,16a1,1,0,0,0-1,1v2a3,3,0,0,1-3,3H17a1,1,0,0,0,0,2h2a5.006,5.006,0,0,0,5-5V17A1,1,0,0,0,23,16Z"/><path d="M1,8A1,1,0,0,0,2,7V5A3,3,0,0,1,5,2H7A1,1,0,0,0,7,0H5A5.006,5.006,0,0,0,0,5V7A1,1,0,0,0,1,8Z"/><path d="M7,22H5a3,3,0,0,1-3-3V17a1,1,0,0,0-2,0v2a5.006,5.006,0,0,0,5,5H7a1,1,0,0,0,0-2Z"/><path d="M19,0H17a1,1,0,0,0,0,2h2a3,3,0,0,1,3,3V7a1,1,0,0,0,2,0V5A5.006,5.006,0,0,0,19,0Z"/><path d="M18.707,17.293,11.121,9.707a3,3,0,0,0-4.242,0L4.586,12A2,2,0,0,0,4,13.414V16a3,3,0,0,0,3,3H18a1,1,0,0,0,.707-1.707Z"/>
        </svg>
    '''

    st.markdown(
        "<div style='background-color: #f0f2f8; font-size: 18px; display: flex; justify-content: center; align-items: center; padding: 12px 0; gap: 15px; border-radius: 8px;'>"
            f"{svg_code}"
            "No ha seleccionado una imagen para su procesamiento"
        "</div>",
        unsafe_allow_html=True
    )
