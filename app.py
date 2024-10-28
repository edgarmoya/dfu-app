import streamlit as st
from helper import load_model, get_image_download_buffer, draw_bounding_boxes
from pathlib import Path
import PIL
import settings
import zipfile
import io
import csv
from typing import List, Dict, Any

def clear_session() -> None:
    """Limpia las imágenes cargadas y procesadas del estado de sesión de Streamlit.

    Esta función restablece las claves 'uploaded_images' y 'processed_images' en 
    el estado de sesión, eliminando cualquier dato de imagen que haya sido subido o procesado.
    """
    if 'uploaded_images' in st.session_state:
        st.session_state.uploaded_images = []   # Limpiar imágenes cargadas
    if 'processed_images' in st.session_state:
        st.session_state.processed_images = []  # Limpiar imágenes procesadas

def write_csv(processed_images: List[Dict[str, Any]]) -> str:
    """Genera un archivo CSV con las coordenadas de las cajas delimitadoras de las imágenes procesadas.

    Args:
        processed_images (List[Dict[str, Any]]): Lista de diccionarios donde cada diccionario
            contiene el nombre de archivo y las cajas delimitadoras de una imagen procesada. 

    Returns:
        str: El contenido del archivo CSV como una cadena de texto, con el nombre del archivo 
        y las coordenadas de las cajas delimitadoras (xmin, ymin, xmax, ymax) para cada objeto detectado.
    """
    # Crear un archivo CSV en memoria para almacenar las coordenadas
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    # Escribir la cabecera del CSV
    csv_writer.writerow(['filename', 'xmin', 'ymin', 'xmax', 'ymax'])

    for img in processed_images:
        # Procesar cada caja delimitadora y escribir las coordenadas redondeadas
        for box in img['boxes']:
            xmin, ymin, xmax, ymax = [round(coord.item(), 2) for coord in box.xyxy[0]]
            csv_writer.writerow([img['filename'], xmin, ymin, xmax, ymax])

    # Devolver el contenido del CSV como una cadena de texto
    return csv_buffer.getvalue()

def initialize_session() -> None:
    """Inicializa el estado de la sesión de Streamlit."""
    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = []
    if 'processed_images' not in st.session_state:
        st.session_state.processed_images = []
    if 'confidence' not in st.session_state:
        st.session_state.confidence = 30  # Valor inicial de confianza

def process_images(model, confidence: float, iou_thres: float) -> None:
    """Realiza la detección en las imágenes subidas utilizando el modelo de detección."""
    for image in st.session_state.uploaded_images:
        uploaded_image = PIL.Image.open(image)
        res = model.predict(uploaded_image, conf=confidence, iou=iou_thres)  # Realiza la detección utilizando el modelo
        bboxes = res[0].boxes
        processed_image = draw_bounding_boxes(uploaded_image, res, {0: 'UPD'})

        # Almacena la imagen procesada y las cajas en el estado de la sesión
        st.session_state.processed_images.append({
            'image': processed_image,
            'filename': image.name,
            'boxes': bboxes
        })

def export_results(processed_images: List[Dict[str, Any]]) -> None:
    """Permite descargar las imágenes procesadas y las anotaciones en formato CSV en un archivo ZIP.

    Args:
        processed_images (List[Dict[str, Any]]): Lista de diccionarios donde cada diccionario
            contiene el nombre de archivo y las cajas delimitadoras de una imagen procesada.
    """
    # Crear un archivo ZIP en memoria
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for processed in processed_images:
            # Guardar cada imagen procesada en el ZIP
            img_buffer = get_image_download_buffer(processed['image']).getvalue()
            zip_file.writestr(processed['filename'], img_buffer)

        # Agregar el archivo CSV al ZIP
        zip_file.writestr('anotaciones.csv', write_csv(processed_images))

    # Preparar el archivo ZIP para la descarga
    zip_buffer.seek(0)  # Volver al inicio del buffer
    zip_data = zip_buffer.getvalue()  # Convertir a bytes

    # Agrega un botón para descarga la imagen
    try:
        st.sidebar.download_button(
            use_container_width=True,
            help='Exportar imágenes procesadas y anotaciones',
            label="Exportar",
            data=zip_data,
            file_name="upd.zip",
            mime="application/zip"
        )
    except Exception as ex:
        st.error("¡No se ha subido ninguna imagen aún!")
        st.error(ex)

if __name__ == '__main__':
    # NMS
    iou_thres = 0.5

    # Configuración del diseño de la página
    st.set_page_config(
        page_title="UPD",
        page_icon="🦶",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Título de la página principal
    st.title("Detección de UPD")

    #Inicializar estado de la sesión
    initialize_session()

    # Título de la barra lateral
    st.sidebar.header("Configuración del modelo")

    # Control deslizante para la confianza del modelo
    confidence = st.sidebar.slider( 
        label="Seleccionar confianza de detección",
        min_value=0,
        max_value=100, 
        value=st.session_state.confidence,
        help='Probabilidad de certeza en la detección de la úlcera'
    )

    # Revisa si ha cambiado el valor y ejecuta clear_session
    if confidence != st.session_state.confidence:
        st.session_state.confidence = confidence
        clear_session()

    # Cargador de archivos para seleccionar imágenes
    source_imgs = st.sidebar.file_uploader(
        label="Seleccionar una imagen", 
        help='Imagen del pie que desea analizar', 
        type=("jpg", "jpeg", "png"), 
        accept_multiple_files=True)

    # Botón para analizar las imágenes, mostrar solo cuando se carguen las imágenes
    if len(source_imgs) != 0:
        text_btn = 'Analizar imágenes' if len(source_imgs) > 1 else 'Analizar imagen'
        detect_button = st.sidebar.button(  # Botón para iniciar la detección
            label=text_btn, 
            use_container_width=True,
            help='Iniciar procesamiento de las imágenes cargadas')

    # Ruta del modelo de detección
    detection_model_path = Path(settings.DETECTION_MODEL)

    # Cargar el modelo
    try:
        model = load_model(detection_model_path)
    except Exception as ex:
        st.error(f"No se pudo cargar el modelo. Verifique la ruta especificada: {detection_model_path}")
        st.error(ex)

    # Verificar si la imagen original ha cambiado
    if 'uploaded_images' in st.session_state:
        if source_imgs is not None and st.session_state.uploaded_images != source_imgs:
            clear_session()  # Limpia el estado de la sesión

    if len(source_imgs) != 0:
        st.session_state.uploaded_images = source_imgs

        # Usar un selector para elegir la imagen a mostrar
        if len(st.session_state.uploaded_images) > 1:
            image_filenames = [img.name for img in st.session_state.uploaded_images]
            selected_image = st.selectbox("Selecciona la imagen que desea visualizar:", image_filenames)

            # Mostrar la imagen original correspondiente
            original_image_index = image_filenames.index(selected_image)
            source_img = source_imgs[original_image_index]
        else:
            selected_image = source_imgs[0].name
            source_img = source_imgs[0]

        col1, col2 = st.columns(2)   # Crear dos columnas

        # Crear columnas para mostrar las imágenes
        with col1:
            try:
                # Abrir y mostrar la imagen subida por el usuario
                st.image(source_img, caption="Imagen original", use_column_width='auto')
            except Exception as ex:
                st.error("Ocurrió un error al abrir la imagen.")
                st.error(ex)

        with col2:
            # Procesar imágenes al presionar el botón
            if detect_button:
                process_images(model=model, 
                            confidence=st.session_state.confidence/100, 
                            iou_thres=iou_thres)

            # Mostrar imágenes procesadas
            for processed in st.session_state.processed_images:
                if processed['filename'] == selected_image:
                    st.image(processed['image'], caption='Ulceraciones detectadas', use_column_width='auto')

            # Si las imágenes procesadas presentan úlceras mostrar botón para exportar
            if len(st.session_state.processed_images) == len(st.session_state.uploaded_images):  # Verificar que se procesen todas
                # Verifica si alguna imagen procesada tiene cajas
                if any(len(p['boxes']) > 0 for p in st.session_state.processed_images):
                    export_results(st.session_state.processed_images)
                else:
                    st.info('No se han detectado ulceraciones', icon="ℹ️")
    else:
        camera_svg = '''
            <svg xmlns="http://www.w3.org/2000/svg" fill="gray" viewBox="0 0 24 24" width="24" height="24">
                <circle cx="16" cy="8.011" r="2.5"/><path d="M23,16a1,1,0,0,0-1,1v2a3,3,0,0,1-3,3H17a1,1,0,0,0,0,2h2a5.006,5.006,0,0,0,5-5V17A1,1,0,0,0,23,16Z"/><path d="M1,8A1,1,0,0,0,2,7V5A3,3,0,0,1,5,2H7A1,1,0,0,0,7,0H5A5.006,5.006,0,0,0,0,5V7A1,1,0,0,0,1,8Z"/><path d="M7,22H5a3,3,0,0,1-3-3V17a1,1,0,0,0-2,0v2a5.006,5.006,0,0,0,5,5H7a1,1,0,0,0,0-2Z"/><path d="M19,0H17a1,1,0,0,0,0,2h2a3,3,0,0,1,3,3V7a1,1,0,0,0,2,0V5A5.006,5.006,0,0,0,19,0Z"/><path d="M18.707,17.293,11.121,9.707a3,3,0,0,0-4.242,0L4.586,12A2,2,0,0,0,4,13.414V16a3,3,0,0,0,3,3H18a1,1,0,0,0,.707-1.707Z"/>
            </svg>'''

        # Mostrar marco temporal hasta que no se seleccionen las imágenes
        with st.container(border=True):
            st.markdown(
                f"<div style='font-size: 16px; display: flex; justify-content: center; align-items: center; padding: 0 0 10px 0; gap: 15px; border-radius: 8px;'>"
                    f"{camera_svg}"
                    "No ha seleccionado una imagen para su procesamiento"
                "</div>",
                unsafe_allow_html=True
            )