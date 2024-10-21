from ultralytics import YOLO
from io import BytesIO
import cv2
import PIL
import numpy as np
from matplotlib.cm import get_cmap
from typing import List, Dict

def load_model(model_path: str) -> YOLO:
    """
    Carga un modelo de detección de objetos YOLO desde la ruta especificada.

    Args:
        model_path (str): La ruta al archivo del modelo YOLO.

    Returns:
        YOLO: El modelo de detección de objetos YOLO cargado.
    """
    return YOLO(model_path)

def get_image_download_buffer(img_array: np.ndarray) -> BytesIO:
    """
    Convierte un array de imagen en un buffer de bytes descargable en formato JPEG.

    Args:
        img_array (numpy.ndarray): El array de la imagen a convertir.

    Returns:
        BytesIO: Un buffer de bytes de la imagen en formato JPEG, listo para descargar.
    """
    # Convierte el array de la imagen en una imagen PIL
    img = PIL.Image.fromarray(img_array)

    # Crea un objeto BytesIO que actuará como un buffer en memoria para la imagen
    buffered = BytesIO()

    # Guarda la imagen en el buffer en formato JPEG
    img.save(buffered, format="JPEG")

    # Retorna el puntero del buffer al inicio para asegurar que se pueda leer desde el principio
    buffered.seek(0)
    return buffered

def draw_bounding_boxes(image, results, classes):
    """
    Dibuja los cuadros delimitadores en la imagen según los resultados de la predicción.

    Args:
        image (numpy.ndarray): La imagen original en formato RGB en la que se dibujarán los cuadros.
        results (List): Resultados de la predicción del modelo YOLO, incluyendo las cajas, clases y confianza.
        classes (Dict[int, str]): Diccionario que mapea las clases originales (int) a nuevas etiquetas (str).

    Returns:
        numpy.ndarray: La imagen con los cuadros delimitadores dibujados, en formato RGB.
    """
    # Obtener un colormap de 'Set1'
    cmap = get_cmap('Set1')

    # Convertir la imagen de RGB a BGR (porque OpenCV usa BGR)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Iterar sobre cada resultado de la predicción
    for result in results:
        boxes = result.boxes  # Cuadros delimitadores
        for i, box in enumerate(boxes):
            # Obtener las coordenadas del cuadro delimitador
            xmin, ymin, xmax, ymax = box.xyxy[0]  
            # Obtener la confianza del modelo
            conf = box.conf[0]
            # Obtener la clase original del objeto
            cls = int(box.cls[0])
            
            # Obtener un color del colormap normalizado para este cuadro
            cmap_color = cmap(i / len(boxes))  
            color = (int(cmap_color[2] * 255), int(cmap_color[1] * 255), int(cmap_color[0] * 255))  # Convertir a BGR

            # Obtener la nueva etiqueta de clase mapeada desde el diccionario
            new_label = classes.get(cls, 'None')  # Si no existe la clase, asignar 'None'

            # Definir el texto con la etiqueta y la confianza
            label = f'{new_label} {conf:.2f}'
            font_scale = 1.1  # Escala del texto
            font_thickness = 2  # Grosor del texto
            baseline = 3  # Margen inferior para ajustar el texto

            # Medir el tamaño del texto para dibujar el fondo del rectángulo de la etiqueta
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Dibujar un rectángulo de fondo para el texto
            cv2.rectangle(image, (int(xmin), int(ymin) - text_height - baseline), 
                        (int(xmin) + text_width, int(ymin)), color, 3)
            cv2.rectangle(image, (int(xmin), int(ymin) - text_height - baseline), 
                        (int(xmin) + text_width, int(ymin)), color, cv2.FILLED)

            # Dibujar el cuadro delimitador del objeto
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 3)

            # Dibujar el texto (etiqueta + confianza) en la imagen
            cv2.putText(image, label, (int(xmin), int(ymin) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), font_thickness)

    # Convertir la imagen de nuevo a RGB antes de devolverla
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)