from ultralytics import YOLO
from io import BytesIO
import cv2
import PIL
import numpy as np
from matplotlib.cm import get_cmap

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.
    Parameters:
        model_path (str): The path to the YOLO model file.
    Returns:
        A YOLO object detection model.
    """
    return YOLO(model_path)

def get_image_download_buffer(img_array):
    """
    Convierte un array de imagen en un buffer de bytes descargable en formato JPEG.

    Args:
        img_array (numpy.ndarray): El array de la imagen a convertir.

    Return (BytesIO): Un buffer de bytes de la imagen en formato JPEG.
    """
    img = PIL.Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    buffered.seek(0)
    return buffered

def draw_bounding_boxes(image, results, classes):
    """
    Dibuja los cuadros delimitadores en la imagen según los resultados de la predicción.

    Args:
        image: La imagen original en la que se dibujarán los cuadros.
        results: Resultados de la predicción del modelo YOLO.
        classes: Diccionario que mapea las clases originales a nuevas etiquetas.

    Return: La imagen con los cuadros delimitadores dibujados.
    """
    # Obtener un colormap
    cmap = get_cmap('Set1')

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for result in results:
        boxes = result.boxes  # Cuadros delimitadores
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box.xyxy[0]  # Coordenadas del cuadro
            conf = box.conf[0]  # Confianza
            cls = int(box.cls[0])  # Clase original
            
            # Obtener un color del colormap
            cmap_color = cmap(i / len(boxes))  # Normalizar el índice para obtener un color
            color = (int(cmap_color[2] * 255), int(cmap_color[1] * 255), int(cmap_color[0] * 255))  # Convertir a BGR

            # Obtener la nueva etiqueta de clase
            new_label = classes.get(cls, 'None')  # 'None' si no hay mapeo

            # Definir el texto y su tamaño
            label = f'{new_label} {conf:.2f}'
            font_scale = 1.1  # Aumentar el tamaño del texto
            font_thickness = 2  # Grosor del texto
            baseline = 3

            # Medir el tamaño del texto para dibujar el fondo
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Dibuja un rectángulo azul como fondo para la etiqueta
            cv2.rectangle(image, (int(xmin), int(ymin) - text_height - baseline), (int(xmin) + text_width, int(ymin)), color, 3)
            cv2.rectangle(image, (int(xmin), int(ymin) - text_height - baseline), (int(xmin) + text_width, int(ymin)), color, cv2.FILLED)

            # Dibuja el cuadro delimitador del objeto
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 3)

            # Dibuja el texto en blanco
            cv2.putText(image, label, (int(xmin), int(ymin) - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)