from ultralytics import YOLO
from io import BytesIO
import base64
import PIL
import svgutils.transform as sg

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
    Parámetros:
        img_array (numpy.ndarray): El array de la imagen a convertir.
    Retorna:
        BytesIO: Un buffer de bytes de la imagen en formato JPEG.
    """
    img = PIL.Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    buffered.seek(0)
    return buffered