from pathlib import Path
import sys

# Obtiene la ruta absoluta del archivo actual
FILE = Path(__file__).resolve()
# Obtiene el directorio padre del archivo actual
ROOT = FILE.parent
# Agrega la ruta raíz a la lista sys.path si no está ya presente
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Obtiene la ruta relativa del directorio raíz con respecto al directorio de trabajo actual
ROOT = ROOT.relative_to(Path.cwd())

# Directorio donde se almacenan las imágenes
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'office.jpg'  # Imagen predeterminada
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_detected.jpg'  # Imagen predeterminada con detección

# Directorio donde se almacenan los modelos
MODEL_DIR = ROOT / 'weights'  
DETECTION_MODEL = MODEL_DIR / 'best.pt'
