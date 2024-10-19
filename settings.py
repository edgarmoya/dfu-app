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

# Directorio donde se almacenan los modelos
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL_YOLOV8N = MODEL_DIR / 'yolov8n.pt'
DETECTION_MODEL_YOLOV8M = MODEL_DIR / 'yolov8m.pt'
DETECTION_MODEL_YOLOV8L = MODEL_DIR / 'yolov8l.pt'
DETECTION_MODEL_YOLOV8X = MODEL_DIR / 'yolov8x.pt'
