# Paquete pipeline: cada etapa vive en su módulo y se importa desde ahí
# (p.ej. `from src.pipeline.limpieza import limpiar`). No se re-exporta nada
# a nivel de paquete a propósito: así una etapa en revisión no rompe la
# importación de las demás.
from . import config
