# Paquete pipeline: expone las etapas del flujo de estimación de leyes.
from . import config
from .limpieza import limpiar
from .ortogonalizacion import ortogonalizar, solo_features
from .escalado import escalar
from .clustering import entrenar_clustering, asignar_cluster, diagnostico_k
from .modelos import entrenar_modelos_locales
from .inferencia import EstimadorHibrido, EstimadorCluster
from .dataset_supervisado import construir_dataset_supervisado
