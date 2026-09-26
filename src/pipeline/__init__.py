# Paquete pipeline: expone las etapas del flujo de estimación de leyes.
from . import config
from .features import features_cierre, features_regresion, columnas_regresion
from .limpieza import limpiar
from .ortogonalizacion import ortogonalizar, solo_features
from .escalado import escalar
from .clustering import entrenar_clustering, asignar_cluster, diagnostico_k
from .modelos import entrenar_modelos_locales
from .inferencia import EstimadorHibrido, EstimadorCluster
from .dataset_supervisado import construir_dataset_supervisado
from .calibracion_composito import (cargar_composito, cargar_intensidades,
                                    alinear_con_composito, CorrectorSesgo,
                                    entrenar as entrenar_composito,
                                    predecir as predecir_composito,
                                    backtest, metricas)
