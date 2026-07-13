# ============================================================
# src/pipeline/config.py — Configuración central del pipeline
# ============================================================
# Centraliza rutas, nombres de columnas y constantes para que ninguna
# etapa tenga valores "mágicos" dispersos. Un solo lugar que editar.
# ============================================================

from pathlib import Path                                  # manejo de rutas multiplataforma

# --- Raíz del proyecto (dos niveles arriba de este archivo: src/pipeline/config.py) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]        # .../data2Tesis
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"      # carpeta de datos procesados
DATA_RAW = PROJECT_ROOT / "data" / "raw"                  # carpeta de datos crudos
MODELS_DIR = PROJECT_ROOT / "models"                      # carpeta de artefactos .joblib

# --- Canales espectrales (n6sc = sólido en la muestra / dilución) ---
CANALES = ["n1fe", "n2cu", "n3zn", "n4mo", "n6sc"]        # los 5 canales de intensidad
METALES = ["n1fe", "n2cu", "n3zn", "n4mo"]                # canales que se ortogonalizan (sin n6sc)
FEATS_CLUSTER = ["n1fe_ortho", "n2cu_ortho",
                 "n3zn_ortho", "n4mo_ortho"]              # features del clustering (4 ortho, sin n6sc)
FEATS_REGRESION = ["n1fe_ortho", "n2cu_ortho",
                   "n3zn_ortho", "n4mo_ortho", "n6sc"]    # features de la regresión (ortho + n6sc)

# --- Leyes químicas a estimar y el modelo elegido para cada una ---
TARGETS = ["pFe", "pCu", "pMo", "pZn"]                    # leyes de laboratorio
MODELO_POR_TARGET = {"pFe": "Ridge", "pCu": "SVR",        # familia de modelo por ley
                     "pMo": "Ridge", "pZn": "PLS"}

# --- Semilla global para reproducibilidad ---
RANDOM_STATE = 42                                         # fija el azar en todo el pipeline

# --- Nombres de los artefactos serializados (una fuente de verdad) ---
ART_ANOMALIAS = "anomaly_detector.joblib"                # celda 1
ART_ORTHO = "orthogonalization_regressions.joblib"       # celda 2
ART_POWER = "power_transformer.joblib"                   # celda 3
ART_SCALER = "scaler.joblib"                             # celda 3
ART_KMEANS = "kmeans_model.joblib"                       # celda 4
ART_GMM = "gmm_model.joblib"                             # celda 4
ART_REGRESION = "modelos_locales_por_cluster.joblib"     # celda 5 (bundle + tabla de ruteo)
