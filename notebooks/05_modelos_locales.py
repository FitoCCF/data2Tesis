#!/usr/bin/env python3
# ============================================================
# CELDA 5 — MODELOS LOCALES (una regresión de leyes por cluster)
# ============================================================
# Entrena un modelo de regresión por cada (ley, cluster) y construye la TABLA
# DE RUTEO (local vs global). Delega en src.pipeline.modelos, que recomputa
# ortho y cluster con los artefactos CONGELADOS de las celdas 2-4, para que
# entrenamiento (aquí) e inferencia (celda 6) coincidan exactamente.
#
# Requiere que ya exista data/processed/intensidad_cobre_24_completo_filtrado.csv,
# generado por la fusión con las leyes de laboratorio
# (python -m src.pipeline.dataset_supervisado, ver README).
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd
import joblib

from src.pipeline.config import DATA_PROCESSED, MODELS_DIR, ART_ORTHO, ART_POWER, ART_SCALER, ART_GMM, ART_REGRESION, TARGETS, MODELO_POR_TARGET
from src.pipeline.modelos import entrenar_modelos_locales

# --- 1. Cargar dataset de calibración (intensidades CRUDAS + leyes de lab) ---
file_path = DATA_PROCESSED / 'intensidad_cobre_24_completo_filtrado.csv'
df = pd.read_csv(file_path)

# --- Cargar las transformaciones YA GUARDADAS por las celdas 2, 3 y 4 ---
regresiones_ortho = joblib.load(MODELS_DIR / ART_ORTHO)
power = joblib.load(MODELS_DIR / ART_POWER)
scaler_cluster = joblib.load(MODELS_DIR / ART_SCALER)
gmm = joblib.load(MODELS_DIR / ART_GMM)

# --- 2. Entrenar modelos locales + globales + tabla de ruteo (etapa 5) ---
bundle = entrenar_modelos_locales(df, regresiones_ortho, power, scaler_cluster, gmm)

for tgt in TARGETS:
    r2g = bundle['r2_global_por_ley'][tgt]
    print(f"{tgt}: R2 global={r2g:.3f}  ({MODELO_POR_TARGET[tgt]})")

print('\nTabla de ruteo (local vs global) por (ley, cluster):')
for (tgt, cl), decision in sorted(bundle['tabla_ruteo'].items()):
    print(f'  {tgt} / cluster {cl}: {decision}')

# --- 3. Guardar modelos + tabla de ruteo en el bundle ---
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, MODELS_DIR / ART_REGRESION)
print(f'\nModelos locales + tabla de ruteo guardados en: {MODELS_DIR / ART_REGRESION}')
