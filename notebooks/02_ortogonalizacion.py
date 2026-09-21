#!/usr/bin/env python3
# ============================================================
# CELDA 2 — ORTOGONALIZACIÓN (corrección por dilución vía n6sc)
# ============================================================
# Delega en src.pipeline.ortogonalizacion.ortogonalizar() (misma función que
# usa la inferencia en producción).
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd
import joblib

from src.pipeline.config import DATA_PROCESSED, MODELS_DIR, ART_ORTHO
from src.pipeline.ortogonalizacion import ortogonalizar

# --- 1. Cargar el dataset limpio (salida de la Celda 1) ---
file_path = DATA_PROCESSED / 'intensidad_cobre_24_clean.csv'
df = pd.read_csv(file_path)

# --- 2. Ortogonalizar (etapa 2) ---
df_ortho, regresiones = ortogonalizar(df)

# --- 3. Guardar el dataset ortogonalizado ---
output_path = DATA_PROCESSED / 'intensidad_cobre_24_orthogonalized.csv'
df_ortho.to_csv(output_path, index=False)

# --- Guardar los modelos de regresión de la ortogonalización ---
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(regresiones, MODELS_DIR / ART_ORTHO)

print(f'Metales ortogonalizados: {list(regresiones.keys())}')
