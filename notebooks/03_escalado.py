#!/usr/bin/env python3
# ============================================================
# CELDA 3 — ESCALADO (corrección de asimetría + estandarización)
# ============================================================
# Delega en src.pipeline.escalado.escalar().
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd
import joblib

from src.pipeline.config import DATA_PROCESSED, MODELS_DIR, ART_POWER, ART_SCALER, FEATS_CLUSTER
from src.pipeline.escalado import escalar

# --- 1. Cargar el dataset ortogonalizado (salida de la Celda 2) ---
input_path = DATA_PROCESSED / 'intensidad_cobre_24_orthogonalized.csv'
df = pd.read_csv(input_path)

# --- 2. Escalar (etapa 3) ---
X_scaled, power, scaler = escalar(df)

df_scaled = df.copy()
df_scaled[FEATS_CLUSTER] = X_scaled

# --- 3. Guardar el dataset escalado ---
output_path = DATA_PROCESSED / 'intensidad_cobre_24_scaled.csv'
df_scaled.to_csv(output_path, index=False)

# --- Guardar AMBOS transformadores (orden importa en tiempo real) ---
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(power, MODELS_DIR / ART_POWER)
joblib.dump(scaler, MODELS_DIR / ART_SCALER)

print('Escalado listo:', output_path)
