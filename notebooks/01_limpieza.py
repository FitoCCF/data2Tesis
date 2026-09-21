#!/usr/bin/env python3
# ============================================================
# CELDA 1 — LIMPIEZA
# ============================================================
# A diferencia de V1 (data2Tesis), esta celda NO reimplementa la lógica: llama
# a src.pipeline.limpieza.limpiar(), que es la MISMA función que usa
# src/pipeline/inferencia.py en producción. Un solo lugar donde vive la lógica.
#
# Entrada = intensidades_cobre_TRAIN.csv (split de scripts/merge_cobre_data.py
# --train-hasta), NO intensidades_cobre.csv (el archivo completo, sin dividir)
# ni el _test.csv (ese es solo un subconjunto de VALIDACIÓN dentro del mismo
# período histórico, no se re-entrena con él).
#
# Los datos de PRODUCCIÓN (p.ej. septiembre en adelante) NO son el _test.csv:
# son un extracto aparte (ver notebooks/00_getdata.py --desde/--hasta) que
# nunca pasa por esta etapa -> se scorea directo con la etapa 6
# (src.pipeline.inferencia / EstimadorCluster / EstimadorHibrido) usando los
# artefactos ya congelados aquí. Ver "Flujo train/test vs. producción" en el README.
# ============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # independiente del CWD

import pandas as pd
import joblib

from src.pipeline.config import DATA_PROCESSED, MODELS_DIR, ART_ANOMALIAS
from src.pipeline.limpieza import limpiar

# --- Carga de datos (split de calibración, no el archivo completo) ---
file_path = DATA_PROCESSED / 'intensidades_cobre_train.csv'
xx = pd.read_csv(file_path)

# --- Limpieza + detección de anomalías (etapa 1) ---
df_clean, detector = limpiar(xx)

# --- Guardar el CSV limpio ---
output_path = DATA_PROCESSED / 'intensidad_cobre_24_clean.csv'
df_clean.to_csv(output_path, index=False)

# --- Guardar el modelo detector de anomalías ---
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(detector, MODELS_DIR / ART_ANOMALIAS)

print(f'Filas finales tras limpieza: {len(df_clean)}')
